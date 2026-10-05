"""
Оркестратор пайплайна MomentSnap.

Собирает всё вместе:
    1. Извлекает аудио из видео.
    2. Ищет всплески громкости (имитация значка).
    3. Режет 30-секундные клипы вокруг пиков в папку с датой.
    4. Удаляет временные файлы.
    5. Сохраняет отчёт в results.json.

Сам ничего не считает — только вызывает модули.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import config
from audio_utils import (
    extract_audio,
    find_peaks,
    cut_clip,
    get_video_duration,
)


def _get_video_date(video_path: Path) -> str:
    """
    Дата съёмки видео в формате YYYY-MM-DD.
    Берём из времени последнего изменения файла (mtime) —
    это когда видео было записано и скинуто.
    """
    mtime = video_path.stat().st_mtime
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")


def _prepare_dirs(video_path: Path) -> Path:
    """
    Создаёт все выходные папки.
    Возвращает путь к папке клипов для этого видео:
    clips/YYYY-MM-DD/
    """
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    date_str = _get_video_date(video_path)
    clips_dir = config.OUTPUT_DIR / "clips" / date_str
    clips_dir.mkdir(parents=True, exist_ok=True)

    return clips_dir


def _cleanup_temp() -> None:
    """
    Удаляет временный full_audio.wav после работы.
    Клипы и results.json НЕ трогает.
    """
    if not config.CLEANUP_TEMP_AFTER_RUN:
        return
    if config.FULL_WAV.exists():
        try:
            config.FULL_WAV.unlink()
            print(f"  Удалён временный файл: {config.FULL_WAV.name}")
        except Exception as e:
            print(f"  ⚠ Не удалось удалить {config.FULL_WAV.name}: {e}")


def _compute_clip_bounds(center: float, video_duration: float) -> tuple[float, float]:
    """
    По центру пика считает start/end клипа.

    Обычно: start = center - PRE_PEAK, end = center + POST_PEAK.
    Но если пик близко к началу или концу видео — сдвигаем окно внутрь,
    чтобы не вылезти за пределы файла.
    """
    start = max(0.0, center - config.PRE_PEAK)
    end = min(video_duration, center + config.POST_PEAK)

    # Если клип получился короче нужного — прижимаем к началу или концу
    if end - start < config.CLIP_DURATION:
        if start <= 0.01:
            end = min(video_duration, config.CLIP_DURATION)
        elif end >= video_duration - 0.01:
            start = max(0.0, video_duration - config.CLIP_DURATION)

    return start, end


def _process_clip(
    idx: int,
    total: int,
    center: float,
    video_duration: float,
    clips_dir: Path,
) -> dict:
    """
    Обрабатывает один клип: режет из исходного видео в clips_dir.

    Возвращает dict с результатом для записи в results.json.
    """
    start, end = _compute_clip_bounds(center, video_duration)

    clip_name = f"clip_{idx:02d}_center_{center:.1f}s.mp4"
    clip_path = clips_dir / clip_name

    print(f"\n[{idx}/{total}] Клип вокруг {center:.1f} сек")
    print(f"  Режу {start:.1f}–{end:.1f} сек -> {clip_name}")

    # Вырезать клип из исходного видео
    cut_clip(config.VIDEO_PATH, start, end, clip_path)

    return {
        "clip": clip_name,
        "start": round(start, 2),
        "end": round(end, 2),
        "center": round(center, 2),
    }


def run() -> None:
    """
    Главная функция. Запускает весь пайплайн.

    Вызывается из main.py — там только `from pipeline import run; run()`.
    """
    t_start = time.time()

    print("=" * 60)
    print("MomentSnap PoC — локальный тест")
    print("=" * 60)

    # Подготовка папок + определение даты
    clips_dir = _prepare_dirs(config.VIDEO_PATH)
    print(f"\nДата видео: {_get_video_date(config.VIDEO_PATH)}")
    print(f"Клипы будут в: {clips_dir}")

    # ---------- Шаг 1: извлечь аудио из полного видео ----------
    print("\n[1/3] Извлекаю аудио из видео...")
    extract_audio(config.VIDEO_PATH, config.FULL_WAV)

    # ---------- Шаг 2: найти пики громкости ----------
    print("\n[2/3] Ищу всплески громкости...")
    peaks = find_peaks(config.FULL_WAV)
    video_duration = get_video_duration(config.FULL_WAV)

    print(f"  Длительность видео: {video_duration:.1f} сек "
          f"({video_duration / 60:.1f} мин)")
    print(f"  Найдено пиков: {len(peaks)}")
    for p in peaks:
        print(f"    - {p:.1f} сек")

    if not peaks:
        print("\n⚠ Пиков не найдено. Попробуй понизить PEAK_PERCENTILE "
              "или MIN_RMS_THRESHOLD в config.py.")
        _cleanup_temp()
        return

    # ---------- Шаг 3: обработать каждый клип ----------
    print(f"\n[3/3] Режу {len(peaks)} клипов...")

    results = []
    for idx, center in enumerate(peaks, 1):
        try:
            result = _process_clip(
                idx=idx,
                total=len(peaks),
                center=center,
                video_duration=video_duration,
                clips_dir=clips_dir,
            )
            results.append(result)
        except Exception as e:
            # Один упавший клип не должен убивать весь пайплайн
            print(f"  ⚠ Ошибка на клипе {idx}: {e}")
            results.append({
                "clip": f"clip_{idx:02d}",
                "center": round(center, 2),
                "error": str(e),
            })

    # ---------- Очистка временных ----------
    print("\nОчистка временных файлов...")
    _cleanup_temp()

    # ---------- Итог ----------
    with open(config.RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    t_total = time.time() - t_start

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)
    print(f"  Всего клипов:  {len(results)}")
    print(f"  Время:         {t_total:.1f} сек")
    print(f"  Клипы:         {clips_dir}")
    print(f"  Отчёт:         {config.RESULTS_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    run()