"""
Оркестратор пайплайна MomentSnap.

Собирает всё вместе:
    1. Извлекает аудио из видео.
    2. Ищет всплески громкости (имитация значка).
    3. Режет 30-секундные клипы вокруг пиков.
    4. Транскрибирует каждый клип через Whisper.
    5. Сортирует каждый клип через LLM.
    6. Сохраняет результаты в results.json.

Сам ничего не считает — только вызывает модули.
"""
import json
import time
from pathlib import Path

import config
from audio_utils import (
    extract_audio,
    find_peaks,
    cut_clip,
    get_video_duration,
)
from classifier import classify
from transcriber import Transcriber


def _prepare_dirs() -> None:
    """
    Создаёт все выходные папки, если их нет.
    mkdir(parents=True, exist_ok=True) — создаст всю цепочку
    и не упадёт, если папка уже есть.
    """
    for d in (config.OUTPUT_DIR, config.CLIPS_DIR, config.AUDIO_DIR):
        d.mkdir(parents=True, exist_ok=True)


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
    transcriber: Transcriber,
) -> dict:
    """
    Обрабатывает один клип: режет, транскрибирует, классифицирует.

    Возвращает dict с результатом для записи в results.json.
    """
    start, end = _compute_clip_bounds(center, video_duration)

    clip_name = f"clip_{idx:02d}_center_{center:.1f}s.mp4"
    clip_path = config.CLIPS_DIR / clip_name
    clip_wav = config.AUDIO_DIR / f"clip_{idx:02d}.wav"

    print(f"\n[{idx}/{total}] Клип вокруг {center:.1f} сек")
    print(f"  Режу {start:.1f}–{end:.1f} сек -> {clip_name}")

    # Шаг 1: вырезать клип из исходного видео
    cut_clip(config.VIDEO_PATH, start, end, clip_path)

    # Шаг 2: вытащить из клипа аудио (Whisper ест только WAV)
    extract_audio(clip_path, clip_wav)

    # Шаг 3: транскрибировать
    print("  Транскрибирую...")
    t0 = time.time()
    transcript = transcriber.transcribe(clip_wav)
    t_transcribe = time.time() - t0

    preview = transcript[:200] + ("..." if len(transcript) > 200 else "")
    print(f"  Текст ({t_transcribe:.1f}с): {preview or '<пусто>'}")

    # Шаг 4: классифицировать через LLM
    print("  Сортирую через ИИ...")
    t0 = time.time()
    verdict = classify(transcript)
    t_classify = time.time() - t0

    keep_mark = "✅ keep" if verdict.get("keep") else "❌ delete"
    print(f"  Вердикт ({t_classify:.1f}с): {keep_mark} — {verdict.get('reason', '')}")

    return {
        "clip": clip_name,
        "start": round(start, 2),
        "end": round(end, 2),
        "center": round(center, 2),
        "transcript": transcript,
        "verdict": verdict,
        "timing": {
            "transcribe_sec": round(t_transcribe, 2),
            "classify_sec": round(t_classify, 2),
        },
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

    _prepare_dirs()

    # ---------- Шаг 1: извлечь аудио из полного видео ----------
    print("\n[1/4] Извлекаю аудио из видео...")
    extract_audio(config.VIDEO_PATH, config.FULL_WAV)

    # ---------- Шаг 2: найти пики громкости ----------
    print("\n[2/4] Ищу всплески громкости...")
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
        return

    # ---------- Шаг 3: загрузить Whisper ----------
    print("\n[3/4] Загружаю Whisper...")
    transcriber = Transcriber()

    # ---------- Шаг 4: обработать каждый клип ----------
    print(f"\n[4/4] Обрабатываю {len(peaks)} клипов...")

    results = []
    for idx, center in enumerate(peaks, 1):
        try:
            result = _process_clip(
                idx=idx,
                total=len(peaks),
                center=center,
                video_duration=video_duration,
                transcriber=transcriber,
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

    # ---------- Итог ----------
    keep_count = sum(1 for r in results if r.get("verdict", {}).get("keep"))
    total_count = len(results)

    with open(config.RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    t_total = time.time() - t_start

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)
    print(f"  Всего клипов:  {total_count}")
    print(f"  В архив (keep): {keep_count}")
    print(f"  В мусор:       {total_count - keep_count}")
    print(f"  Время:         {t_total:.1f} сек")
    print(f"  Результаты:    {config.RESULTS_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    # Позволяет запустить pipeline.py напрямую, без main.py
    run()