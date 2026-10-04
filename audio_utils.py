"""
Всё, что связано со звуком:
- вызов FFmpeg для извлечения/нарезки
- чтение WAV
- поиск всплесков громкости (имитация аудио-триггера значка)
"""
import subprocess
import wave
from pathlib import Path
from typing import List

import numpy as np

import config


def run_ffmpeg(cmd: List[str]) -> None:
    """
    Обёртка над subprocess для запуска FFmpeg.
    Печатает команду (удобно дебажить) и бросает исключение при ошибке.
    """
    print("FFmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_audio(video_path: Path, wav_path: Path) -> None:
    """
    Достаёт моно-дорожку 16 кГц из видео в WAV.
    16 кГц моно — стандарт для Whisper и экономит память.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                # без видео
        "-ac", "1",           # моно
        "-ar", "16000",       # 16 кГц
        "-f", "wav", str(wav_path)
    ]
    run_ffmpeg(cmd)


def read_wav_mono(wav_path: Path):
    """
    Читает WAV и возвращает (samples, sample_rate).
    Samples — float32 в диапазоне [-1, 1].
    """
    with wave.open(str(wav_path), "rb") as wf:
        fs = wf.getframerate()
        n = wf.getnframes()
        data = wf.readframes(n)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, fs


def find_peaks(wav_path: Path) -> List[float]:
    """
    Имитация аудио-фильтра значка.
    Считает RMS по окнам VOLUME_WINDOW секунд, порог — PEAK_PERCENTILE
    (верхние N% самых громких окон). Возвращает список таймкодов
    (секунды от начала) — центры всплесков громкости.
    Близкие пики склеиваются, чтобы не резать дубли.
    """
    samples, fs = read_wav_mono(wav_path)
    win = int(config.VOLUME_WINDOW * fs)
    if win <= 0 or len(samples) < win:
        return []

    # RMS по каждому окну
    rms = []
    for i in range(0, len(samples) - win, win):
        chunk = samples[i:i + win]
        rms.append(np.sqrt(np.mean(chunk ** 2)))
    rms = np.array(rms)
    if len(rms) == 0:
        return []

    # Порог = максимум из "статистического" и "абсолютного"
    threshold = max(np.percentile(rms, config.PEAK_PERCENTILE),
                    config.MIN_RMS_THRESHOLD)
    loud = rms > threshold

    # Ищем непрерывные "громкие" группы и берём пик внутри каждой
    events = []
    i = 0
    while i < len(loud):
        if loud[i]:
            j = i
            while j < len(loud) and loud[j]:
                j += 1
            group = rms[i:j]
            max_idx = i + int(np.argmax(group))
            center_time = max_idx * config.VOLUME_WINDOW + config.VOLUME_WINDOW / 2
            events.append((center_time, float(rms[max_idx])))
            i = j
        else:
            i += 1

    # Склейка близких пиков: оставляем самый громкий
    merged = []
    for t, amp in events:
        if not merged or t - merged[-1][0] > config.MIN_GAP_BETWEEN_PEAKS:
            merged.append([t, amp])
        elif amp > merged[-1][1]:
            merged[-1] = [t, amp]

    return [t for t, _ in merged]


def cut_clip(video_path: Path, start: float, end: float, out_path: Path) -> None:
    """
    Режет кусок видео [start, end] и сохраняет как отдельный mp4
    с перекодированием в H.264/AAC (чтобы потом без проблем читать).
    """
    duration = max(0.1, end - start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",   # быстрый seek по ключевым кадрам
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path)
    ]
    run_ffmpeg(cmd)


def get_video_duration(wav_path: Path) -> float:
    """Длительность видео в секундах (по длине извлечённого WAV)."""
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()