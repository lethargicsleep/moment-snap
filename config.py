"""
Центральный конфиг проекта MomentSnap.
Только настройки нарезки клипов. Без Whisper и LLM.
"""
from pathlib import Path

# ---------- Пути ----------
VIDEO_PATH = Path(r"C:\Users\USER\Desktop\MomentSnap\20260930_191736.mp4")
OUTPUT_DIR = Path(r"C:\MomentSnap_PoC")
CLIPS_DIR = OUTPUT_DIR / "clips"
AUDIO_DIR = OUTPUT_DIR / "audio"
FULL_WAV = OUTPUT_DIR / "full_audio.wav"
RESULTS_JSON = OUTPUT_DIR / "results.json"

# ---------- Параметры нарезки ----------
CLIP_DURATION = 30          # длина клипа, сек
PRE_PEAK = 15               # сколько секунд взять ДО пика
POST_PEAK = 15              # сколько секунд взять ПОСЛЕ пика
MIN_GAP_BETWEEN_PEAKS = 20.0  # если два пика ближе — оставляем самый громкий
VOLUME_WINDOW = 0.5         # окно анализа RMS, сек
PEAK_PERCENTILE = 95        # верхние 5% громкости считаем пиками
MIN_RMS_THRESHOLD = 0.01    # абсолютный минимум RMS, ниже — игнор

# ---------- Очистка ----------
CLEANUP_TEMP_AFTER_RUN = True   # удалять full_audio.wav после нарезки