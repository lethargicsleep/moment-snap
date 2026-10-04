"""
Центральный конфиг проекта MomentSnap PoC.
Все "магические числа" и пути живут только здесь.
Меняешь поведение пайплайна — правишь только этот файл.
"""
from pathlib import Path

# ---------- Пути ----------
VIDEO_PATH = Path(r"C:\Users\USER\Desktop\MomentSnap\20260930_191736.mp4")   # исходное 20-мин видео
OUTPUT_DIR = Path(r"C:\Users\USER\Desktop\MomentSnap") # куда складывать всё
CLIPS_DIR = OUTPUT_DIR / "clips"                        # нарезанные 30-сек клипы
AUDIO_DIR = OUTPUT_DIR / "audio"                        # wav-дорожки клипов
FULL_WAV = OUTPUT_DIR / "full_audio.wav"                # wav всего видео
RESULTS_JSON = OUTPUT_DIR / "results.json"              # финальный отчёт

# ---------- Параметры нарезки ----------
CLIP_DURATION = 30          # длина клипа, сек
PRE_PEAK = 15               # сколько секунд взять ДО пика
POST_PEAK = 15              # сколько секунд взять ПОСЛЕ пика
MIN_GAP_BETWEEN_PEAKS = 20.0  # если два пика ближе — оставляем самый громкий
VOLUME_WINDOW = 0.5         # окно анализа RMS, сек
PEAK_PERCENTILE = 95        # верхние 5% громкости считаем пиками
MIN_RMS_THRESHOLD = 0.01    # абсолютный минимум RMS, ниже — игнор

# ---------- Whisper ----------
WHISPER_MODEL = "large-v3-turbo"      # "tiny"|"base"|"small"|"medium"|"large-v3"
WHISPER_DEVICE = "cuda"         # "cuda" для RTX 4060, "cpu" как fallback
WHISPER_COMPUTE_TYPE = "float16"  # "float16" на GPU, "int8" на CPU
WHISPER_LANGUAGE = "ru"         # язык распознавания

# ---------- Выбор LLM-бэкенда ----------
AI_BACKEND = "ollama"           # "ollama" или "gemini"

# Ollama (локально на RTX 4060)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"     # или "llama3:8b"

# Google Gemini (бесплатный API)
GEMINI_MODEL = "gemini-1.5-flash"   # при ошибке — "gemini-2.0-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_TIMEOUT = 120

# ---------- Прочее ----------
REQUEST_TIMEOUT = 120           # таймаут HTTP-запросов к LLM
TEMPERATURE = 0.1               # почти детерминированный вывод