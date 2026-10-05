import os
import sys
from pathlib import Path

# ВАЖНО: добавляем PATH ДО импорта faster_whisper
venv_site = Path(sys.prefix) / "Lib" / "site-packages"
for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
    p = venv_site / sub
    if p.is_dir():
        os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
        print(f"[CUDA] PATH += {p}")

from faster_whisper import WhisperModel
import config


class Transcriber:
    def __init__(self):
        print(f"Загружаю Whisper '{config.WHISPER_MODEL}' "
              f"на {config.WHISPER_DEVICE} ({config.WHISPER_COMPUTE_TYPE})...")
        self.model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )

    def transcribe(self, wav_path: Path) -> str:
        segments, _info = self.model.transcribe(
            str(wav_path),
            language=config.WHISPER_LANGUAGE,
            vad_filter=True,
            beam_size=5,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()