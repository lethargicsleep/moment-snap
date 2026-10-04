"""
Классификатор клипов через локальный Ollama.

Принимает транскрипт, возвращает вердикт:
    {"keep": bool, "reason": str, "tags": list[str], "summary": str}

Gemini-бэкенд добавим позже, когда понадобится.
"""
import json
from typing import Any, Dict, Optional

import requests

import config
from prompts import build_messages


# ============ ПАРСИНГ ОТВЕТА LLM ============

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Достаёт JSON-объект из ответа LLM.
    Модель иногда оборачивает JSON в markdown (```json ... ```)
    или добавляет пояснения до/после. Находим первую { и последнюю }
    и пытаемся распарсить.
    Возвращает dict или None, если не получилось.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_verdict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Приводит ответ LLM к строгой схеме.
    Модели (особенно 3B) любят:
    - возвращать keep как строку "true" вместо boolean true;
    - забывать tags и summary;
    - класть в tags строку вместо массива.

    Эта функция лечит все эти случаи и гарантирует,
    что downstream-код получит предсказуемый dict.
    """
    # keep — строго boolean
    keep = data.get("keep")
    if isinstance(keep, str):
        keep = keep.strip().lower() == "true"
    elif not isinstance(keep, bool):
        keep = bool(keep)
    else:
        keep = bool(keep)

    # reason — строка, обрезаем
    reason = str(data.get("reason", "")).strip()
    if not reason:
        reason = "без пояснения"

    # tags — всегда список строк
    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip().lower() for t in tags if str(t).strip()]
    tags = tags[:3]  # максимум 3 тега, как в промпте

    # summary — строка
    summary = str(data.get("summary", "")).strip()

    return {
        "keep": keep,
        "reason": reason,
        "tags": tags,
        "summary": summary,
    }


def _fallback(raw: str) -> Dict[str, Any]:
    """
    Что вернуть, если LLM не дала валидный JSON.
    По умолчанию — keep=False (не тащим мусор в архив).
    Сырой ответ сохраняем в поле raw для отладки.
    """
    return {
        "keep": False,
        "reason": "LLM не вернула валидный JSON",
        "tags": [],
        "summary": "",
        "raw": raw[:500],  # обрезаем, чтобы не раздувать JSON-отчёт
    }


# ============ OLLAMA ============

def classify(transcript: str) -> Dict[str, Any]:
    """
    Классификация через локальный Ollama.
    Использует chat-API (messages с ролями) вместо старого prompt.
    format="json" заставляет модель вернуть строго JSON-объект.

    Если транскрипт пустой — сразу keep=False, без обращения к LLM.
    """
    if not transcript or not transcript.strip():
        return {
            "keep": False,
            "reason": "пустой транскрипт",
            "tags": [],
            "summary": "",
        }

    messages = build_messages(transcript)
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": config.TEMPERATURE},
    }

    response = requests.post(
        config.OLLAMA_URL,
        json=payload,
        timeout=config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    print("=== OLLAMA RAW ===")
    print(response.text[:500])
    print("==================")

    # В chat-API ответ лежит в data["message"]["content"]
    raw = response.json().get("message", {}).get("content", "")

    parsed = _extract_json(raw)
    if parsed is None:
        return _fallback(raw)
    return _normalize_verdict(parsed)