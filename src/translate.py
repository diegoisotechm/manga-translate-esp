"""
Traducción japonés → español.
Motor principal: OpenRouter (modelos abiertos, sin filtros de contenido adulto).
Respaldo automático: Gemini (con reintentos).
"""

import time
import random
from typing import List

import requests
from google import genai
from google.genai import types


def is_cjk(character: str) -> bool:
    """Verifica si un carácter es CJK (chino/japonés/coreano)."""
    return any(start <= ord(character) <= end for start, end in [
        (4352, 4607), (11904, 42191), (43072, 43135), (44032, 55215),
        (63744, 64255), (65072, 65103), (65381, 65500), (131072, 196607)
    ])


def _contains_japanese(text: str) -> bool:
    """Verifica si el texto contiene al menos un carácter CJK."""
    return bool(text and text.strip() and any(is_cjk(c) for c in text))


# ============================================================
# OPENROUTER (motor principal, modelos abiertos sin filtros)
# ============================================================
def _translate_with_openrouter(texts: List[str], api_key: str, target: str = "es",
                               model: str = "meta-llama/llama-4-scout") -> List[str]:
    """
    Traduce con OpenRouter (compatible con la API de OpenAI).
    Usa meta-llama/llama-4-scout por defecto.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://manga-translator.local",
        "X-Title": "Manga Translator CLI",
    }

    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    prompt = f"""TRADUCE ESTAS {len(texts)} LÍNEAS DE MANGA AL ESPAÑOL.

REGLAS CRÍTICAS:
1. Debes devolver EXACTAMENTE {len(texts)} líneas.
2. Cada línea debe empezar con su número seguido de punto y espacio (ej: "1. Texto").
3. NO uses saltos de línea dentro de una misma traducción. Si una burbuja tiene varias frases, únelas en una sola línea separadas por espacios.
4. No añadas notas ni explicaciones.

Entrada:
{numbered}

Salida:"""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if "choices" not in data or not data["choices"]:
            raise ValueError(f"OpenRouter no devolvió choices. Respuesta: {data}")

        result_text = data["choices"][0]["message"]["content"].strip()

        lines = result_text.splitlines()
        translations = []

        current_translation_parts = []
        expected_count = len(texts)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            is_new_line = False
            if ". " in line:
                parts = line.split(". ", 1)
                try:
                    num = int(parts[0])
                    if 1 <= num <= expected_count:
                        is_new_line = True
                except ValueError:
                    pass

            if is_new_line:
                if current_translation_parts:
                    translations.append(" ".join(current_translation_parts))
                current_translation_parts = [line.split(". ", 1)[1]]
            else:
                current_translation_parts.append(line)

        if current_translation_parts:
            translations.append(" ".join(current_translation_parts))

        if len(translations) > expected_count:
            translations = translations[:expected_count]
        elif len(translations) < expected_count:
            translations.extend([""] * (expected_count - len(translations)))

        return translations

    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            raise ValueError(
                f"Error 404: El modelo '{model}' no existe o no está disponible en OpenRouter. "
                f"Verifica que el nombre del modelo sea correcto. "
                f"Respuesta del servidor: {resp.text[:200]}"
            )
        elif resp.status_code == 401:
            raise ValueError(
                "Error 401: API key de OpenRouter inválida o no proporcionada."
            )
        elif resp.status_code == 429:
            raise ValueError(
                "Error 429: Límite de rate excedido en OpenRouter. Espera un momento."
            )
        else:
            raise ValueError(f"Error HTTP {resp.status_code} de OpenRouter: {resp.text[:200]}")
    except Exception as e:
        raise ValueError(f"Error en OpenRouter: {e}")


# ============================================================
# GEMINI (respaldo automático, con reintentos)
# ============================================================
def _translate_with_gemini(texts: List[str], api_key: str, target: str = "es",
                           max_retries: int = 5) -> List[str]:
    """
    Traduce con Gemini como respaldo. Reintenta con backoff exponencial ante 503/429.
    """
    client = genai.Client(api_key=api_key)

    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = f"""Eres un traductor profesional de manga japonés. Traduce el siguiente texto japonés al español.

Requisitos:
- El texto proviene de diálogos de personajes de manga; el lenguaje puede ser coloquial, tener onomatopeyas o expresiones emocionales.
- Traduce el contexto completo como un todo, manteniendo la coherencia y naturalidad del diálogo.
- Devuelve estrictamente el mismo número de líneas que la entrada, cada línea en el formato "número. traducción".
- Solo devuelve el texto traducido, sin explicaciones adicionales.

Texto japonés:
{numbered}

IMPORTANTE: Devuelve exactamente {len(texts)} líneas, cada una comenzando con el número correspondiente. No fusiones líneas, no omitas líneas, no añadas explicaciones."""

    safety_settings = [
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE),
    ]

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(safety_settings=safety_settings),
            )

            if not response.text or not response.text.strip():
                raise ValueError("Gemini rechazó la traducción (posible filtro de seguridad)")

            result_text = response.text.strip()
            lines = result_text.splitlines()
            translations = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if ". " in line:
                    line = line.split(". ", 1)[1]
                elif "．" in line:
                    line = line.split("．", 1)[1]
                translations.append(line)

            if len(translations) != len(texts):
                raise ValueError(
                    f"Gemini devolvió {len(translations)} líneas, se esperaban {len(texts)}. "
                    f"Respuesta: {result_text[:300]}"
                )

            return translations

        except Exception as e:
            last_error = e
            error_str = str(e)
            if "503" in error_str or "UNAVAILABLE" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    delay = (2 ** (attempt + 1)) + random.uniform(0, 1)
                    print(f"  Gemini sobrecargado (intento {attempt + 1}/{max_retries}), "
                          f"reintentando en {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise ValueError(
                        f"Gemini sigue sobrecargado tras {max_retries} intentos. "
                        f"Último error: {e}"
                    )
            else:
                raise

    raise ValueError(f"Error inesperado en traducción: {last_error}")


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================
def translate_batch(texts: List[str], gemini_api_key: str,
                    openrouter_api_key: str = None,
                    openrouter_model: str = "meta-llama/llama-4-scout",
                    target: str = "es", silent: bool = False) -> List[str]:
    """
    Traduce una lista de textos japoneses al español.
    Intenta primero con OpenRouter; si falla o no hay openrouter_api_key,
    usa Gemini como respaldo automático.
    """
    if not texts:
        return []

    texts_to_translate = []
    text_indices = []
    for idx, text in enumerate(texts):
        if _contains_japanese(text):
            texts_to_translate.append(text)
            text_indices.append(idx)

    if not texts_to_translate:
        return [""] * len(texts)

    if not silent:
        print(f"Traduciendo {len(texts_to_translate)} textos...")

    if openrouter_api_key:
        try:
            translated = _translate_with_openrouter(
                texts_to_translate, openrouter_api_key, target, openrouter_model
            )
        except Exception as e:
            if gemini_api_key:
                if not silent:
                    print(f"  OpenRouter falló ({e}). Usando Gemini como respaldo...")
                translated = _translate_with_gemini(texts_to_translate, gemini_api_key, target)
            else:
                raise
    else:
        translated = _translate_with_gemini(texts_to_translate, gemini_api_key, target)

    result = [""] * len(texts)
    for idx, trans in zip(text_indices, translated):
        result[idx] = trans

    return result


def translate_phrase(text: str, gemini_api_key: str,
                     openrouter_api_key: str = None,
                     openrouter_model: str = "meta-llama/llama-4-scout",
                     target: str = "es", silent: bool = False):
    """Traduce un solo texto. Devuelve (traducción, éxito)."""
    if not _contains_japanese(text):
        return "", False
    try:
        translated = translate_batch([text], gemini_api_key, openrouter_api_key,
                                     openrouter_model, target, silent)
        return (translated[0] if translated else ""), True
    except Exception:
        return "", False


def translate_individual(texts: List[str], gemini_api_key: str,
                         openrouter_api_key: str = None,
                         openrouter_model: str = "meta-llama/llama-4-scout",
                         target: str = "es", silent: bool = False) -> List[str]:
    return translate_batch(texts, gemini_api_key, openrouter_api_key,
                           openrouter_model, target, silent)