"""Configuración del pipeline de traducción de manga."""

from dataclasses import dataclass, field
from typing import Optional
import os
import torch # Importamos torch para la detección

def get_default_device():
    """Detecta automáticamente CUDA, MPS (Mac) o CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

@dataclass
class Config:
    """Configuración del pipeline."""

    # Detección
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    parent_box_threshold: int = 20
    bbox_processing: str = 'remove-parent'

    # Burbujas
    threshold_value: int = 200

    # Renderizado
    font_path: str = 'fonts/CC Astro City Int Regular.ttf'

    # Traducción
    gemini_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "meta-llama/llama-4-scout"
    
    # Dispositivo: Si es None, se detecta automáticamente
    translation_device: str = field(default_factory=get_default_device)

    # OCR
    ocr_model_id: str = 'jzhang533/PaddleOCR-VL-For-Manga'
    ocr_max_new_tokens: int = 2048

    # Salidas
    save_speech_bubbles: bool = False
    save_bubble_interiors: bool = False
    save_cleaned: bool = False
    save_translated: bool = True

    # Opciones
    silent: bool = False
    stop_on_error: bool = False
    batch: bool = False
    batch_amount: Optional[int] = None

    @classmethod
    def from_env(cls) -> 'Config':
        """Crea una configuración desde variables de entorno (.env)."""
        return cls(
            gemini_api_key=os.getenv('GEMINI_API_KEY'),
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
            openrouter_model=os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-4-scout'),
        )