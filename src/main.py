#!/usr/bin/env python3
"""
CLI para traducir páginas de manga de japonés a español (OpenRouter principal, respaldo Gemini).
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Cargar .env antes de leer variables de entorno
load_dotenv(Path(__file__).parent.parent / ".env")

from .manga_translate import (
    translate_manga_page,
    translate_manga_folder,
    translate_manga_page_batch,
    translate_manga_folder_batch,
)
from .config import Config
from .console import Console

DEFAULT_FONT_PATH = str(Path(__file__).parent / "fonts" / "CC Astro City Int Regular.ttf")


def get_default_device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


DEFAULT_DEVICE = get_default_device()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traduce páginas de manga de japonés a español usando OpenRouter (respaldo: Gemini)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Traducir una sola página
  python -m src.main input/page1.png --output output

  # Traducir una carpeta en modo batch
  python -m src.main input/ --batch --output output

  # Sobrescribir la API key de OpenRouter del .env
  python -m src.main input/page1.png --openrouter-key "OTRA_CLAVE"
        """
    )

    parser.add_argument('input', type=str, nargs='+',
                        help='Archivo(s) o carpeta(s) de entrada')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Carpeta de salida (default: output)')

    # Detección
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Umbral de confianza para detección (0-1, default: 0.25)')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='Umbral IoU para NMS (0-1, default: 0.45)')
    parser.add_argument('--parent-box-threshold', type=int, default=20,
                        help='Umbral para burbujas compuestas (default: 20)')
    parser.add_argument('--bbox-processing', type=str, default='remove-parent',
                        choices=['remove-parent', 'combine-children', 'none'],
                        help='Modo de procesamiento de burbujas compuestas')

    # Burbujas
    parser.add_argument('--threshold-value', type=int, default=200,
                        help='Umbral para detección de máscara de burbuja')

    # Fuente
    parser.add_argument('--font', type=str, default=DEFAULT_FONT_PATH,
                        help='Ruta de la fuente para el texto traducido')

    # Dispositivo (OCR)
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                        choices=['cpu', 'cuda'],
                        help=f'Dispositivo para OCR (default: {DEFAULT_DEVICE})')

    # OCR
    parser.add_argument('--ocr-model', type=str,
                        default='jzhang533/PaddleOCR-VL-For-Manga',
                        help='ID del modelo OCR')
    parser.add_argument('--ocr-max-tokens', type=int, default=2048,
                        help='Máximo de tokens para OCR')

    # Traducción (OpenRouter principal, Gemini respaldo)
    parser.add_argument('--openrouter-key', type=str,
                        default=os.environ.get("OPENROUTER_API_KEY"),
                        help='API Key de OpenRouter (motor principal, o variable OPENROUTER_API_KEY)')
    parser.add_argument('--openrouter-model', type=str,
                        default=os.environ.get("OPENROUTER_MODEL",
                                               "meta-llama/llama-4-scout"),
                        help='Modelo de OpenRouter para traducción principal')
    parser.add_argument('--api-key', type=str,
                        default=os.environ.get("GEMINI_API_KEY"),
                        help='API Key de Google Gemini (respaldo si OpenRouter falla, o GEMINI_API_KEY)')

    # Salidas
    parser.add_argument('--save-all', action='store_true',
                        help='Guardar todas las salidas intermedias')
    parser.add_argument('--save-speech-bubbles', action='store_true',
                        help='Guardar imágenes de detección anotadas')
    parser.add_argument('--save-bubble-interiors', action='store_true',
                        help='Guardar visualizaciones de interiores de burbujas')
    parser.add_argument('--save-cleaned', action='store_true',
                        help='Guardar imágenes limpias antes de dibujar texto')
    parser.add_argument('--no-translated', action='store_true',
                        help='No guardar imágenes traducidas')

    # Opciones
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suprimir mensajes de progreso')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='Detener el procesamiento al primer error')

    # Batch
    parser.add_argument('--batch', action='store_true',
                        help='Modo batch (procesar todas las páginas juntas por paso)')
    parser.add_argument('--batch-amount', type=int, default=None,
                        help='Máximo de páginas por lote (default: todas)')

    parser.add_argument('--benchmark', action='store_true',
                        help='Medir y mostrar tiempo de procesamiento')

    return parser.parse_args()


def validate_args(args):
    supported_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.webp', '.WEBP'}
    all_valid = True

    for input_str in args.input:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"Error: la ruta no existe: {input_str}", file=sys.stderr)
            all_valid = False
            continue
        if input_path.is_dir():
            continue
        elif input_path.is_file():
            if input_path.suffix not in supported_extensions:
                print(f"Error: formato no soportado: {input_path.suffix} para {input_str}", file=sys.stderr)
                all_valid = False
        else:
            print(f"Error: la ruta no es archivo ni carpeta: {input_str}", file=sys.stderr)
            all_valid = False

    if args.font:
        font_path = Path(args.font)
        if not font_path.exists() or not font_path.is_file():
            print(f"Error: fuente no encontrada: {args.font}", file=sys.stderr)
            all_valid = False

    if not 0 <= args.conf_threshold <= 1:
        print("Error: conf-threshold debe estar entre 0 y 1", file=sys.stderr)
        all_valid = False
    if not 0 <= args.iou_threshold <= 1:
        print("Error: iou-threshold debe estar entre 0 y 1", file=sys.stderr)
        all_valid = False

    return all_valid


def main():
    args = parse_args()

    if not validate_args(args):
        sys.exit(1)

    if not args.openrouter_key and not args.api_key:
        print("Error: se requiere OPENROUTER_API_KEY (o --openrouter-key) en .env", file=sys.stderr)
        sys.exit(1)

    if not args.openrouter_key:
        print("Advertencia: no se encontró OPENROUTER_API_KEY. "
              "Se usará Gemini directamente como motor.", file=sys.stderr)
    elif not args.api_key:
        print("Advertencia: no se encontró GEMINI_API_KEY. "
              "Si OpenRouter falla, no habrá respaldo.", file=sys.stderr)

    if args.save_all:
        save_speech_bubbles = True
        save_bubble_interiors = True
        save_cleaned = True
        save_translated = True
    else:
        save_speech_bubbles = args.save_speech_bubbles
        save_bubble_interiors = args.save_bubble_interiors
        save_cleaned = args.save_cleaned
        save_translated = not args.no_translated

    config = Config(
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        parent_box_threshold=args.parent_box_threshold,
        bbox_processing=args.bbox_processing,
        threshold_value=args.threshold_value,
        font_path=args.font,
        gemini_api_key=args.api_key,
        openrouter_api_key=args.openrouter_key,
        openrouter_model=args.openrouter_model,
        translation_device=args.device,
        ocr_model_id=args.ocr_model,
        ocr_max_new_tokens=args.ocr_max_tokens,
        save_speech_bubbles=save_speech_bubbles,
        save_bubble_interiors=save_bubble_interiors,
        save_cleaned=save_cleaned,
        save_translated=save_translated,
        silent=args.quiet,
        stop_on_error=args.stop_on_error,
        batch=args.batch,
        batch_amount=args.batch_amount,
    )

    console = Console(quiet=args.quiet)

    console.info("Traductor de manga (OpenRouter → español, respaldo Gemini)")
    console.info(f"Dispositivo OCR: {args.device.upper()}")
    console.info(f"Salida: {args.output}")
    if args.batch:
        console.info("Modo: batch")
    console.info("Cargando modelos (la primera vez descarga)...")

    try:
        start_time = time.time() if args.benchmark else None

        supported_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.webp', '.WEBP'}
        total_processed = 0
        total_failed = 0

        all_batch_images = []

        if args.batch:
            for input_str in args.input:
                input_path = Path(input_str)
                if input_path.is_dir():
                    console.section(f"Procesando carpeta en batch: {input_str}")
                    results = translate_manga_folder_batch(
                        input_folder=input_str,
                        output_folder=args.output,
                        config=config,
                        console=console,
                        batch_amount=args.batch_amount,
                    )
                    total_processed += results['successful_count']
                    total_failed += results['failed_count']
                elif input_path.is_file() and input_path.suffix in supported_extensions:
                    all_batch_images.append(input_str)

            if all_batch_images:
                console.section(f"Procesando {len(all_batch_images)} archivo(s) en batch...")
                try:
                    results_dict = translate_manga_page_batch(
                        input_image_paths=all_batch_images,
                        output_folder=args.output,
                        config=config,
                        console=console,
                    )
                    for input_str in all_batch_images:
                        r = results_dict.get(input_str, {})
                        if r.get('translated_image'):
                            total_processed += 1
                        else:
                            total_failed += 1
                except Exception as e:
                    console.error(f"Error en batch: {e}")
                    if args.stop_on_error:
                        raise
                    total_failed = len(all_batch_images)
        else:
            for input_str in args.input:
                input_path = Path(input_str)
                if input_path.is_dir():
                    console.section(f"Procesando carpeta: {input_str}")
                    results = translate_manga_folder(
                        input_folder=input_str,
                        output_folder=args.output,
                        config=config,
                        console=console,
                    )
                    total_processed += results['successful_count']
                    total_failed += results['failed_count']
                elif input_path.is_file() and input_path.suffix in supported_extensions:
                    console.section(f"Procesando archivo: {input_str}")
                    results = translate_manga_page(
                        input_image_path=input_str,
                        output_folder=args.output,
                        config=config,
                        console=console,
                    )
                    if results.get('translated_image'):
                        total_processed += 1
                        path = results['output_paths'].get('translated')
                        if path:
                            console.success(f"Listo: {path}")
                    else:
                        total_failed += 1
                        console.error(f"Falló: {input_str}")

        if len(args.input) > 1:
            console.section("Resumen")
            console.print(f"Total entradas: {len(args.input)}")
            console.print(f"Procesadas: {total_processed}")
            console.print(f"Fallidas: {total_failed}")

        if args.benchmark and start_time is not None:
            from rich.table import Table
            elapsed = time.time() - start_time
            table = Table(title="Resultados")
            table.add_column("Métrica", style="cyan")
            table.add_column("Valor", style="green")
            table.add_row("Tiempo total", f"{elapsed:.2f} s")
            if total_processed > 0:
                table.add_row("Tiempo por página", f"{elapsed / total_processed:.2f} s")
            table.add_row("Páginas procesadas", str(total_processed))
            console.console.print(table)

    except KeyboardInterrupt:
        console.error("\nInterrumpido por el usuario")
        sys.exit(130)
    except Exception as e:
        console.error(f"Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()