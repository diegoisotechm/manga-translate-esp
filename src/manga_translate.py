"""
Pipeline completo de traducción de páginas de manga.
Traducción japonés → español con Gemini.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import sys
import numpy as np
import cv2
from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.webp', '.WEBP'}

from src.bubble_detect import load_bubble_detection_model, run_detection
from src.image_utils import get_cropped_images, save_pil_image
from src.bbox import BoundingBox, remove_parent_boxes, combine_overlapping_bubbles
from src.draw_text import draw_text_on_image
from src.bubble_clean import fill_bubble_interiors, get_bubble_text_mask
from src.config import Config
from src.console import Console
from src.output_manager import OutputManager


def _get_output_dir_path(output_base: Path, folder_name: str, output_subfolder: Optional[str] = None) -> Path:
    if output_subfolder:
        return output_base / folder_name / output_subfolder
    return output_base / folder_name


def _get_output_file_path(output_base: Path, folder_name: str, filename: str,
                          output_subfolder: Optional[str] = None) -> Path:
    if output_subfolder:
        return output_base / folder_name / output_subfolder / filename
    return output_base / folder_name / filename


def _detect_speech_bubbles(input_image, detection_model, conf_threshold, iou_threshold,
                           save_speech_bubbles, output_base, output_filename,
                           output_subfolder, quiet=False):
    if not quiet:
        print("Ejecutando detección de burbujas...")

    annotated_bubble_img, boxes = run_detection(
        detection_model, input_image,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold, silent=quiet
    )

    output_paths = {}
    if not quiet and annotated_bubble_img:
        print(f"Burbujas detectadas: {len(boxes)}")

    if save_speech_bubbles and annotated_bubble_img:
        p = _get_output_file_path(output_base, "speech_bubbles",
                                  f"{output_filename}_speech_bubbles.png", output_subfolder)
        save_pil_image(annotated_bubble_img, str(p), print_message=not quiet)
        output_paths['speech_bubbles'] = str(p)

    return annotated_bubble_img, boxes, output_paths


def _process_bounding_boxes(boxes, bbox_processing, parent_box_threshold, quiet=False):
    if not boxes:
        if not quiet:
            print("No se detectaron burbujas.")
        return []

    if bbox_processing == 'remove-parent':
        filtered = remove_parent_boxes(boxes, threshold=parent_box_threshold)
    elif bbox_processing == 'combine-children':
        filtered = combine_overlapping_bubbles(boxes, touch_threshold=parent_box_threshold)
    else:
        filtered = boxes

    if not quiet:
        print(f"Burbujas tras procesar: {len(filtered)}")
    return filtered


def _extract_and_translate_text(cropped_images, ocr_model_id, ocr_max_new_tokens,
                                gemini_api_key, translation_device, quiet=False,
                                openrouter_api_key=None, openrouter_model=None):
    """Extrae texto con OCR y traduce con Gemini."""
    from src.ocr import extract_text_batch
    from src.translate import translate_batch

    if not quiet:
        print("\n" + "=" * 50)
        print("Ejecutando OCR en las burbujas...")
        print("=" * 50)

    images_to_process = []
    bbox_indices = []
    for i, (cropped_img, bbox) in enumerate(cropped_images, 1):
        images_to_process.append(cropped_img)
        bbox_indices.append((bbox, i))

    extracted_texts = []
    try:
        extracted_text_list = extract_text_batch(
            images_to_process,
            model_id=ocr_model_id,
            max_new_tokens=ocr_max_new_tokens,
            device=translation_device,
            silent=quiet,
        )
        for (bbox, index), text in zip(bbox_indices, extracted_text_list):
            if not quiet:
                try:
                    sys.stdout.reconfigure(encoding='utf-8')
                except (AttributeError, ValueError):
                    pass
                print(f"\nBurbuja {index} (bbox: {bbox}):")
                print(f"  Texto extraído: {text}")
            extracted_texts.append((bbox, text, index))
    except Exception as e:
        if not quiet:
            print(f"Error en OCR: {e}")
        for bbox, index in bbox_indices:
            extracted_texts.append((bbox, "", index))

    texts_list = [t for _, t, _ in extracted_texts]

    if not quiet:
        print("\n" + "=" * 50)
        print("Traduciendo textos con Gemini (ja → es)...")
        print("=" * 50)

    translated_texts_list = translate_batch(
        texts_list,
        gemini_api_key=gemini_api_key,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        target="es",
        silent=quiet,
    )

    bubble_texts = []
    japanese_bboxes = []
    for (bbox, original, index), translated in zip(extracted_texts, translated_texts_list):
        if translated.strip():
            bubble_texts.append((bbox, translated))
            japanese_bboxes.append(bbox)
            if not quiet:
                print(f"\nBurbuja {index}:")
                print(f"  Original: {original}")
                print(f"  Traducido: {translated}")

    return bubble_texts, japanese_bboxes


def _generate_bubble_masks(input_image, bubble_texts, threshold_value, quiet=False):
    img_array = np.array(input_image.convert("RGB"))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    bubble_masks = {}
    for bbox, _ in bubble_texts:
        clipped = bbox.clip(img_array.shape[1], img_array.shape[0])
        if not clipped.is_valid():
            continue
        x1, y1, x2, y2 = clipped
        crop = img_array[y1:y2, x1:x2].copy()
        mask, _ = get_bubble_text_mask(crop, threshold_value=threshold_value)
        bubble_masks[bbox] = mask
    return bubble_masks


def _clean_bubble_interiors(input_image, japanese_bboxes, threshold_value,
                            save_cleaned, output_base, output_filename, output_subfolder,
                            quiet=False):
    cleaned = fill_bubble_interiors(input_image, japanese_bboxes, threshold_value=threshold_value)
    cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    cleaned_pil = Image.fromarray(cleaned_rgb)

    output_paths = {}
    if save_cleaned:
        p = _get_output_file_path(output_base, "cleaned",
                                  f"{output_filename}_cleaned.png", output_subfolder)
        save_pil_image(cleaned_pil, str(p), print_message=not quiet)
        output_paths['cleaned'] = str(p)
    return cleaned_pil, output_paths


def _draw_translated_text(cleaned_image, bubble_texts, bubble_masks, font_path,
                          save_translated, output_base, output_filename, output_subfolder,
                          quiet=False):
    translated_image = draw_text_on_image(cleaned_image, bubble_texts, bubble_masks,
                                          font_path=font_path)
    output_paths = {}
    if save_translated:
        p = _get_output_file_path(output_base, "translated",
                                  f"{output_filename}_translated.png", output_subfolder)
        save_pil_image(translated_image, str(p), print_message=not quiet)
        output_paths['translated'] = str(p)
    return translated_image, output_paths


def translate_manga_page(input_image_path, output_folder, config, console,
                         detection_model=None, output_filename=None, output_subfolder=None):
    results = {
        'annotated_bubble_image': None,
        'boxes': [],
        'filtered_boxes': [],
        'cleaned_image': None,
        'translated_image': None,
        'bubble_texts': [],
        'bubble_masks': {},
        'output_paths': {},
    }

    if output_filename is None:
        output_filename = Path(input_image_path).stem

    output_manager = OutputManager(output_folder, output_subfolder)
    output_manager.setup(
        save_speech_bubbles=config.save_speech_bubbles,
        save_bubble_interiors=config.save_bubble_interiors,
        save_cleaned=config.save_cleaned,
        save_translated=config.save_translated,
    )
    output_base = output_manager.base

    try:
        input_image = Image.open(input_image_path).convert("RGB")

        if detection_model is None:
            console.info("Cargando modelo de detección...")
            detection_model = load_bubble_detection_model(silent=console.quiet)

        annotated, boxes, out_paths = _detect_speech_bubbles(
            input_image, detection_model,
            config.conf_threshold, config.iou_threshold,
            config.save_speech_bubbles, output_base, output_filename,
            output_subfolder, console.quiet,
        )
        results['annotated_bubble_image'] = annotated
        results['boxes'] = boxes
        results['output_paths'].update(out_paths)

        filtered = _process_bounding_boxes(boxes, config.bbox_processing,
                                           config.parent_box_threshold, console.quiet)
        results['filtered_boxes'] = filtered

        if not filtered:
            if config.save_translated:
                p = _get_output_file_path(output_base, "translated",
                                          f"{output_filename}_translated.png", output_subfolder)
                save_pil_image(input_image, str(p), print_message=not console.quiet)
                results['output_paths']['translated'] = str(p)
                results['translated_image'] = input_image
            return results

        console.info("Recortando burbujas...")
        cropped = get_cropped_images(input_image, filtered)
        console.info(f"Recortadas {len(cropped)} burbujas")

        bubble_texts, japanese_bboxes = _extract_and_translate_text(
            cropped, config.ocr_model_id, config.ocr_max_new_tokens,
            config.gemini_api_key, config.translation_device, console.quiet,
            openrouter_api_key=config.openrouter_api_key,
            openrouter_model=config.openrouter_model,
        )
        results['bubble_texts'] = bubble_texts

        if not bubble_texts:
            console.info("No se encontró texto japonés en las burbujas.")
            return results

        bubble_masks = _generate_bubble_masks(input_image, bubble_texts,
                                              config.threshold_value, console.quiet)
        results['bubble_masks'] = bubble_masks

        cleaned_pil, out_paths = _clean_bubble_interiors(
            input_image, japanese_bboxes, config.threshold_value,
            config.save_cleaned, output_base, output_filename, output_subfolder,
            console.quiet,
        )
        results['cleaned_image'] = cleaned_pil
        results['output_paths'].update(out_paths)

        translated_image, out_paths = _draw_translated_text(
            cleaned_pil, bubble_texts, bubble_masks, config.font_path,
            config.save_translated, output_base, output_filename, output_subfolder,
            console.quiet,
        )
        results['translated_image'] = translated_image
        results['output_paths'].update(out_paths)

        console.section("¡Procesamiento completo!")

    except Exception as e:
        console.error(f"Error en traducción: {e}")
        raise

    return results


def translate_manga_page_batch(input_image_paths, output_folder, config, console,
                               detection_model=None, output_subfolder=None,
                               batch_amount=None):
    if not input_image_paths:
        return {}

    if batch_amount is not None and batch_amount > 0:
        all_results = {}
        for i in range(0, len(input_image_paths), batch_amount):
            chunk = input_image_paths[i:i + batch_amount]
            console.section(f"Procesando lote {i//batch_amount + 1} ({len(chunk)} páginas)...")
            chunk_results = translate_manga_page_batch(
                chunk, output_folder, config, console, detection_model,
                output_subfolder, batch_amount=None,
            )
            all_results.update(chunk_results)
        return all_results

    results_by_path = {}

    output_manager = OutputManager(output_folder, output_subfolder)
    output_manager.setup(
        save_speech_bubbles=config.save_speech_bubbles,
        save_bubble_interiors=config.save_bubble_interiors,
        save_cleaned=config.save_cleaned,
        save_translated=config.save_translated,
    )
    output_base = output_manager.base

    try:
        console.section(f"Cargando {len(input_image_paths)} imagen(es)...")

        input_images = []
        output_filenames = []
        for p_str in input_image_paths:
            p = Path(p_str)
            input_images.append(Image.open(p_str).convert("RGB"))
            output_filenames.append(p.stem)
            results_by_path[p_str] = {
                'annotated_bubble_image': None,
                'boxes': [],
                'filtered_boxes': [],
                'cleaned_image': None,
                'translated_image': None,
                'bubble_texts': [],
                'bubble_masks': {},
                'output_paths': {},
            }

        if detection_model is None:
            console.info("Cargando modelo de detección...")
            detection_model = load_bubble_detection_model(silent=console.quiet)

        console.section(f"Detectando burbujas en {len(input_images)} imagen(es)...")

        all_boxes = []
        for input_image, p_str, fname in zip(input_images, input_image_paths, output_filenames):
            annotated, boxes, out_paths = _detect_speech_bubbles(
                input_image, detection_model,
                config.conf_threshold, config.iou_threshold,
                config.save_speech_bubbles, output_base, fname,
                output_subfolder, True,
            )
            all_boxes.append(boxes)
            results_by_path[p_str]['annotated_bubble_image'] = annotated
            results_by_path[p_str]['boxes'] = boxes
            results_by_path[p_str]['output_paths'].update(out_paths)

        console.section("Procesando bounding boxes...")

        all_filtered = []
        for boxes, p_str in zip(all_boxes, input_image_paths):
            filtered = _process_bounding_boxes(boxes, config.bbox_processing,
                                               config.parent_box_threshold, True)
            all_filtered.append(filtered)
            results_by_path[p_str]['filtered_boxes'] = filtered

        console.section("Recortando burbujas...")

        all_cropped = []
        page_bubble_indices = []
        for page_idx, (input_image, filtered) in enumerate(zip(input_images, all_filtered)):
            if not filtered:
                page_bubble_indices.append((len(all_cropped), len(all_cropped)))
                continue
            start = len(all_cropped)
            cropped = get_cropped_images(input_image, filtered)
            for c_img, bbox in cropped:
                all_cropped.append((page_idx, c_img, bbox))
            page_bubble_indices.append((start, len(all_cropped)))
            console.info(f"Página {page_idx + 1}: {len(cropped)} burbujas")

        if not all_cropped:
            console.info("No se encontraron burbujas en ninguna imagen.")
            if config.save_translated:
                for input_image, p_str, fname in zip(input_images, input_image_paths, output_filenames):
                    p = _get_output_file_path(output_base, "translated",
                                              f"{fname}_translated.png", output_subfolder)
                    save_pil_image(input_image, str(p), print_message=not console.quiet)
                    results_by_path[p_str]['output_paths']['translated'] = str(p)
                    results_by_path[p_str]['translated_image'] = input_image
            return results_by_path

        console.section(f"Ejecutando OCR en {len(all_cropped)} burbujas...")
        from src.ocr import extract_text_batch
        ocr_images = [c for _, c, _ in all_cropped]
        ocr_results = extract_text_batch(
            ocr_images, model_id=config.ocr_model_id,
            max_new_tokens=config.ocr_max_new_tokens,
            device=config.translation_device, silent=console.quiet,
        )

        all_extracted = []
        bubble_counter = {}
        for (page_idx, _, bbox), text in zip(all_cropped, ocr_results):
            bubble_counter[page_idx] = bubble_counter.get(page_idx, 0) + 1
            idx = bubble_counter[page_idx]
            all_extracted.append((page_idx, bbox, text, idx))
            if not console.quiet:
                try:
                    sys.stdout.reconfigure(encoding='utf-8')
                except (AttributeError, ValueError):
                    pass
                console.print(f"\nPágina {page_idx + 1}, Burbuja {idx} (bbox: {bbox}):")
                console.print(f"  Texto extraído: {text}")

        texts_list = [t for _, _, t, _ in all_extracted]

        from src.translate import translate_batch
        console.section("Traduciendo textos con Gemini...")
        translated_list = translate_batch(
            texts_list,
            gemini_api_key=config.gemini_api_key,
            openrouter_api_key=config.openrouter_api_key,
            openrouter_model=config.openrouter_model,
            target="es",
            silent=console.quiet,
        )

        all_bubble_texts = [[] for _ in input_images]
        all_japanese_bboxes = [[] for _ in input_images]
        for (page_idx, bbox, original, idx), translated in zip(all_extracted, translated_list):
            if translated.strip():
                all_bubble_texts[page_idx].append((bbox, translated))
                all_japanese_bboxes[page_idx].append(bbox)
                if not console.quiet:
                    console.print(f"\nPágina {page_idx + 1}, Burbuja {idx}:")
                    console.print(f"  Original: {original}")
                    console.print(f"  Traducido: {translated}")

        console.section("Generando máscaras de burbujas...")
        all_masks = []
        for page_idx, (input_image, bubble_texts) in enumerate(zip(input_images, all_bubble_texts)):
            if bubble_texts:
                masks = _generate_bubble_masks(input_image, bubble_texts,
                                               config.threshold_value, console.quiet)
                all_masks.append(masks)
                results_by_path[input_image_paths[page_idx]]['bubble_masks'] = masks
            else:
                all_masks.append({})

        console.section("Limpiando burbujas...")
        all_cleaned = []
        for page_idx, (input_image, j_bboxes, p_str, fname) in enumerate(
            zip(input_images, all_japanese_bboxes, input_image_paths, output_filenames)
        ):
            if j_bboxes:
                cleaned_pil, out_paths = _clean_bubble_interiors(
                    input_image, j_bboxes, config.threshold_value,
                    config.save_cleaned, output_base, fname, output_subfolder,
                    console.quiet,
                )
                all_cleaned.append(cleaned_pil)
                results_by_path[p_str]['cleaned_image'] = cleaned_pil
                results_by_path[p_str]['output_paths'].update(out_paths)
            else:
                all_cleaned.append(input_image)

        console.section("Dibujando texto traducido...")
        for page_idx, (cleaned, btexts, bmasks, p_str, fname) in enumerate(
            zip(all_cleaned, all_bubble_texts, all_masks, input_image_paths, output_filenames)
        ):
            if btexts and bmasks:
                translated_image, out_paths = _draw_translated_text(
                    cleaned, btexts, bmasks, config.font_path,
                    config.save_translated, output_base, fname, output_subfolder,
                    console.quiet,
                )
                results_by_path[p_str]['translated_image'] = translated_image
                results_by_path[p_str]['bubble_texts'] = btexts
                results_by_path[p_str]['output_paths'].update(out_paths)
            elif config.save_translated:
                p = _get_output_file_path(output_base, "translated",
                                          f"{fname}_translated.png", output_subfolder)
                save_pil_image(cleaned, str(p), print_message=not console.quiet)
                results_by_path[p_str]['output_paths']['translated'] = str(p)
                results_by_path[p_str]['translated_image'] = cleaned

        console.section(f"¡Batch completo! {len(input_image_paths)} página(s) procesadas")

    except Exception as e:
        console.error(f"Error en batch: {e}")
        raise

    return results_by_path


def translate_manga_folder(input_folder, output_folder, config, console):
    input_path = Path(input_folder)
    if not input_path.exists():
        raise ValueError(f"La carpeta no existe: {input_folder}")
    if not input_path.is_dir():
        raise ValueError(f"No es una carpeta: {input_folder}")

    folder_name = input_path.name
    image_files = sorted([
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix in SUPPORTED_IMAGE_EXTENSIONS
    ])
    total = len(image_files)

    if total == 0:
        console.info(f"No se encontraron imágenes en {input_folder}")
        return {'processed_files': [], 'failed_files': [], 'results_by_file': {},
                'total_files': 0, 'successful_count': 0, 'failed_count': 0}

    console.section(f"Encontradas {total} imagen(es)")

    detection_model = None
    console.info("Cargando modelo de detección...")
    try:
        detection_model = load_bubble_detection_model(silent=console.quiet)
    except Exception as e:
        if config.stop_on_error:
            raise
        console.error(f"Advertencia: no se pudo cargar el modelo: {e}")

    processed, failed, results_by_file = [], [], []

    for i, img_file in enumerate(image_files, 1):
        console.section(f"Procesando {i}/{total}: {img_file.name}")
        try:
            result = translate_manga_page(
                str(img_file), output_folder, config, console,
                detection_model=detection_model,
                output_filename=None,
                output_subfolder=folder_name,
            )
            processed.append(str(img_file))
            results_by_file.append((str(img_file), result))
        except Exception as e:
            failed.append((str(img_file), str(e)))
            console.error(f"Error en {img_file.name}: {e}")
            if config.stop_on_error:
                raise

    console.section("Carpeta procesada")
    console.info(f"Total: {total}")
    console.success(f"Procesadas: {len(processed)}")
    if failed:
        console.error(f"Fallidas: {len(failed)}")
        for f, e in failed:
            console.print(f"  - {Path(f).name}: {e}")

    return {
        'processed_files': processed,
        'failed_files': failed,
        'results_by_file': dict(results_by_file),
        'total_files': total,
        'successful_count': len(processed),
        'failed_count': len(failed),
    }


def translate_manga_folder_batch(input_folder, output_folder, config, console,
                                 batch_amount=None):
    input_path = Path(input_folder)
    if not input_path.exists():
        raise ValueError(f"La carpeta no existe: {input_folder}")
    if not input_path.is_dir():
        raise ValueError(f"No es una carpeta: {input_folder}")

    folder_name = input_path.name
    image_files = sorted([
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix in SUPPORTED_IMAGE_EXTENSIONS
    ])
    total = len(image_files)

    if total == 0:
        console.info(f"No se encontraron imágenes en {input_folder}")
        return {'processed_files': [], 'failed_files': [], 'results_by_file': {},
                'total_files': 0, 'successful_count': 0, 'failed_count': 0}

    console.section(f"Encontradas {total} imagen(es) (batch)")

    detection_model = None
    console.info("Cargando modelo de detección...")
    try:
        detection_model = load_bubble_detection_model(silent=console.quiet)
    except Exception as e:
        console.error(f"No se pudo cargar el modelo: {e}")
        raise

    image_paths = [str(f) for f in image_files]
    processed, failed, results_by_file = [], [], {}

    try:
        batch_results = translate_manga_page_batch(
            image_paths, output_folder, config, console,
            detection_model=detection_model,
            output_subfolder=folder_name,
            batch_amount=batch_amount,
        )
        for p, r in batch_results.items():
            processed.append(p)
            results_by_file[p] = r
    except Exception as e:
        for p in image_paths:
            if p not in processed:
                failed.append((p, str(e)))
        console.error(f"Error en batch: {e}")
        raise

    console.section("Carpeta procesada (batch)")
    console.info(f"Total: {total}")
    console.success(f"Procesadas: {len(processed)}")
    if failed:
        console.error(f"Fallidas: {len(failed)}")
        for f, e in failed:
            console.print(f"  - {Path(f).name}: {e}")

    return {
        'processed_files': processed,
        'failed_files': failed,
        'results_by_file': results_by_file,
        'total_files': total,
        'successful_count': len(processed),
        'failed_count': len(failed),
    }