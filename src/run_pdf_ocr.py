import argparse
import importlib
import importlib.util
import io
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import fitz  # PyMuPDF
import img2pdf
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from transformers import AutoModel, AutoTokenizer


DEFAULT_CONFIG: Dict[str, object] = {
    "MODEL_PATH": "./models",
    "INPUT_PATH": "input/sample.pdf",
    "OUTPUT_PATH": "./output",
    "PROMPT": "<image>\n<|grounding|>Convert the document to markdown. ",
    "SKIP_REPEAT": True,
    "CROP_MODE": True,
    "DPI": 144,
    "BASE_SIZE": 1024,
    "IMAGE_SIZE": 640,
    "TEST_COMPRESS": False,
}


class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


def load_config() -> Dict[str, object]:
    config = DEFAULT_CONFIG.copy()
    if importlib.util.find_spec("config"):
        user_config = importlib.import_module("config")
        for key in config:
            if hasattr(user_config, key):
                config[key] = getattr(user_config, key)
    return config


def parse_args(config: Dict[str, object]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek OCR on a PDF file.")
    parser.add_argument("--input", dest="input_path", default=config["INPUT_PATH"])
    parser.add_argument("--output", dest="output_path", default=config["OUTPUT_PATH"])
    parser.add_argument("--model", dest="model_path", default=config["MODEL_PATH"])
    parser.add_argument("--prompt", default=config["PROMPT"])
    parser.add_argument(
        "--skip-repeat",
        action=argparse.BooleanOptionalAction,
        default=config["SKIP_REPEAT"],
        help="Skip pages that do not emit the expected end token.",
    )
    parser.add_argument(
        "--crop-mode",
        action=argparse.BooleanOptionalAction,
        default=config["CROP_MODE"],
    )
    parser.add_argument("--dpi", type=int, default=config["DPI"])
    parser.add_argument(
        "--base-size",
        type=int,
        default=config["BASE_SIZE"],
        help="Base resolution used by the model.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=config["IMAGE_SIZE"],
        help="Image resolution for local crops.",
    )
    parser.add_argument(
        "--test-compress",
        action=argparse.BooleanOptionalAction,
        default=config["TEST_COMPRESS"],
        help="Enable model's compression ratio logging.",
    )
    return parser.parse_args()


def pdf_to_images_high_quality(pdf_path: str, dpi: int = 144, image_format: str = "PNG") -> List[Image.Image]:
    images: List[Image.Image] = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        if image_format.upper() != "PNG" and img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    pdf_document.close()
    return images


def pil_to_pdf_img2pdf(pil_images: Sequence[Image.Image], output_path: str) -> None:
    if not pil_images:
        return

    image_bytes_list: List[bytes] = []
    for img in pil_images:
        img_buffer = io.BytesIO()
        img.convert("RGB").save(img_buffer, format="JPEG", quality=95)
        image_bytes_list.append(img_buffer.getvalue())

    pdf_bytes = img2pdf.convert(image_bytes_list)
    with open(output_path, "wb") as file:
        file.write(pdf_bytes)


def re_match(text: str) -> Tuple[List[Tuple[str, str, str]], List[str], List[str]]:
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    matches_image = [match[0] for match in matches if "<|ref|>image<|/ref|>" in match[0]]
    matches_other = [match[0] for match in matches if "<|ref|>image<|/ref|>" not in match[0]]
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text: Tuple[str, str, str], image_width: int, image_height: int):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as exc:
        print(exc)
        return None
    return label_type, cor_list


def draw_bounding_boxes(image: Image.Image, refs: Iterable[Tuple[str, str, str]], prefix_index: int, output_path: Path) -> Image.Image:
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    img_idx = 0

    for ref in refs:
        result = extract_coordinates_and_label(ref, image_width, image_height)
        if not result:
            continue
        label_type, points_list = result
        color = (
            int(np.random.randint(0, 200)),
            int(np.random.randint(0, 200)),
            int(np.random.randint(0, 255)),
        )
        color_with_alpha = color + (20,)

        for points in points_list:
            x1, y1, x2, y2 = points
            x1 = int(x1 / 999 * image_width)
            y1 = int(y1 / 999 * image_height)
            x2 = int(x2 / 999 * image_width)
            y2 = int(y2 / 999 * image_height)

            if label_type == "image":
                cropped = image.crop((x1, y1, x2, y2))
                cropped.save(output_path / f"{prefix_index}_{img_idx}.jpg")
                img_idx += 1

            rect_width = 4 if label_type == "title" else 2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=rect_width)
            draw_overlay.rectangle([x1, y1, x2, y2], fill=color_with_alpha, outline=None, width=1)
            text_x = x1
            text_y = max(0, y1 - 15)
            text_bbox = draw.textbbox((0, 0), label_type, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=(255, 255, 255, 30))
            draw.text((text_x, text_y), label_type, font=font, fill=color)

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def ensure_env() -> None:
    if torch.version.cuda == "11.8":
        os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-11.8/bin/ptxas")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)
    return tokenizer, model


def run_pdf_ocr(args: argparse.Namespace) -> None:
    ensure_env()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"{Colors.RED}PDF loading .....{Colors.RESET}")
    images = pdf_to_images_high_quality(str(input_path), dpi=args.dpi)
    print(f"{Colors.GREEN}Loaded {len(images)} page images{Colors.RESET}")

    tokenizer, model = load_model(args.model_path)

    tmp_dir = output_path / "tmp_pages"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem
    mmd_det_path = output_path / f"{base_name}_det.mmd"
    mmd_path = output_path / f"{base_name}.mmd"
    pdf_out_path = output_path / f"{base_name}_layouts.pdf"

    contents_det = ""
    contents = ""
    draw_images: List[Image.Image] = []

    for page_index, img in enumerate(images):
        page_tmp_path = tmp_dir / f"{base_name}_{page_index:04d}.jpg"
        img.save(page_tmp_path, format="JPEG", quality=95)
        page_out_dir = tmp_dir / f"{base_name}_{page_index:04d}_out"
        text = model.infer(
            tokenizer,
            prompt=args.prompt,
            image_file=str(page_tmp_path),
            output_path=str(page_out_dir),
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=args.crop_mode,
            save_results=False,
            test_compress=args.test_compress,
            eval_mode=True,
        )

        if text is None:
            if args.skip_repeat:
                continue
            text = ""

        raw_text = text
        contents_det += f"{raw_text}\n<--- Page Split --->\n"
        image_draw = img.copy()
        matches_ref, matches_images, matches_other = re_match(raw_text)
        result_image = draw_bounding_boxes(image_draw, matches_ref, page_index, images_dir)
        draw_images.append(result_image)

        for idx, match_image in enumerate(matches_images):
            text = text.replace(match_image, f"![](images/{page_index}_{idx}.jpg)\n")

        for match_other in matches_other:
            text = (
                text.replace(match_other, "")
                .replace("\\coloneqq", ":=")
                .replace("\\eqqcolon", "=:")
                .replace("\n\n\n\n", "\n\n")
                .replace("\n\n\n", "\n\n")
            )

        contents += f"{text}\n<--- Page Split --->\n"

        page_clean_path = output_path / f"{base_name}_{page_index:04d}.mmd"
        page_det_path = output_path / f"{base_name}_{page_index:04d}_det.mmd"
        page_det_path.write_text(raw_text, encoding="utf-8")
        page_clean_path.write_text(text, encoding="utf-8")

    mmd_det_path.write_text(contents_det, encoding="utf-8")
    mmd_path.write_text(contents, encoding="utf-8")
    (output_path / "result_det.mmd").write_text(contents_det, encoding="utf-8")
    (output_path / "result.mmd").write_text(contents, encoding="utf-8")
    pil_to_pdf_img2pdf(draw_images, str(pdf_out_path))
    print(f"{Colors.BLUE}Results saved to {output_path}{Colors.RESET}")


def main() -> None:
    config = load_config()
    args = parse_args(config)
    run_pdf_ocr(args)


if __name__ == "__main__":
    main()
