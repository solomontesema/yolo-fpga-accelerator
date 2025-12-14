"""Generate label PNGs for printable ASCII characters using ImageMagick."""

from __future__ import annotations

import string
import subprocess
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

# Use a widely available default font to avoid missing-font warnings.
FONT = "DejaVu-Sans"
POINT_SIZES = (12, 24, 36, 48, 60, 72, 84, 96)
OUTPUT_DIR = Path(__file__).resolve().parent
SKIP_CODES = {9, 10, 11, 12, 13, 14}


def create_label_image(
    character: str,
    point_size: int,
    font: str = FONT,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Render a single character to a PNG file and return the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ord(character)}_{point_size // 12 - 1}.png"

    # ImageMagick reads from files when text starts with '@', so escape a few
    # characters to ensure they are rendered literally.
    label_text = character
    if character == " ":
        label_text = r"\ "
    elif character == "@":
        label_text = r"\@"
    elif character == "\\":
        label_text = r"\\\\"

    command = [
        "convert",
        "-fill",
        "black",
        "-background",
        "white",
        "-bordercolor",
        "white",
        "-font",
        font,
        "-pointsize",
        str(point_size),
        f"label:{label_text}",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ImageMagick `convert` command is required but not installed or not on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"`convert` failed for {character!r} at size {point_size}."
        ) from exc

    return output_path


def make_labels(
    point_size: int, font: str = FONT, output_dir: Path = OUTPUT_DIR
) -> None:
    """Generate PNG label images for printable ASCII characters at a given point size."""
    printable_chars = [c for c in string.printable if ord(c) not in SKIP_CODES]
    for character in tqdm(
        printable_chars, desc=f"Chars @ {point_size}pt", leave=False
    ):
        create_label_image(character, point_size, font=font, output_dir=output_dir)


def generate_all_labels(
    point_sizes: Iterable[int] = POINT_SIZES,
    font: str = FONT,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generate PNG label images for each configured font size."""
    for point_size in tqdm(point_sizes, desc="Point sizes"):
        make_labels(point_size, font=font, output_dir=output_dir)


if __name__ == "__main__":
    generate_all_labels()
