
"""
normalizator pt. imaginile cu sobolani
primeste imaginile din folderul (1) specificat
le face alb negru (un singur canal) si le aduce la aceeasi dimensiune
apoi sa le scuipe in folderul (2) dorit
"""

import os
import glob
from PIL import Image
import numpy as np

input_dir = "input/dir"
output_dir = "output/dir"

OUT_SIZE = 256

def process_image(in_path, out_dir, size=OUT_SIZE):
    base_name = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(out_dir, base_name + ".jpg")

    with Image.open(in_path) as im:
        im = im.convert("L")
        im = im.resize((size, size), Image.LANCZOS)
        im_array = np.array(im) / 255.0
        normalized_im = Image.fromarray((im_array * 255).astype(np.uint8))
        normalized_im.save(out_path, format="JPEG", quality=95, optimize=True)

    return out_path

def main():
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory does not exist: {input_dir!r}")

    os.makedirs(output_dir, exist_ok=True)

    patterns = [
        "*.png", "*.PNG",
        "*.jpg", "*.JPG",
        "*.jpeg", "*.JPEG",
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_dir, p)))

    if not files:
        print("No PNG/JPG/JPEG files found in the input directory.")
        return

    print(f"Processing {len(files)} files...")
    for f in files:
        try:
            saved = process_image(f, output_dir)
            print("Saved:", saved)
        except Exception as e:
            print(f"Failed processing {f}: {e}")

if __name__ == "__main__":
    main()
