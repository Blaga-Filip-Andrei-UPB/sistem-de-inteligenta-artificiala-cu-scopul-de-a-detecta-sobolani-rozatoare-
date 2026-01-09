import pandas as pd

INPUT_LABELS = "etichete poze.csv"
OUTPUT_LABELS = "etichete256.csv"

OUT_SIZE = 256

df = pd.read_csv(INPUT_LABELS)

def normalize_bbox(row):
    scale_x = OUT_SIZE / row["image_width"]
    scale_y = OUT_SIZE / row["image_height"]

    return pd.Series({
        "label_name": row["label_name"],
        "image_name": row["image_name"],

        "bbox_x": row["bbox_x"] * scale_x,
        "bbox_y": row["bbox_y"] * scale_y,
        "bbox_width": row["bbox_width"] * scale_x,
        "bbox_height": row["bbox_height"] * scale_y,

        "image_width": OUT_SIZE,
        "image_height": OUT_SIZE,
    })

df_norm = df.apply(normalize_bbox, axis=1)

df_norm.to_csv(OUTPUT_LABELS, index=False)

print("✔ etichetele au fost normalizate pentru imagini 256×256")
