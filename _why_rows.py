import json
from pathlib import Path
import numpy as np
from PIL import Image

nb = json.loads(Path("ocr.ipynb").read_text(encoding="utf-8"))
src = "".join(next(c["source"] for c in nb["cells"] if c.get("id") == "layout-mark"))
src = src.split("import matplotlib.pyplot as plt")[0]
ns = {}
exec(src, ns)
predictor = ns["ocr_predictor"]("db_resnet50", "crnn_vgg16_bn", pretrained=True, assume_straight_pages=True).cuda().eval()
names = [
    "08410076601919_C4N1_700-8802161.png",
    "08410076610355_C4N1_2200907.png",
    "08410076801753_C4N1_700-8805387.png",
    "00016000178496_S_s00_CPV101.png",
]
for name in names:
    image = Image.open(Path("outputs/panels") / name).convert("RGB")
    page = predictor([np.asarray(image)]).pages[0]
    found = ns["extract_headers"](page)
    columns = [c for c in found["columns"] if c["header"].strip() and not ns["is_serving_line"](c["header"])]
    header_bottom = max(c["header_box"][3] for c in columns) if columns else 0
    print("\n====", name, "header_bottom", round(header_bottom), "cols", len(columns))
    for row in found["rows"]:
        text = row["text"]
        values = [w["text"] for w in row["words"] if ns["is_value"](w["text"])]
        flag = []
        if ns["is_serving_line"](text):
            flag.append("serving")
        if text.lower().strip().startswith("nutrition"):
            flag.append("title")
        if text.strip().startswith("*"):
            flag.append("footnote")
        if row["y1"] <= header_bottom:
            flag.append("above-header")
        if not values:
            flag.append("no-value")
        print(f"  y {row['y0']:.0f}-{row['y1']:.0f} {flag} | {text[:110]}")
