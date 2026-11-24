import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

nb = json.loads(Path("ocr.ipynb").read_text(encoding="utf-8"))
src = "".join(next(c["source"] for c in nb["cells"] if c.get("id") == "layout-mark"))
src = src.split("panel_dir = Path")[0]
ns = {"__name__": "nb"}
exec(src, ns)
predictor = ns["ocr_predictor"]("db_resnet50", "crnn_vgg16_bn", pretrained=True, assume_straight_pages=True).cuda().eval()
ns["predictor"] = predictor
names = [
    "08410076601919_C4N1_700-8802161.png",
    "08410076610355_C4N1_2200907.png",
    "08410076801753_C4N1_700-8805387.png",
    "00016000178496_S_s00_CPV101.png",
]
out = Path("outputs/marks")
out.mkdir(exist_ok=True)
for name in names:
    image = Image.open(Path("outputs/panels") / name).convert("RGB")
    page = predictor([np.asarray(image)]).pages[0]
    found = ns["extract_headers"](page)
    columns = [c for c in found["columns"] if c["header"].strip() and not ns["is_serving_line"](c["header"])]
    columns = ns["split_wide_header"](columns, found["rows"], image.width)
    labels = ns["row_labels_from"](found["rows"], columns)
    canvas = image.copy()
    pen = ImageDraw.Draw(canvas)
    for column in columns:
        pen.rectangle(column["header_box"], outline=(20, 150, 60), width=4)
    for label in labels:
        pen.rectangle(label["box"], outline=(30, 90, 220), width=3)
    canvas.save(out / name)
    print("\n====", name)
    for i, column in enumerate(columns, 1):
        print(f"  col {i}: {column['header']}")
    for label in labels:
        print(f"  row: {label['text']}")
