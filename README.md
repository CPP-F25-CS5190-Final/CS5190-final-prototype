# Real-Time Fire & Smoke Detection (Prototype)

This prototype shows the end-to-end pipeline your teammate described:

- **Input:** a path (single image or a folder of images)
- **Detection engine:**  
  - **Mock mode** (default for now): no weights required. Uses simple HSV heuristics to find likely fire/smoke areas.
  - **YOLOv8 mode** (later): plug in your trained `best.pt` to get real detections.
- **Output (console):** JSON with detections formatted as  
  `[x1, x2, y1, y2, confidence, class]`
- **Output (files):** annotated images saved to `outputs/`

This lets everyone agree on I/O, CLI, drawing, and logging **now**, and swap to real YOLO weights **later** with one flag change.

---

## Features

- Accepts **file or folder** paths.
- Prints **JSON** detections per image.
- Draws **bounding boxes + labels** and saves annotated images.
- Optional window preview: show **original** then **annotated** (press space/enter to advance, ESC to quit).
- Drop-in YOLOv8 support once weights are ready.

---

## Repo layout (suggested)

```
.
├─ detect_path.py          # CLI tool (mock + YOLO modes)
├─ requirements.txt
├─ data/                   # put sample images here (not committed if large)
└─ outputs/                # annotated results (auto-created)
```

---

## Prerequisites

- **Python** 3.9–3.12 (recommended 3.10+)
- OS: macOS / Linux / Windows
- (Optional) **GPU + CUDA** for YOLOv8 later

---

## 1) Set up environment

Create and activate a virtual environment:

**macOS/Linux (bash/zsh)**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 2) Install dependencies

Make sure `requirements.txt` contains:

```
opencv-python
numpy
ultralytics
```

Then install:
```bash
pip install -r requirements.txt
```

> Note: YOLOv8 is installed now so the team won’t need to change the env later when `best.pt` arrives. You can still run the prototype without weights using `--mock`.

---

## 3) Add a test image

Put one or more images into the `data/` folder, e.g.:

```
data/
  fire_1.jpg
```

(Any of: `.jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp`)

---

## 4) Run the prototype (no weights yet)

**Preview original, then process and show annotated**:

```bash
python detect_path.py --path data --mock --show
```

- First window: **Original** (press space/enter to continue, ESC to quit).
- Second window: **Annotated** (press space/enter for next image, ESC to quit).
- Outputs saved to `outputs/<name>_annotated.jpg`
- Console prints one JSON line per image, for example:
  ```json
  {
    "image": "data/fire_1.jpg",
    "detections": [[120, 980, 210, 690, 0.88, "fire"]]
  }
  ```

If you don’t want windows to pop up:
```bash
python detect_path.py --path data --mock
```

---

## 5) Switch to YOLOv8 (when weights are ready)

Once training provides a weights file (e.g., `runs/detect/train/weights/best.pt`):

```bash
python detect_path.py --path data   --weights runs/detect/train/weights/best.pt   --conf 0.35 --iou 0.5 --show
```

- `--classes fire smoke` are default; leave as-is or customize.
- Add `--device cuda:0` to force GPU (if available), or use CPU by default.

---

## CLI options

```
--path       (required)  Image file or directory
--mock                    Use heuristic detector (no weights)
--weights                 YOLO .pt file (omit if --mock)
--conf      0.35          Confidence threshold (YOLO mode)
--iou       0.5           IoU/NMS threshold (YOLO mode)
--device    auto          e.g., auto | cpu | cuda:0
--classes   fire smoke    Class whitelist (YOLO mode)
--outdir    outputs       Output folder for annotated images
--show                    Show Original then Annotated windows
```

---

## Output format (contract)

Each detection is:
```
[x1, x2, y1, y2, confidence, class]
```
- Coordinates are integer **pixel** indices in the original image space.
- `confidence` is a float in `[0,1]` (shown as % on the annotation).
- `class` is `"fire"` or `"smoke"` (by default).

> Note: This ordering (`x1, x2, y1, y2`) is kept to match your teammate’s request, even though many libraries use `(x1, y1, x2, y2)`. Be consistent throughout the project.

---

## Troubleshooting

- **No images found**  
  Ensure `--path` points to an image file or a folder with supported extensions.

- **Windows: OpenCV window doesn’t respond**  
  Click the image window to focus it before pressing keys. Use `--show` only when you need visual confirmation.

- **Headless environments (e.g., servers/SSH)**  
  Omit `--show` and check `outputs/` + console JSON.

- **YOLO mode is slow on CPU**  
  That’s expected. For real-time testing, use a machine with a CUDA-capable GPU and run with `--device cuda:0`.

---

## What to build next (team discussion)

- Add **video/webcam** support (`--source` with videos or `0` for webcam).
- **CSV/JSON logging**: write detections with timestamps.
- **RTSP** camera input for real streams.
- **Simple UI** (Streamlit/Gradio) that wraps the same detector class.
- Export to **ONNX/TensorRT** if you need higher FPS on edge devices.

---

## License

Internal academic project prototype. Add a license if you plan to share publicly.
