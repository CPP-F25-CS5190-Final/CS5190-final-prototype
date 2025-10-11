import argparse, os, sys, glob, json
import numpy as np
import cv2

# -------- utilities --------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p): return os.path.splitext(p)[1].lower() in IMG_EXTS

def gather_images(path):
    if os.path.isdir(path):
        imgs = []
        for ext in IMG_EXTS:
            imgs.extend(glob.glob(os.path.join(path, f"*{ext}")))
        imgs.sort()
        return imgs
    elif os.path.isfile(path) and is_image(path):
        return [path]
    raise FileNotFoundError(f"No images found at: {path}")

def draw_detections(img, dets):
    for x1, x2, y1, y2, confv, cls_name in dets:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_name} {confv*100:.1f}%"
        cv2.putText(img, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return img

# -------- MOCK detector (no weights) --------
def find_contours(mask, min_area=500):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x + w, y + h))
    return boxes

def mock_detect_fire_smoke(img_bgr):
    # crude HSV heuristics
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    fire_mask1 = cv2.inRange(hsv, (5, 120, 120), (35, 255, 255))
    fire_mask2 = cv2.inRange(hsv, (0, 160, 160), (10, 255, 255))
    fire_mask = cv2.bitwise_or(fire_mask1, fire_mask2)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    s = hsv[:, :, 1]; v = hsv[:, :, 2]
    smoke_mask = cv2.inRange(s, 0, 60) & cv2.inRange(v, 120, 255)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    dets = []
    for (x1, y1, x2, y2) in find_contours(fire_mask, min_area=800):
        v_mean = float(np.mean(v[y1:y2, x1:x2])) / 255.0
        dets.append([x1, x2, y1, y2, min(0.99, 0.6 + 0.4 * v_mean), "fire"])
    for (x1, y1, x2, y2) in find_contours(smoke_mask, min_area=2000):
        s_mean = 1.0 - float(np.mean(s[y1:y2, x1:x2])) / 255.0
        dets.append([x1, x2, y1, y2, min(0.95, 0.5 + 0.5 * s_mean), "smoke"])
    return dets

# -------- YOLO wrapper (real weights) --------
class YoloDetector:
    def __init__(self, weights, device="auto", conf=0.35, iou=0.5, whitelist=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.model.to(device)
        self.conf, self.iou = conf, iou
        self.whitelist = set(w.lower() for w in whitelist) if whitelist else None

    def __call__(self, img_bgr):
        results = self.model.predict(source=img_bgr, conf=self.conf, iou=self.iou, verbose=False)
        r = results[0]
        if r.boxes is None:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        names = r.names
        out = []
        for (x1, y1, x2, y2), cid, cf in zip(boxes, clss, confs):
            cname = names[int(cid)]
            if self.whitelist and cname.lower() not in self.whitelist:
                continue
            out.append([int(x1), int(x2), int(y1), int(y2), float(cf), cname])
        return out

# -------- main --------
def main():
    ap = argparse.ArgumentParser("Fire/Smoke detection on images (preview then annotate)")
    ap.add_argument("--path", required=True, help="Image file or directory")
    ap.add_argument("--outdir", default="outputs", help="Where to save annotated images")
    ap.add_argument("--show", action="store_true",
                    help="Show windows (Original first, then Annotated)")
    # YOLO params
    ap.add_argument("--weights", help=".pt weights (omit if --mock)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--classes", nargs="*", default=["fire", "smoke"])
    # Mock flag
    ap.add_argument("--mock", action="store_true", help="Use heuristic detector (no weights)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    images = gather_images(args.path)

    if args.mock:
        detect_fn = mock_detect_fire_smoke
    else:
        if not args.weights:
            print("Error: provide --weights or use --mock", file=sys.stderr)
            sys.exit(1)
        detect_fn = YoloDetector(args.weights, args.device, args.conf, args.iou, args.classes)

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}", file=sys.stderr)
            continue

        # 1) Show ORIGINAL first (if --show)
        if args.show:
            cv2.imshow("Original", img)
            # press SPACE/ENTER to continue, or ESC to quit
            k = cv2.waitKey(0) & 0xFF
            if k == 27:  # ESC
                break

        # 2) Detect and print JSON to console
        dets = detect_fn(img)
        print(json.dumps({"image": img_path, "detections": dets}))

        # 3) Draw + save + show ANNOTATED
        annotated = draw_detections(img.copy(), dets)
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.outdir, f"{base}_annotated.jpg")
        cv2.imwrite(out_path, annotated)

        if args.show:
            cv2.imshow("Annotated", annotated)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                break

    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
