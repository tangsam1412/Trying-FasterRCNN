import torch
import cv2
import os
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ==========================
# CONFIG
# ==========================

# Dataset của bạn có 8 lớp + 1 background → num_classes = 9
NUM_CLASSES = 9

# Tên class theo train_300_remap.json
CLASS_NAMES = {
    1: "concave",
    2: "axis",
    3: "container",
    4: "dentado",
    5: "perforation",
    6: "mildew",
    7: "puncture",
    8: "scratch",
}


# ==========================
# LOAD MODEL
# ==========================

def load_model(model_path, num_classes=NUM_CLASSES):
    print(f"Loading model from: {model_path}")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Model loaded. Device: {device}")
    return model


# ==========================
# PREPROCESS
# ==========================

def preprocess(img):
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).permute(2, 0, 1)


# ==========================
# INFERENCE 1 ẢNH
# ==========================

def inference_single(model, img_path, conf=0.5):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot open image: {img_path}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    device = next(model.parameters()).device

    x = preprocess(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)[0]

    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    # vẽ bbox
    for box, score, label in zip(boxes, scores, labels):
        if score < conf:
            continue

        x1, y1, x2, y2 = map(int, box)

        cls_name = CLASS_NAMES.get(label, f"cls_{label}")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{cls_name} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return img


# ==========================
# INFERENCE FOLDER
# ==========================

def inference_folder(model_path, src_folder, out_folder, conf=0.5):
    os.makedirs(out_folder, exist_ok=True)

    model = load_model(model_path)

    files = [
        f for f in os.listdir(src_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Running inference on {len(files)} images...")

    for f in files:
        inp = os.path.join(src_folder, f)
        out = os.path.join(out_folder, f)

        result = inference_single(model, inp, conf)
        cv2.imwrite(out, result)

    print("Inference completed!")


# ==========================
# API ĐỂ GỌI TỪ damage_detection.py
# ==========================

def inference_model(infer_source, infer_dest, model_path, num_classes=NUM_CLASSES):
    if os.path.isfile(infer_source):
        # inference 1 ảnh
        os.makedirs(infer_dest, exist_ok=True)
        model = load_model(model_path, num_classes)
        result = inference_single(model, infer_source)
        out = os.path.join(infer_dest, os.path.basename(infer_source))
        cv2.imwrite(out, result)
        print("Saved:", out)

    else:
        # inference folder
        inference_folder(model_path, infer_source, infer_dest)
