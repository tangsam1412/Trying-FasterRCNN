import os
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_model(model_path, num_classes=3, device="cuda"):
    """Load trained FasterRCNN model"""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def inference_model(infer_source, infer_dest, model_path, num_classes=3):
    """Run inference on a folder of images."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(infer_dest):
        os.makedirs(infer_dest)

    model = load_model(model_path, num_classes=num_classes, device=device)

    class_names = {1: "class_1", 2: "class_2"}  # chỉnh theo COCO categories của bạn

    print(f"Running inference on: {infer_source}")
    print(f"Saving results to:     {infer_dest}")

    for img_file in os.listdir(infer_source):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(infer_source, img_file)
        img = cv2.imread(img_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # convert to tensor
        t_img = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(t_img)[0]

        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()

        # draw results
        for box, score, label in zip(boxes, scores, labels):
            if score < 0.4:
                continue

            x1, y1, x2, y2 = box.astype(int)
            name = class_names.get(int(label), f"id_{label}")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{name} {score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        out_path = os.path.join(infer_dest, img_file)
        cv2.imwrite(out_path, img)

    print("Inference completed.")