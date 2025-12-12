import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# -------------------------------
# CoCoDataSet: đọc file COCO JSON
# -------------------------------

class CoCoDataSet(Dataset):
    def __init__(self, images_root, annotations):
        self.images_root = images_root

        with open(annotations, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        # gom annotation theo image_id
        ann_dict = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            ann_dict.setdefault(img_id, []).append(ann)

        # preload ảnh
        self.imageData = []

        for img in self.images:
            img_id = img["id"]
            filename = img["file_name"]

            path = f"{self.images_root}/{filename}"
            im = cv2.imread(path)

            if im is None:
                continue

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            boxes = []
            labels = []

            # lấy bbox
            if img_id in ann_dict:
                for ann in ann_dict[img_id]:
                    x, y, w, h = ann["bbox"]

                    # bỏ box w/h <= 1
                    if w <= 1 or h <= 1:
                        continue

                    boxes.append([x, y, w, h])
                    labels.append(ann["category_id"])

            if len(boxes) == 0:
                continue

            self.imageData.append({
                "image": im,
                "boxes": boxes,
                "labels": labels,
                "filename": filename,
                "image_id": img_id,
            })

    def __len__(self):
        return len(self.imageData)

    def __getitem__(self, index):
        d = self.imageData[index]
        return d["image"], d["boxes"], d["labels"], d["filename"], d["image_id"]


# -------------------------------
# Preprocessing
# -------------------------------

def preprocess(image_np):
    img = image_np.astype(np.float32) / 255.0
    return torch.tensor(img).permute(2, 0, 1)

# Giữ tên hàm cũ để trainer import được
def preprocess_image(image_np):
    return preprocess(image_np)


# -------------------------------
# DamageDataset cho FasterRCNN
# -------------------------------

class DamageDataset(Dataset):
    def __init__(self, coco_dataset):
        self.coco_ds = coco_dataset

    def __len__(self):
        return len(self.coco_ds)

    def __getitem__(self, idx):
        img_np, boxes, labels, filename, img_id = self.coco_ds[idx]

        img = preprocess_image(img_np)

        # convert COCO bbox → XYXY
        xyxy = []
        for (x, y, w, h) in boxes:
            xyxy.append([x, y, x + w, y + h])

        boxes = torch.tensor(xyxy, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }

        return img, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))
