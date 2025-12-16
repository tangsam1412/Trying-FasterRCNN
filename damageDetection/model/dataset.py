import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class CoCoDataSet(Dataset):
    def __init__(self, image_dir, annotation_file, max_images=None, target_size=512):
        """
        image_dir: folder chứa images
        annotation_file: file COCO json
        max_images: giới hạn số ảnh (None = load hết)
        target_size: resize ảnh về (target_size x target_size)
        """
        self.image_dir = image_dir
        self.target_size = target_size

        print("Loading COCO annotations...")
        with open(annotation_file, "r") as f:
            coco = json.load(f)

        # categories → remap id liên tục 1..N
        categories = sorted(coco["categories"], key=lambda x: x["id"])
        self.cat_id_map = {cat["id"]: idx + 1 for idx, cat in enumerate(categories)}

        # image_id -> file_name
        self.images = {img["id"]: img["file_name"] for img in coco["images"]}

        # image_id -> list annotation
        self.annotations = {}
        for ann in tqdm(coco["annotations"], desc="Parsing annotations"):
            self.annotations.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images.keys())
        if max_images is not None:
            self.image_ids = self.image_ids[:max_images]

        print(f"COCO loaded: {len(self.image_ids)} images")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        file_name = self.images[img_id]
        img_path = os.path.join(self.image_dir, file_name)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # ===== RESIZE IMAGE =====
        scale_x = self.target_size / w
        scale_y = self.target_size / h
        img = cv2.resize(img, (self.target_size, self.target_size))

        # ===== PROCESS BBOX =====
        boxes = []
        labels = []

        for ann in self.annotations.get(img_id, []):
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue

            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + bw) * scale_x
            y2 = (y + bh) * scale_y

            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_map[ann["category_id"]])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # ===== IMAGE TO TENSOR =====
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
