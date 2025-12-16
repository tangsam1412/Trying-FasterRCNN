import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class CoCoDataSet(Dataset):
    def __init__(self, image_dir, annotation_file, max_images=None):
        """
        image_dir: folder chứa images
        annotation_file: file COCO json
        max_images: giới hạn số ảnh (None = load hết)
        """

        self.image_dir = image_dir

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
            if ann["image_id"] not in self.annotations:
                self.annotations[ann["image_id"]] = []
            self.annotations[ann["image_id"]].append(ann)

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
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)

        boxes = []
        labels = []

        for ann in self.annotations.get(img_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_map[ann["category_id"]])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
