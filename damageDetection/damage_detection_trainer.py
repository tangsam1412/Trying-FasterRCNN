import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.dataset import DamageDataset
import numpy as np


class DamageDataset(torch.utils.data.Dataset):
    """Dataset wrapper để chuyển output từ CoCoDataSet → format FasterRCNN."""

    def __init__(self, images_root, coco_dataset):
        self.images_root = images_root
        self.coco_ds = coco_dataset

    def __len__(self):
        return len(self.coco_ds)

    def __getitem__(self, idx):
        img_np, boxes, labels, filename, img_id = self.coco_ds[idx]

        img = img_np.astype(np.float32) / 255.0
        img = preprocess_image(img)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }
        return img, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))


class DamageDetectionTrainer:
    def __init__(self, train_json, train_images, val_json, val_images,
                 n_epochs=1, model_path="trained_models/frcnn_damage.pt",
                 num_classes=9):

        self.train_json = train_json
        self.train_images = train_images
        self.val_json = val_json
        self.val_images = val_images

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_epochs = n_epochs
        self.writer = SummaryWriter()
        self.model_path = model_path
        self.num_classes = num_classes

        self._load_data()

    def _load_data(self):
        print("Loading COCO datasets...")

        # ===== TRAIN DATASET =====
        self.train_dataset = CoCoDataSet(
            image_dir="dataset/train/images",
            annotation_file="dataset/train/train.json",
            max_images=1000    # tăng dần sau
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=2,      # GPU Colab an toàn
            shuffle=True,
            num_workers=0,     # 🔥 BẮT BUỘC
            collate_fn=collate_fn
        )

        # ===== VAL DATASET =====
        self.val_dataset = CoCoDataSet(
            image_dir="dataset/val/images",
            annotation_file="dataset/val/val.json",
            max_images=200
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        print(f"Train images: {len(self.train_dataset)}")
        print(f"Val images:   {len(self.val_dataset)}")


    def _get_model(self):
        print("Loading FasterRCNN...")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def _train_batch(self, inputs, model, optimizer):
        model.train()
        imgs, targets = inputs

        imgs = [i.to(self.device) for i in imgs]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        losses = model(imgs, targets)
        loss = sum(losses.values())
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def _validate_batch(self, inputs, model):
        model.train()  # FasterRCNN yêu cầu train mode để tính loss
        imgs, targets = inputs

        imgs = [i.to(self.device) for i in imgs]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        losses = model(imgs, targets)
        loss = sum(losses.values())
        return loss.item()

    def train_and_validate(self):
        model = self._get_model().to(self.device)
        optimizer = SGD(model.parameters(), lr=0.005,
                        momentum=0.9, weight_decay=0.0005)

        print("Training start...")
        best_val = float("inf")

        for epoch in range(self.n_epochs):
            print(f"\n===== Epoch {epoch+1}/{self.n_epochs} =====")
            train_losses = []
            val_losses = []

            for batch in tqdm(self.train_loader, desc="Training"):
                train_losses.append(self._train_batch(batch, model, optimizer))

            avg_train = sum(train_losses) / len(train_losses)
            print(f"Train Loss: {avg_train:.4f}")

            for batch in tqdm(self.val_loader, desc="Validating"):
                val_losses.append(self._validate_batch(batch, model))

            avg_val = sum(val_losses) / len(val_losses)
            print(f"Val Loss: {avg_val:.4f}")

            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(), self.model_path)
                print("Saved best model!")

        print("Training completed!")
        self.writer.close()
