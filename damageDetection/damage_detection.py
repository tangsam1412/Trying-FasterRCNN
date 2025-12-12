import argparse
import os
from damage_detection_trainer import DamageDetectionTrainer
from inference import inference_model

def main(train=False, infer=False, infer_source="", infer_dest=""):
    train_coco = "dataset/train/train.json"
    train_images = "dataset/train/images"

    val_coco = "dataset/val/val.json"
    val_images = "dataset/val/images"

    model_path = "trained_models/frcnn_damage.pt"

    if train:
        print("Training using COCO dataset...")
        trainer = DamageDetectionTrainer(
            train_json=train_coco,
            train_images=train_images,
            val_json=val_coco,
            val_images=val_images,
            model_path=model_path
        )
        trainer.train_and_validate()

    if infer:
        print("Running inference...")
        inference_model(
            infer_source=infer_source,
            infer_dest=infer_dest,
            model_path=model_path
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--infer", action="store_true", help="Run inference")
    parser.add_argument("-infer_source", type=str, default="")
    parser.add_argument("-infer_dest", type=str, default="")
    args = parser.parse_args()

    main(
        train=args.train,
        infer=args.infer,
        infer_source=args.infer_source,
        infer_dest=args.infer_dest,
    )