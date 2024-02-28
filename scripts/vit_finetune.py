import argparse
from sklearn.metrics import accuracy_score
import numpy as np
import transformers
import torch
import torchvision
from transformers import TrainingArguments, Trainer
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)


def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))


 def main(args):
    dataset = load_dataset("imagefolder", data_dir=args.data_dir)

    train_ds = dataset["train"]
    test_ds = dataset["test"]
    val_ds = dataset["validation"]

    id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}

    processor = ViTImageProcessor.from_pretrained(args.model_name)

    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    normalize = Normalize(mean=image_mean, std=image_std)

    # Set the transforms
    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)
    test_ds.set_transform(val_transforms)

    train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=args.train_batch_size)

    model = ViTForImageClassification.from_pretrained(args.model_name,
                                                      id2label=id2label,
                                                      label2id=label2id)

    metric_name = "accuracy"

    training_args = TrainingArguments(
        args.output_dir,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_dir=args.logging_dir,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    trainer.train()

    outputs = trainer.predict(test_ds)

    print(outputs.metrics)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)

    labels = train_ds.features['label'].names
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--data_dir", type=str, default="LCZs", help="Path to the dataset directory")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224-in21k", help="Pretrained model name")
    parser.add_argument("--train_batch_size", type=int, default=10, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory")

    args = parser.parse_args()
    main(args)
