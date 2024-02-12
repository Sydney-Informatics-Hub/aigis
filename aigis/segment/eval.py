import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

class SegmentationModelEvaluator:
    """
    Class for evaluating a segmentation model.

    Args:
        model (torch.nn.Module): The segmentation model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader for the evaluation dataset.
        device (str, optional): The device to run the evaluation on (default: 'cuda').

    Attributes:
        model (torch.nn.Module): The segmentation model being evaluated.
        data_loader (torch.utils.data.DataLoader): The data loader for the evaluation dataset.
        device (str): The device used for evaluation.

    """

    def __init__(self, model, data_loader, device='cuda'):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def evaluate(self):
        """
        Evaluate the segmentation model using COCOEvaluator.

        Returns:
            None

        """
        evaluator = COCOEvaluator("dataset_name", cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(cfg, "dataset_name")
        inference_on_dataset(self.model, val_loader, evaluator)

    def calculate_mean_iou(self):
        """
        Calculate the mean Intersection over Union (IoU) for the segmentation model.

        Returns:
            float: The mean IoU.

        """
        self.model.eval()
        with torch.no_grad():
            ious = []
            for images, targets in self.data_loader:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = self.model(images)
                pred_masks = outputs['instances'].pred_masks.cpu().numpy()
                gt_masks = [t['masks'].cpu().numpy() for t in targets]

                iou = self._compute_iou(pred_masks, gt_masks)
                ious.append(iou)

            mean_iou = np.mean(ious)
            return mean_iou

    def _compute_iou(self, pred_masks, gt_masks):
        """
        Compute the Intersection over Union (IoU) between predicted masks and ground truth masks.

        Args:
            pred_masks (numpy.ndarray): Predicted masks.
            gt_masks (list): List of ground truth masks.

        Returns:
            float: The IoU.

        """
        intersection = np.logical_and(pred_masks, gt_masks)
        union = np.logical_or(pred_masks, gt_masks)
        iou = np.sum(intersection) / np.sum(union)
        return iou
    

def plot_confusion_matrix(self):
    """
    Plot the confusion matrix of the segmentation model evaluation.

    Returns:
        None

    """
    self.model.eval()
    with torch.no_grad():
        true_labels = []
        predicted_labels = []
        for images, targets in self.data_loader:
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            outputs = self.model(images)
            pred_masks = outputs['instances'].pred_masks.cpu().numpy()
            gt_masks = [t['masks'].cpu().numpy() for t in targets]

            true_labels.extend([1 if np.any(mask) else 0 for mask in gt_masks])
            predicted_labels.extend([1 if np.any(mask) else 0 for mask in pred_masks])

        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()