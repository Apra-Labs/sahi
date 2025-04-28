# Code written by Kushal Jain, 2025.

import logging
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements
from sahi.utils.yolov9pytorch import non_max_suppression, xywh2xyxy

logger = logging.getLogger(__name__)

class Yolov9TorchDetectionModel(DetectionModel):
    def __init__(self, *args, iou_threshold: float = 0.7, device: Optional[str] = None, **kwargs):
        """
        Args:
            iou_threshold: float
                IOU threshold for non-max suppression, defaults to 0.7.
            device: str
                'cuda' or 'cpu'. If None, will auto-select cuda if available.
        """
        super().__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def check_dependencies(self) -> None:
        check_requirements(["torch"])
    
    def load_model(self, model_kwargs: Optional[dict] = {}) -> None:
        """Load a PyTorch model (.pt file)."""
        try:
            model = torch.load(self.model_path, map_location=self.device, weights_only=False)
            model.eval()
            self.set_model(model)
        except Exception as e:
            raise TypeError(f"model_path is not a valid PyTorch model path: {e}")
    
    def set_model(self, model: Any) -> None:
        """
            Sets the underlying PyTorch model.
        
            Args:
            model: Any
                A pytorch model
        """
        self.model = model
        if not self.category_mapping:
            raise TypeError("Category mapping values are required")
    
    def _preprocess_image(self, image: np.ndarray, input_shape: Tuple[int, int]) -> np.ndarray:
        """Prepapre image for inference by resizing, normalizing and changing dimensions.

        Args:
            image: np.ndarray
                Input image with color channel order RGB.
        """
        input_image = cv2.resize(image, input_shape)
        input_image = input_image / 255.0
        input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
        image_tensor = torch.from_numpy(input_image).float().unsqueeze(0).to(self.device)
        return image_tensor

    def _post_process(self, outputs: torch.Tensor, input_shape: Tuple[int, int], image_shape: Tuple[int, int]) -> List[torch.Tensor]:
        """Same as your original `_post_process` method."""
        image_h, image_w = image_shape
        input_w, input_h = input_shape

        predictions = outputs[0].cpu().numpy().T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.confidence_threshold, :]
        scores = scores[scores > self.confidence_threshold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = predictions[:, :4]

        # Scale boxes to original dimensions
        input_shape = np.array([input_w, input_h, input_w, input_h])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_w, image_h, image_w, image_h])
        boxes = boxes.astype(np.int32)

        # Convert from xywh to xyxy
        boxes = xywh2xyxy(boxes).round().astype(np.int32)

        # Perform non-max suppression
        indices = non_max_suppression(boxes, scores, self.iou_threshold)

        # Format results
        prediction_result = []
        for bbox, score, label in zip(boxes[indices], scores[indices], class_ids[indices]):
            bbox = bbox.tolist()
            cls_id = int(label)
            prediction_result.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cls_id])

        prediction_result = [torch.tensor(prediction_result)]
        return prediction_result

    def perform_inference(self, image: np.ndarray):
        """Perform prediction with PyTorch model."""
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        # Assume model expects a square input (e.g., (640,640))
        input_shape = (self.model.yaml.get("imgsz", 640), self.model.yaml.get("imgsz", 640))  # You might need to adjust how to get size
        image_shape = image.shape[:2]

        # Preprocess
        image_tensor = self._preprocess_image(image, input_shape)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Post-process
        prediction_results = self._post_process(outputs, input_shape, image_shape)
        self._original_predictions = prediction_results

    @property
    def category_names(self):
        return list(self.category_mapping.values())

    @property
    def num_categories(self):
        return len(self.category_mapping)

    @property
    def has_mask(self):
        return False

    def _create_object_prediction_list_from_original_predictions(self,
                                                                  shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
                                                                  full_shape_list: Optional[List[List[int]]] = None):
        """Same as your original."""
        # Same code as you already have
        original_predictions = self._original_predictions

        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            for prediction in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1, y1, x2, y2, score, category_id = prediction
                bbox = [max(0, x1), max(0, y1), max(0, x2), max(0, y2)]

                if full_shape:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"Ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=int(category_id),
                    score=score,
                    segmentation=None,
                    category_name=self.category_mapping[str(int(category_id))],
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
