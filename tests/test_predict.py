# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import os
import shutil
import sys
import unittest

import numpy as np
import pytest

from unittest.mock import MagicMock

from sahi.prediction import Category, ObjectPrediction, PredictionScore
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import read_image
from sahi.utils.ultralytics import UltralyticsTestConstants, download_yolo11n_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320


class TestPredict(unittest.TestCase):
    def test_prediction_score(self):
        from sahi.prediction import PredictionScore

        prediction_score = PredictionScore(np.array(0.6))  # type: ignore
        self.assertEqual(type(prediction_score.value), float)
        self.assertEqual(prediction_score.is_greater_than_threshold(0.5), True)
        self.assertEqual(prediction_score.is_greater_than_threshold(0.7), False)

    @pytest.mark.skipif(sys.version_info[:2] > (3, 10), reason="Requires Python 3.10 or lower")
    def test_get_prediction_mmdet(self):
        from sahi.models.mmdet import MmdetDetectionModel
        from sahi.predict import get_prediction
        from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_yolox_tiny_model

        # init model
        download_mmdet_yolox_tiny_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_YOLOX_TINY_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            image_size=IMAGE_SIZE,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image, detection_model=mmdet_detection_model, shift_amount=[0, 0], full_shape=None
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 2)
        num_person = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
        self.assertEqual(num_person, 0)
        num_truck = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "truck":
                num_truck += 1
        self.assertEqual(num_truck, 0)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_car, 2)

    def test_get_prediction_automodel_yolo11(self):
        from sahi.auto_model import AutoDetectionModel
        from sahi.predict import get_prediction
        from sahi.utils.ultralytics import UltralyticsTestConstants, download_yolo11n_model

        # init model
        download_yolo11n_model()

        yolo11_detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        yolo11_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image, detection_model=yolo11_detection_model, shift_amount=[0, 0], full_shape=None, postprocess=None
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertGreater(len(object_prediction_list), 0)
        num_person = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
        self.assertEqual(num_person, 0)
        num_truck = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "truck":
                num_truck += 1
        self.assertEqual(num_truck, 0)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertGreater(num_car, 0)

    @pytest.mark.skipif(sys.version_info[:2] > (3, 10), reason="Requires Python 3.10 or lower")
    def test_get_sliced_prediction_mmdet(self):
        from sahi.models.mmdet import MmdetDetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_yolox_tiny_model

        # init model
        download_mmdet_yolox_tiny_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_YOLOX_TINY_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"

        slice_height = 512
        slice_width = 512
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2
        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=mmdet_detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=False,
            postprocess_type=postprocess_type,
            postprocess_match_threshold=match_threshold,
            postprocess_match_metric=match_metric,
            postprocess_class_agnostic=class_agnostic,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 15)
        num_person = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
        self.assertEqual(num_person, 0)
        num_truck = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "truck":
                num_truck += 1
        self.assertEqual(num_truck, 0)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_car, 15)

    @pytest.mark.skipif(sys.version_info[:2] > (3, 10), reason="Requires Python 3.10 or lower")
    def test_get_sliced_prediction_mmdet_no_standard_pred(self):
        """Test sliced prediction without standard pred."""
        from sahi.models.mmdet import MmdetDetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_yolox_tiny_model

        # init model
        download_mmdet_yolox_tiny_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_YOLOX_TINY_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"

        slice_height = 512
        slice_width = 512
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2
        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=mmdet_detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=False,  # Explicitly False
            postprocess_type=postprocess_type,
            postprocess_match_threshold=match_threshold,
            postprocess_match_metric=match_metric,
            postprocess_class_agnostic=class_agnostic,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare - should be same as test_get_sliced_prediction_mmdet
        self.assertEqual(len(object_prediction_list), 15)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_car, 15)

    @pytest.mark.skipif(sys.version_info[:2] > (3, 10), reason="Requires Python 3.10 or lower")
    def test_get_sliced_prediction_mmdet_with_standard_pred(self):
        """Test sliced prediction with standard pred."""
        from sahi.models.mmdet import MmdetDetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_yolox_tiny_model

        # init model
        download_mmdet_yolox_tiny_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_YOLOX_TINY_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"

        slice_height = 512
        slice_width = 512
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2
        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=mmdet_detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=True,  # Explicitly True
            postprocess_type=postprocess_type,
            postprocess_match_threshold=match_threshold,
            postprocess_match_metric=match_metric,
            postprocess_class_agnostic=class_agnostic,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare - Number might change slightly due to NMM/NMS merging standard preds
        # We expect at least the number from standard prediction and likely more from slicing
        self.assertGreaterEqual(len(object_prediction_list), 2) # Standard pred found 2 cars
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertGreaterEqual(num_car, 2)


    def test_get_prediction_yolo11(self):
        from sahi.models.ultralytics import UltralyticsDetectionModel
        # init model
        download_yolo11n_model()

        yolo11_detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        yolo11_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image, detection_model=yolo11_detection_model, shift_amount=[0, 0], full_shape=None, postprocess=None
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertGreater(len(object_prediction_list), 0)
        num_person = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
        self.assertEqual(num_person, 0)
        num_truck = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "truck":
                num_truck += 1
        self.assertEqual(num_truck, 0)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertGreater(num_car, 0)

    def test_get_sliced_prediction_yolo11(self):
        from sahi.models.ultralytics import UltralyticsDetectionModel
        # init model
        download_yolo11n_model()

        yolo11_detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        yolo11_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"

        slice_height = 512
        slice_width = 512
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2
        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=yolo11_detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=False,
            postprocess_type=postprocess_type,
            postprocess_match_threshold=match_threshold,
            postprocess_match_metric=match_metric,
            postprocess_class_agnostic=class_agnostic,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertGreater(len(object_prediction_list), 0)
        num_person = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
        self.assertEqual(num_person, 0)
        num_truck = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "truck":
                num_truck += 1
        self.assertEqual(num_truck, 0)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertGreater(num_car, 0)

    @pytest.mark.skipif(sys.version_info[:2] > (3, 10), reason="Requires Python 3.10 or lower")
    def test_mmdet_yolox_tiny_prediction(self):
        from sahi.predict import predict
        from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_yolox_tiny_model

        # init model
        download_mmdet_yolox_tiny_model()

        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # prepare paths
        dataset_json_path = "tests/data/coco_utils/terrain_all_coco.json"
        source = "tests/data/coco_utils/"
        project_dir = "tests/data/predict_result"

        # get sliced prediction
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
        predict(
            model_type="mmdet",
            model_path=MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_PATH,
            model_config_path=MmdetTestConstants.MMDET_YOLOX_TINY_CONFIG_PATH,
            model_confidence_threshold=CONFIDENCE_THRESHOLD,
            model_device=MODEL_DEVICE,
            model_category_mapping=None,
            model_category_remapping=None,
            source=source,
            no_sliced_prediction=False,
            no_standard_prediction=True,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type=postprocess_type,
            postprocess_match_metric=match_metric,
            postprocess_match_threshold=match_threshold,
            postprocess_class_agnostic=class_agnostic,
            novisual=True,
            export_pickle=False,
            export_crop=False,
            dataset_json_path=dataset_json_path,
            project=project_dir,
            name="exp",
            verbose=1,
        )

    def test_ultralytics_yolo11n_prediction(self):
        # init model
        download_yolo11n_model()

        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # prepare paths
        dataset_json_path = "tests/data/coco_utils/terrain_all_coco.json"
        source = "tests/data/coco_utils/"
        project_dir = "tests/data/predict_result"

        # get sliced prediction
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
        predict(
            model_type="ultralytics",
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            model_config_path=None,
            model_confidence_threshold=CONFIDENCE_THRESHOLD,
            model_device=MODEL_DEVICE,
            model_category_mapping=None,
            model_category_remapping=None,
            source=source,
            no_sliced_prediction=False,
            no_standard_prediction=True,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type=postprocess_type,
            postprocess_match_metric=match_metric,
            postprocess_match_threshold=match_threshold,
            postprocess_class_agnostic=class_agnostic,
            novisual=True,
            export_pickle=False,
            export_crop=False,
            dataset_json_path=dataset_json_path,
            project=project_dir,
            name="exp",
            verbose=1,
        )

    def test_video_prediction(self):
        from os import path

        from sahi.predict import predict
        from sahi.utils.file import download_from_url

        # download video file
        source_url = "https://github.com/obss/sahi/releases/download/0.9.2/test.mp4"
        destination_path = "tests/data/test.mp4"
        if not path.exists(destination_path):
            download_from_url(source_url, destination_path)

        # init model
        download_yolo11n_model()

        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        image_size = 320
        class_agnostic = True

        # prepare paths
        source = destination_path
        project_dir = "tests/data/predict_result"

        # get sliced inference from video input without exporting visual
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
        predict(
            model_type="ultralytics",
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            model_config_path=None,
            model_confidence_threshold=CONFIDENCE_THRESHOLD,
            model_device=MODEL_DEVICE,
            model_category_mapping=None,
            model_category_remapping=None,
            source=source,
            no_sliced_prediction=False,
            no_standard_prediction=True,
            slice_height=512,
            slice_width=512,
            image_size=image_size,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type=postprocess_type,
            postprocess_match_metric=match_metric,
            postprocess_match_threshold=match_threshold,
            postprocess_class_agnostic=class_agnostic,
            novisual=True,
            export_pickle=False,
            export_crop=False,
            dataset_json_path=None,
            project=project_dir,
            name="exp",
            verbose=1,
        )

        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        image_size = 320
        class_agnostic = True

        # get standard inference from video input without exporting visual
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
        predict(
            model_type="ultralytics",
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            model_config_path=None,
            model_confidence_threshold=CONFIDENCE_THRESHOLD,
            model_device=MODEL_DEVICE,
            model_category_mapping=None,
            model_category_remapping=None,
            source=source,
            no_sliced_prediction=True,
            no_standard_prediction=False,
            image_size=image_size,
            postprocess_type=postprocess_type,
            postprocess_match_metric=match_metric,
            postprocess_match_threshold=match_threshold,
            postprocess_class_agnostic=class_agnostic,
            novisual=True,
            export_pickle=False,
            export_crop=False,
            dataset_json_path=None,
            project=project_dir,
            name="exp",
            verbose=1,
        )

        # get standard inference from video input and export visual
        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        image_size = 320
        class_agnostic = True

        # get full sized prediction
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
        predict(
            model_type="ultralytics",
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            model_config_path=None,
            model_confidence_threshold=CONFIDENCE_THRESHOLD,
            model_device=MODEL_DEVICE,
            model_category_mapping=None,
            model_category_remapping=None,
            source=source,
            no_sliced_prediction=True,
            no_standard_prediction=False,
            image_size=image_size,
            postprocess_type=postprocess_type,
            postprocess_match_metric=match_metric,
            postprocess_match_threshold=match_threshold,
            postprocess_class_agnostic=class_agnostic,
            export_pickle=False,
            export_crop=False,
            dataset_json_path=None,
            project=project_dir,
            name="exp",
            verbose=1,
        )


# Helper function to create a mock ObjectPrediction
def create_mock_object_prediction(bbox, score=0.9, category_id=0, category_name="mock_category"):
    return ObjectPrediction(
        bbox=bbox,
        category_id=category_id,
        score=score,
        bool_mask=None,
        category_name=category_name,
        shift_amount=[0, 0],
        full_shape=None,
    )

# Helper function to configure the mock model's prediction behavior
def configure_mock_model(mock_model_instance, full_shape):
    """Configure mock model to return a fixed relative prediction."""
    mock_model_instance.object_prediction_list = []
    mock_model_instance.category_mapping = {"0": "mock_category"} # Example mapping

    def mock_convert_original_predictions(shift_amount: list = [0, 0], full_shape=None):
        # Simulate adding one prediction relative to the current slice/ROI
        relative_bbox = [5, 5, 15, 15]  # e.g., a 10x10 box at top-left corner (5,5)
        # Create the prediction without shift first
        pred = ObjectPrediction(
            bbox=relative_bbox,
            category_id=0,
            score=0.9,
            bool_mask=None,
            category_name="mock_category",
            shift_amount=[0, 0], # Shift is applied later by get_prediction or get_sliced_prediction
            full_shape=full_shape, # Pass the full_shape if needed
        )
        mock_model_instance.object_prediction_list = [pred] # Replace list for simplicity

    mock_model_instance.perform_inference.return_value = None
    mock_model_instance.convert_original_predictions = MagicMock(side_effect=mock_convert_original_predictions)
    mock_model_instance.model = MagicMock() # Mock the inner model object if accessed


class TestGetSlicedPredictionCustomRois(unittest.TestCase):
    def setUp(self):
        """Set up common resources for tests."""
        self.dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)
        self.mock_model = MagicMock()
        configure_mock_model(self.mock_model, full_shape=[200, 200])

    def test_predict_with_custom_rois(self):
        """Test prediction using only custom ROIs."""
        custom_rois = [[10, 10, 60, 60], [100, 100, 150, 150]] # Two ROIs

        prediction_result = get_sliced_prediction(
            image=self.dummy_image,
            detection_model=self.mock_model,
            custom_rois=custom_rois,
            perform_standard_pred=False, # Only use ROIs
            postprocess_type="NMS", # Use NMS to avoid issues with mock scores
            postprocess_match_threshold=0.1, # Low threshold for mock
            verbose=0,
        )

        self.assertEqual(len(prediction_result.object_prediction_list), 2, "Should have 2 predictions, one for each ROI.")

        # Check coordinates (relative [5,5,15,15] shifted by ROI top-left)
        expected_bbox1 = [10 + 5, 10 + 5, 10 + 15, 10 + 15] # [15, 15, 25, 25]
        expected_bbox2 = [100 + 5, 100 + 5, 100 + 15, 100 + 15] # [105, 105, 115, 115]

        # Convert bbox to list for comparison as it might be numpy array
        actual_bbox1 = prediction_result.object_prediction_list[0].bbox.to_xyxy()
        actual_bbox2 = prediction_result.object_prediction_list[1].bbox.to_xyxy()

        np.testing.assert_array_almost_equal(actual_bbox1, expected_bbox1, decimal=1)
        np.testing.assert_array_almost_equal(actual_bbox2, expected_bbox2, decimal=1)

    def test_predict_without_custom_rois_fallback(self):
        """Test fallback to standard slicing when custom_rois is None."""
        slice_height = 50
        slice_width = 50
        overlap_ratio = 0.0 # No overlap

        prediction_result = get_sliced_prediction(
            image=self.dummy_image,
            detection_model=self.mock_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            custom_rois=None, # Explicitly None to trigger fallback
            perform_standard_pred=False,
            postprocess_type="NMS",
            postprocess_match_threshold=0.1,
            verbose=0,
        )

        # 200x200 image / 50x50 slices = 4x4 = 16 slices
        expected_num_predictions = 16
        self.assertEqual(
            len(prediction_result.object_prediction_list),
            expected_num_predictions,
            f"Should fallback to slicing and produce {expected_num_predictions} predictions."
        )

        # Check coordinates for the first slice (top-left, shift [0,0])
        expected_bbox_first = [0 + 5, 0 + 5, 0 + 15, 0 + 15] # [5, 5, 15, 15]
        actual_bbox_first = prediction_result.object_prediction_list[0].bbox.to_xyxy()
        np.testing.assert_array_almost_equal(actual_bbox_first, expected_bbox_first, decimal=1)

        # Check coordinates for the last slice (bottom-right, shift [150,150])
        expected_bbox_last = [150 + 5, 150 + 5, 150 + 15, 150 + 15] # [155, 155, 165, 165]
        actual_bbox_last = prediction_result.object_prediction_list[-1].bbox.to_xyxy()
        np.testing.assert_array_almost_equal(actual_bbox_last, expected_bbox_last, decimal=1)


    def test_predict_with_custom_rois_and_standard_pred(self):
        """Test prediction with custom ROIs and standard prediction enabled."""
        custom_rois = [[10, 10, 60, 60], [100, 100, 150, 150]] # Two ROIs

        # Need to reconfigure mock for standard prediction call
        # The standard prediction call uses the full image, shift [0,0]
        standard_pred_mock = MagicMock()
        configure_mock_model(standard_pred_mock, full_shape=[200, 200])

        # We need to ensure the same mock instance is used for all calls within get_sliced_prediction
        # or mock get_prediction itself. For simplicity, let's assume the same mock is used.
        # We adjust the side effect to handle both ROI calls and the standard call.

        call_count = 0
        def side_effect_for_rois_and_standard(shift_amount: list = [0, 0], full_shape=None):
            nonlocal call_count
            call_count += 1
            relative_bbox = [5, 5, 15, 15]
            pred = ObjectPrediction(
                 bbox=relative_bbox, category_id=0, score=0.9 - call_count*0.01, # Slightly different scores
                 bool_mask=None, category_name="mock_category",
                 shift_amount=[0, 0], full_shape=full_shape
            )
            # Important: get_prediction modifies the model's list in place
            self.mock_model.object_prediction_list = [pred]

        self.mock_model.convert_original_predictions.side_effect = side_effect_for_rois_and_standard


        prediction_result = get_sliced_prediction(
            image=self.dummy_image,
            detection_model=self.mock_model,
            custom_rois=custom_rois,
            perform_standard_pred=True, # Enable standard prediction
            postprocess_type="NMS", # Use NMS
            postprocess_match_threshold=0.1, # Low threshold
            verbose=0,
        )

        # Expected: 2 predictions from ROIs + 1 from standard prediction
        # Postprocessing (NMS) might merge if boxes overlap significantly and scores are identical
        # With slightly different scores, they should survive NMS if not overlapping heavily.
        self.assertEqual(
            len(prediction_result.object_prediction_list), 3,
            "Should have 3 predictions: 2 from ROIs and 1 from standard prediction."
        )

        # Check coordinates (order might vary due to postprocessing)
        expected_bboxes = [
            [15, 15, 25, 25],    # From ROI [10, 10, 60, 60]
            [105, 105, 115, 115], # From ROI [100, 100, 150, 150]
            [5, 5, 15, 15]       # From standard prediction (shift [0,0])
        ]
        actual_bboxes = sorted([p.bbox.to_xyxy() for p in prediction_result.object_prediction_list], key=lambda b: b[0])

        for i in range(3):
            np.testing.assert_array_almost_equal(actual_bboxes[i], sorted(expected_bboxes, key=lambda b: b[0])[i], decimal=1)


if __name__ == "__main__":
    unittest.main()
