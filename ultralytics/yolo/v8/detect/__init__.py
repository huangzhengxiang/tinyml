# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor, predict
from .train import DetectionTrainer, train, QuantDetectionTrainer, QASDetectionTrainer
from .val import DetectionValidator, val

__all__ = 'DetectionPredictor', 'predict', 'DetectionTrainer', 'train', 'DetectionValidator', 'val', 'QuantDetectionTrainer', 'QASDetectionTrainer'
