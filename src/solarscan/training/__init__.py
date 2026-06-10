"""Training loops for the SolarScan models."""

from solarscan.training.train_classifier import train_classifier
from solarscan.training.train_detector import train_detector

__all__ = ["train_classifier", "train_detector"]
