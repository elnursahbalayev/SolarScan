"""Image transforms for the IR module classifier.

Inputs are tiny (24x40) grayscale crops replicated to 3 channels and normalised
with ImageNet stats to suit pretrained backbones. Train-time augmentation is
kept mild (flips, small rotation, brightness/contrast jitter) — aggressive
spatial crops destroy these small thermal signatures. Brightness/contrast jitter
is the most useful here: it mimics thermal gain/exposure variation.
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(input_size: tuple[int, int], train: bool):
    h, w = input_size
    if train:
        return transforms.Compose(
            [
                transforms.Resize((h, w)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
