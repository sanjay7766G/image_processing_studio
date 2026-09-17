import cv2
import numpy as np


def add_gaussian_noise(image, strength=15):

    noise = np.random.normal(
        0,
        strength,
        image.shape
    )

    noisy = image.astype(
        np.float32
    ) + noise

    noisy = np.clip(
        noisy,
        0,
        255
    )

    return noisy.astype(np.uint8)


def restore_image(image):

    return cv2.medianBlur(
        image,
        3
    )
