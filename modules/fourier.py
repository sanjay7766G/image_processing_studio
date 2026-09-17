import cv2
import numpy as np


def fourier_transform(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    fourier = np.fft.fft2(gray)

    shifted = np.fft.fftshift(
        fourier
    )

    magnitude = 20 * np.log(
        np.abs(shifted) + 1
    )

    return magnitude
