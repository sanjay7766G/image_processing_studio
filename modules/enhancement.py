import cv2


def adjust_brightness_contrast(
    image,
    brightness=0,
    contrast=1.0
):
    return cv2.convertScaleAbs(
        image,
        alpha=contrast,
        beta=brightness
    )


def histogram_equalization(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    return cv2.equalizeHist(gray)


def clahe_enhancement(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)
