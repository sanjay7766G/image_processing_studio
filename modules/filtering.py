import cv2


def average_filter(image, kernel=3):
    return cv2.blur(image, (kernel, kernel))


def gaussian_filter(image, kernel=3):
    return cv2.GaussianBlur(
        image,
        (kernel, kernel),
        0
    )


def median_filter(image, kernel=3):
    return cv2.medianBlur(image, kernel)


def laplacian_filter(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    result = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return cv2.convertScaleAbs(result)
