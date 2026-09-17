import cv2


def transform_image(image, angle=0, scale=1.0, tx=0, ty=0):
    height, width = image.shape[:2]

    center = (width // 2, height // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        scale
    )

    matrix[0, 2] += tx
    matrix[1, 2] += ty

    result = cv2.warpAffine(
        image,
        matrix,
        (width, height)
    )

    return result
