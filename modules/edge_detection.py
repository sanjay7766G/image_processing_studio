import cv2


def canny_edges(
    image,
    lower=50,
    upper=150
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(
        gray,
        lower,
        upper
    )

    return edges
