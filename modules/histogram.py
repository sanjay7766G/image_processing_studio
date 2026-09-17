import cv2
import matplotlib.pyplot as plt


def get_histogram(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    figure, axis = plt.subplots()

    axis.hist(
        gray.ravel(),
        bins=256,
        range=(0, 256)
    )

    axis.set_title("Image Histogram")
    axis.set_xlabel("Intensity")
    axis.set_ylabel("Frequency")

    return figure
