import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSE3010 Image Processing Studio", layout="wide")
st.title("CSE3010 – Image Processing Studio")
st.write("A simple Computer Vision project covering Module 1 concepts: image formation, transformations, convolution/filtering, enhancement, Fourier transform, restoration, and histogram processing.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    st.sidebar.header("Processing")
    operation = st.sidebar.selectbox("Choose operation", [
        "Original / Grayscale",
        "Geometric Transformation",
        "Filtering",
        "Image Enhancement",
        "Histogram Processing",
        "Fourier Transform",
        "Restoration",
        "Edge Detection"
    ])

    if operation == "Original / Grayscale":
        c1, c2 = st.columns(2)
        c1.image(arr, caption="Original", use_container_width=True)
        c2.image(gray, caption="Grayscale", use_container_width=True, clamp=True)

    elif operation == "Geometric Transformation":
        angle = st.sidebar.slider("Rotation angle", -180, 180, 0)
        scale = st.sidebar.slider("Scale", 50, 150, 100) / 100
        tx = st.sidebar.slider("X translation", -200, 200, 0)
        ty = st.sidebar.slider("Y translation", -200, 200, 0)
        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        result = cv2.warpAffine(arr, M, (w, h))
        st.image(result, caption="Transformed Image", use_container_width=True)

    elif operation == "Filtering":
        kind = st.sidebar.selectbox("Filter", ["Average", "Gaussian", "Median", "Laplacian"])
        k = st.sidebar.select_slider("Kernel size", options=[3,5,7,9], value=3)
        if kind == "Average":
            result = cv2.blur(arr, (k, k))
        elif kind == "Gaussian":
            result = cv2.GaussianBlur(arr, (k, k), 0)
        elif kind == "Median":
            result = cv2.medianBlur(arr, k)
        else:
            result = cv2.Laplacian(gray, cv2.CV_64F)
            result = cv2.convertScaleAbs(result)
        c1, c2 = st.columns(2)
        c1.image(arr, caption="Original", use_container_width=True)
        c2.image(result, caption=f"{kind} Filter", use_container_width=True, clamp=True)

    elif operation == "Image Enhancement":
        method = st.sidebar.selectbox("Enhancement", ["Brightness/Contrast", "Histogram Equalization", "CLAHE"])
        if method == "Brightness/Contrast":
            alpha = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
            beta = st.sidebar.slider("Brightness", -100, 100, 0)
            result = cv2.convertScaleAbs(arr, alpha=alpha, beta=beta)
        elif method == "Histogram Equalization":
            result = cv2.equalizeHist(gray)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            result = clahe.apply(gray)
        c1, c2 = st.columns(2)
        c1.image(arr, caption="Original", use_container_width=True)
        c2.image(result, caption=method, use_container_width=True, clamp=True)

    elif operation == "Histogram Processing":
        fig, ax = plt.subplots()
        ax.hist(gray.ravel(), bins=256, range=(0,256))
        ax.set_title("Grayscale Histogram")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        eq = cv2.equalizeHist(gray)
        fig2, ax2 = plt.subplots()
        ax2.hist(eq.ravel(), bins=256, range=(0,256))
        ax2.set_title("Equalized Histogram")
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    elif operation == "Fourier Transform":
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        c1, c2 = st.columns(2)
        c1.image(gray, caption="Input", use_container_width=True, clamp=True)
        c2.image(magnitude, caption="Fourier Magnitude Spectrum", use_container_width=True, clamp=True)

    elif operation == "Restoration":
        noise = st.sidebar.selectbox("Noise", ["Gaussian", "Salt & Pepper"])
        if noise == "Gaussian":
            sigma = st.sidebar.slider("Noise strength", 1, 50, 15)
            n = np.random.normal(0, sigma, arr.shape)
            noisy = np.clip(arr.astype(np.float32) + n, 0, 255).astype(np.uint8)
        else:
            noisy = arr.copy()
            amount = st.sidebar.slider("Noise amount", 0.001, 0.05, 0.01, 0.001)
            count = int(amount * arr.shape[0] * arr.shape[1])
            ys = np.random.randint(0, arr.shape[0], count)
            xs = np.random.randint(0, arr.shape[1], count)
            noisy[ys, xs] = 255
            ys = np.random.randint(0, arr.shape[0], count)
            xs = np.random.randint(0, arr.shape[1], count)
            noisy[ys, xs] = 0
        restored = cv2.medianBlur(noisy, 3)
        c1, c2, c3 = st.columns(3)
        c1.image(arr, caption="Original", use_container_width=True)
        c2.image(noisy, caption="Noisy", use_container_width=True)
        c3.image(restored, caption="Restored", use_container_width=True)

    elif operation == "Edge Detection":
        low = st.sidebar.slider("Lower threshold", 0, 255, 50)
        high = st.sidebar.slider("Upper threshold", 0, 255, 150)
        edges = cv2.Canny(gray, low, high)
        c1, c2 = st.columns(2)
        c1.image(gray, caption="Grayscale", use_container_width=True, clamp=True)
        c2.image(edges, caption="Canny Edges", use_container_width=True, clamp=True)
else:
    st.info("Upload a JPG, JPEG, or PNG image to begin.")
