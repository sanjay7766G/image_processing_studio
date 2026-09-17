# CSE3010 – Image Processing Studio

## Project Overview
A beginner-friendly Computer Vision application based on **Module 1: Digital Image Formation and Low Level Processing**.

The application lets a user upload an image and apply common image-processing operations through a simple Streamlit interface.

## Module 1 Concepts Implemented
- Image representation and grayscale conversion
- Geometric transformation: rotation, scaling and translation
- Convolution/filtering: average, Gaussian, median and Laplacian
- Image enhancement: brightness/contrast, histogram equalization and CLAHE
- Histogram processing
- Fourier Transform and magnitude spectrum
- Image restoration using simulated noise and filtering
- Edge detection using Canny

These concepts correspond to the CSE3010 Module 1 syllabus, which includes image formation, transformations, Fourier Transform, convolution/filtering, image enhancement, restoration and histogram processing.

## Technologies
- Python
- OpenCV
- NumPy
- Pillow
- Matplotlib
- Streamlit

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Input
JPG, JPEG or PNG image.

## Output
The selected operation and processed image/histogram are displayed in the browser.

## Suggested Demo
1. Upload a photograph.
2. Show grayscale conversion.
3. Rotate/scale the image.
4. Apply Gaussian and median filters.
5. Show histogram and equalization.
6. Show Fourier magnitude spectrum.
7. Add noise and demonstrate restoration.
8. Apply Canny edge detection.

## Project Structure
- `app.py` – main application
- `requirements.txt` – dependencies
- `README.md` – setup and usage
- `statement.md` – project statement
- `docs/` – design and documentation notes
