# src/model/fourier.py
import cv2
import numpy as np

def fourier_transform(image):
    """Appliquer la transformée de Fourier et visualiser le spectre de magnitude. (ne sert a rien mais whatever)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return magnitude_spectrum