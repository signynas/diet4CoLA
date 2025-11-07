import cv2
import numpy as np

def normalize(data: np.ndarray) -> np.ndarray:
    return (data - data.min()) / (data.max() - data.min())

def normalize_in(data: np.ndarray, min: float, max: float) -> np.ndarray:
    return (data - min) / (max - min)

def clip(data: np.ndarray, min: float, max: float) -> np.ndarray:
    return np.clip(data, min, max)

def invert(data: np.ndarray) -> np.ndarray:
    return 1 - data

def add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left + right

def sub(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left - right

def mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left * right

def power(data: np.ndarray, exp: float) -> np.ndarray:
    return data ** exp

def lerp(left: np.ndarray, right: np.ndarray, t: float = 0.5) -> np.ndarray:
    return add(t * left, (1 - t) * right)

def blur(data: np.ndarray, kernel: tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(data, kernel, sigma)

def canny(data: np.ndarray, t1: float, t2: float):
    img_8bit = (data * 255).astype(np.uint8)
    edges = cv2.Canny(img_8bit, t1, t2) / 255

    return edges

def gradient(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.gradient(data)

def rotate(data: np.ndarray, angle: int) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    
    if angle not in {90, 180, 270}:
        raise ValueError("Angle must be one of {90, 180, 270} degrees.")

    # Use np.rot90 for clean, efficient rotations
    k = angle // 90
    return np.rot90(data, k=k)

def sigmoid(data: np.ndarray, k: float = 1.0, mid: float = 0.5) -> np.ndarray:
    return 1 / (1 + np.exp(-k * (data - mid)))

def sigmoid_offset(data: np.ndarray, 
                    k: float = 1.0, 
                    mid: float = 0.5, 
                    offset: float = 0.1) -> np.ndarray:
    left = sigmoid(data, k, mid + offset)
    right = 1 - normalize(sigmoid(data, k, mid - offset))
    return mul(left, right)