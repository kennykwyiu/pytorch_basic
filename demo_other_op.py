import torch
import numpy as np
import cv2

### read_image_show_and_convert_to_torch
print("read_image_show_and_convert_to_torch")

# Read an image from disk using OpenCV.
# Note: cv2.imread reads in **BGR** channel order by default (not RGB).
data = cv2.imread("test.png")

print("data (OpenCV image as a NumPy array):")
print(data)

# Debug: print shape + dtype so it’s easy to understand what was loaded.
# Typical color image: (H, W, 3), dtype=uint8.
print("\ndata.shape / data.dtype:")
print(data.shape, data.dtype)
