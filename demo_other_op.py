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

# Display the image in a window named "test".
cv2.imshow("test", data)

# Wait until a key is pressed (0 = wait forever), so the window doesn’t close immediately.
cv2.waitKey(0)

# (Optional best practice) Close the window after keypress.
cv2.destroyAllWindows()

# --- Convert NumPy -> Torch ---
# Convert the NumPy array to a Torch tensor.
# Important: the resulting tensor shares memory with the NumPy array (no copy),
# and will typically be dtype=torch.uint8 with shape (H, W, 3).
out = torch.from_numpy(data)

print("\nout (Torch tensor converted from the image):")
print(out)

# Debug: print tensor shape + dtype for easy reading.
print("\nout.shape / out.dtype:")
print(out.shape, out.dtype)

# If you need a float tensor for a model, you often convert + normalize:
# out_f = out.permute(2, 0, 1).float() / 255.0  # (C, H, W), float in [0,1]