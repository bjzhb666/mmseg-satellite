# visualize_gray_images
from PIL import Image
import numpy as np

# Load the grayscale image
image = Image.open('nusc1024/singapore-queenstown_1_1_sat_2048_1024_3072_2048.png').convert('L')

# Get the image data as a numpy array
data = np.array(image)

# Find the maximum value in the data (excluding 100)
max_value = np.max(data[data != 100])

# Normalize the data (excluding 100) and multiply by 255 to get the grayscale values
normalized_data = np.where(data == 100, 255, (data / max_value) * 255).astype(np.uint8)

# Create a new image from the normalized data
new_image = Image.fromarray(normalized_data)

# Save the new image
new_image.save('visualized_image.png')