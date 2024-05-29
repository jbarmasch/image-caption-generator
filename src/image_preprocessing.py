import os
import cv2
import numpy as np

def preprocess_image(image, target_size):
    # Resize image while preserving aspect ratio
    resized_image = resize_image(image, target_size)

    # Normalize pixel values to [0, 255]
    normalized_image = normalize_image(resized_image)

    return normalized_image

def resize_image(image, target_size):
    """Resize image while preserving aspect ratio"""
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate aspect ratios
    aspect_ratio = w / h
    target_aspect_ratio = target_w / target_h

    if aspect_ratio > target_aspect_ratio:
        # Resize based on width
        new_w = target_w
        new_h = int(new_w / aspect_ratio)
    else:
        # Resize based on height
        new_h = target_h
        new_w = int(new_h * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))

    # Pad the image to match target size
    top_pad = (target_h - new_h) // 2
    bottom_pad = target_h - new_h - top_pad
    left_pad = (target_w - new_w) // 2
    right_pad = target_w - new_w - left_pad

    resized_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad,
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])  # Pad with black

    return resized_image

def normalize_image(image):
    """Normalize pixel values to [0, 1]"""
    normalized_image = image.astype(np.float32) / 255.0
    return normalized_image

def save_image(image, output_path):
    """Save image to filesystem"""
    # Convert image to uint8 data type
    image_uint8 = (image * 255).astype(np.uint8)
    # Save image
    cv2.imwrite(output_path, image_uint8)

# # Directory paths
# input_directory = "./data/images/raw/"  # Directory containing raw images
# output_directory = "./data/images/processed/"  # Directory to save preprocessed images

# # Target size for resizing
# target_size = (224, 224)  

def process_directory(input_directory, output_directory, target_size):
    # Iterate over each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Consider only image files
            # Read image using OpenCV
            image_path = os.path.join(input_directory, filename)
            image = cv2.imread(image_path)

            # Preprocess image
            preprocessed_image = preprocess_image(image, target_size)

            # Output path for preprocessed image
            output_path = os.path.join(output_directory, filename)

            # Save preprocessed image
            save_image(preprocessed_image, output_path)

            # print("Preprocessed image saved to:", output_path)
