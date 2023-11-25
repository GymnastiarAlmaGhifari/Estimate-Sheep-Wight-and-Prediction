import os
from ultralytics import YOLO
import cv2
import numpy as np
from rembg import remove
from PIL import Image
from datetime import datetime
import pandas as pd
from app.db import use_engine
from imageProcess.get_umur import get_kambing

def estimate_weight(length_mm, breadth_mm):
    # c = 0.064443
    # d = 0.010059
    c = 0.080232
    d = 0.030023
    weight = c * breadth_mm + d * length_mm
    return weight

# Load YOLO model for the first detection
model_first = YOLO('D:/Backup agim/document/All Project/Kuliah/semester 5/Python/imageProcess/runs/detect/train2/weights/best.pt')

# Load YOLO model for the second detection
model_second = YOLO('D:/Backup agim/document/All Project/Kuliah/semester 5/Python/imageProcess/runs/detect/train4/weights/best.pt')

# # Path to the input image
# input_path = ('D:/TIF/Semester 5/Project Peternakan Kambing/detection/test/kambing (127).jpg')

def process_image(image_bytes, id):
    try:
        # Load the input image
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        input_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        rotated_image = cv2.rotate(input_image, cv2.ROTATE_90_CLOCKWISE)

        # Perform object detection on the input image with the first YOLO model
        results_first = model_first(rotated_image)[0]

        # Set a detection threshold for the first detection
        threshold_first = 0.5
        # Create a mask for the detected object in the first detection
        mask_first = np.zeros(rotated_image.shape[:2], dtype=np.uint8)

        # Iterate through the detected objects in the first detection
        for result_first in results_first.boxes.data.tolist():
            x1_first, y1_first, x2_first, y2_first, score_first, class_id_first = result_first

            if score_first > threshold_first:
                cv2.rectangle(input_image, (int(x1_first), int(y1_first)), (int(x2_first), int(y2_first)), (0, 255, 0), 4)
                label_first = results_first.names[int(class_id_first)].upper()
                cv2.putText(input_image, label_first, (int(x1_first), int(y1_first - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
                # Create a mask for the detected object in the first detection
                mask_first[int(y1_first):int(y2_first), int(x1_first):int(x2_first)] = 255
        
        # Apply the mask to remove the background from the input image in the first detection
        input_image_no_bg_first = cv2.bitwise_and(rotated_image, rotated_image, mask=mask_first)

        results_second = model_second(input_image_no_bg_first)[0]
        
        # Set a detection threshold for the second detection
        threshold_second = 0.5

        # Create a mask for the detected object in the second detection
        mask_second = np.zeros(rotated_image.shape[:2], dtype=np.uint8)

        # Iterate through the detected objects in the second detection
        for result_second in results_second.boxes.data.tolist():
            x1_second, y1_second, x2_second, y2_second, score_second, class_id_second = result_second

            if score_second > threshold_second:
                cv2.rectangle(rotated_image, (int(x1_second), int(y1_second)), (int(x2_second), int(y2_second)), (0, 255, 0), 4)
                print(results_second.names[int(class_id_second)].upper())
                label_second = results_second.names[int(class_id_second)].upper()
                cv2.putText(rotated_image, label_second, (int(x1_second), int(y1_second - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                # Create a mask for the detected object in the second detection
                mask_second[int(y1_second):int(y2_second), int(x1_second):int(x2_second)] = 255

        # Apply the mask to remove the background from the input image in the second detection
        input_image_no_bg_second = cv2.bitwise_and(rotated_image, rotated_image, mask=mask_second)

        # Remove background using rembg library
        input_image_no_bg_pil = Image.fromarray(cv2.cvtColor(input_image_no_bg_second, cv2.COLOR_BGR2RGB))
        input_image_no_bg_removed = remove(np.array(input_image_no_bg_pil))

        # Convert the processed image to grayscale
        gray_second = cv2.cvtColor(input_image_no_bg_removed, cv2.COLOR_BGR2GRAY)

        # Apply Median Blur
        gray_median_blurred_second = cv2.medianBlur(gray_second, 5)

        # Display Binary image
        ret, binary_image = cv2.threshold(gray_median_blurred_second, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("putih nih boss", binary_image)

        # Estimate the weight based on the contour of the detected object in the second detection
        contours, _ = cv2.findContours(mask_second, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        # Initialize variables to store the width and height
        width, height = 0, 0

        # Iterate through the contours
        for contour in contours:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
    
        # Update width and height if the current bounding box is larger
        width = max(width, w)
        height = max(height, h)

        # Estimate length and breadth from the maximum width and height
        length_mm = height 
        breadth_mm = width

        # Predict the weight
        predicted_weight = estimate_weight(length_mm, breadth_mm)
        
        # Apply additional image processing (e.g., edge detection anad morphological operations)
        kambing_data = get_kambing(id, predicted_weight, rotated_image)
        if 'tanggal_lahir' in kambing_data:
                    tanggal_lahir = kambing_data['tanggal_lahir']
                    print(f"Tanggal Lahir: {tanggal_lahir}")

        # Gunakan tanggal_lahir sebagai bagian dari nama file
        file_path = f'uploads/images/{id}_{tanggal_lahir}.jpeg'
        cv2.imwrite(file_path, rotated_image)

    except Exception as e:
        print(f"Error: {str(e)}")