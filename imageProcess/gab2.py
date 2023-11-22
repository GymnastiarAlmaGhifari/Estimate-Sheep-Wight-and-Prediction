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
    c = 0.064443
    d = 0.010059
    weight = c * breadth_mm + d * length_mm
    return weight

# Load YOLO model
model = YOLO('D:/Backup agim/document/All Project/Kuliah/semester 5/Python/imageProcess/runs/detect/train5/weights/best.pt')

# Path to the input image
# input_path = ('D:/TIF/Semester 5/Project Peternakan Kambing/detection/test/kambing (198).jpg')

def process_image(image_bytes, id):
    try:
        # Load the input image
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        input_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        # input_image = cv2.imread(input_path)

        # Perform object detection on the input image with YOLO
        results = model(input_image)[0]

        # Set a detection threshold
        threshold = 0.5
        # Create a mask for the detected object
        mask = np.zeros(input_image.shape[:2], dtype=np.uint8)

        # Iterate through the detected objects
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(input_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                label = results.names[int(class_id)].upper()
                cv2.putText(input_image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                # Create a mask for the detected object
                mask[int(y1):int(y2), int(x1):int(x2)] = 255

                # Estimate the weight based on the contour of the detected object
                contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contour:
                    current_contour = max(contour, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(current_contour)
                    length_mm = w
                    breadth_mm = h
                    estimated_weight = estimate_weight(length_mm, breadth_mm)
                    cv2.putText(input_image, "Sheep", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(input_image, f"Weight: {estimated_weight:.2f} kg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Apply the mask to remove the background from the input image
        input_image_no_bg = cv2.bitwise_and(input_image, input_image, mask=mask)

        # Remove background using rembg library
        input_image_no_bg_pil = Image.fromarray(cv2.cvtColor(input_image_no_bg, cv2.COLOR_BGR2RGB))
        input_image_no_bg_removed = remove(np.array(input_image_no_bg_pil))

        # Convert the processed image to grayscale
        gray = cv2.cvtColor(input_image_no_bg_removed, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Apply Bilateral Filter
        gray_filtered = cv2.bilateralFilter(gray, 15, 75, 75)

        # Apply Median Blur
        gray_median_blurred = cv2.medianBlur(gray, 5)

        # Apply additional image processing (e.g., edge detection and morphological operations)
        edges = cv2.Canny(gray_median_blurred, 0, 0)
        kernel = np.ones((10, 10), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((15, 15), np.uint8)
        cleaned = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        segmented_frame = np.uint8(cleaned)
        contours, _ = cv2.findContours(segmented_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kambing_data = get_kambing(id)
        if 'tanggal_lahir' in kambing_data:
                tanggal_lahir = kambing_data['tanggal_lahir']
                print(f"Tanggal Lahir: {tanggal_lahir}")

                # Gunakan tanggal_lahir sebagai bagian dari nama file
        file_path = f'uploads/images/{id}_{tanggal_lahir}.jpeg'
        cv2.imwrite(file_path, input_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return file_path  # Mengembalikan path file yang disimpan

    except Exception as e:
        print(f"Error: {str(e)}")