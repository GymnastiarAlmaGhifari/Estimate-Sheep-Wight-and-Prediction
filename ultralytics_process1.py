import cv2
import numpy as np
from datetime import datetime
import os

# Memuat model YOLO

# Fungsi untuk memproses gambar
def process_image(image_bytes, id):
    try:
        # Mengonversi byte gambar ke format OpenCV
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        input_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # menyimpan image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = f'uploads/images/{id}_{timestamp}.jpeg'
        cv2.imwrite(file_path, input_image)

        return file_path  # Mengembalikan path file yang disimpan


    except Exception as e:
        print(f"Error: {str(e)}")
