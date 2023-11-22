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


def save_image(image_bytes, id):
    try:
          # Load the input image
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)

        input_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        rotated_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        kambing_data = get_kambing(id)

        if 'tanggal_lahir' in kambing_data:
                tanggal_lahir = kambing_data['tanggal_lahir']
                print(f"Tanggal Lahir: {tanggal_lahir}")

                # Gunakan tanggal_lahir sebagai bagian dari nama file
        file_path = f'uploads/images/{id}_{tanggal_lahir}.jpeg'
        cv2.imwrite(file_path, rotated_image)
        return file_path  # Mengembalikan path file yang disimpan

    except Exception as e:
        print(f"Error: {str(e)}")