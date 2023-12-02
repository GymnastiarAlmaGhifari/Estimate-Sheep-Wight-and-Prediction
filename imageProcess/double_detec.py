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
import requests
import io
# sheepDetection/get_umur.py
import pandas as pd
from app.db import use_engine
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def estimate_weight(length_mm, breadth_mm):
    c = 0.035143
    d = 0.005059
    weight = c * breadth_mm + d * length_mm

    return weight

# Load YOLO model for the first detection
model_first = YOLO('D:/Backup agim/document/All Project/Kuliah/semester 5/Estimate Sheep Wight and Prediction/runs/detect/train2/weights/best.pt')

# Load YOLO model for the second detection
model_second = YOLO('D:/Backup agim/document/All Project/Kuliah/semester 5/Estimate Sheep Wight and Prediction/runs/detect/train4/weights/best.pt')


def process_image(image_bytes, id):
    try:
        # Load the input image
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        input_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # rotated_image = cv2.rotate(input_image, cv2.ROTATE_90_CLOCKWISE)
        rotated_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

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
        mask_second = np.zeros(input_image_no_bg_first.shape[:2], dtype=np.uint8)

        # Iterate through the detected objects in the second detection
        for result_second in results_second.boxes.data.tolist():
            x1_second, y1_second, x2_second, y2_second, score_second, class_id_second = result_second

            if score_second > threshold_second:
                cv2.rectangle(input_image_no_bg_first, (int(x1_second), int(y1_second)), (int(x2_second), int(y2_second)), (0, 255, 0), 4)
                label_second = results_second.names[int(class_id_second)].upper()
                cv2.putText(input_image_no_bg_first, label_second, (int(x1_second), int(y1_second - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                # Create a mask for the detected object in the second detection
                mask_second[int(y1_second):int(y2_second), int(x1_second):int(x2_second)] = 255

        # Apply the mask to remove the background from the input image in the second detection
        input_image_no_bg_second = cv2.bitwise_and(input_image_no_bg_first, input_image_no_bg_first, mask=mask_second)

        # Remove background using rembg library
        input_image_no_bg_pil = Image.fromarray(cv2.cvtColor(input_image_no_bg_second, cv2.COLOR_BGR2RGB))
        input_image_no_bg_removed = remove(np.array(input_image_no_bg_pil))

        # Convert the processed image to grayscale
        gray_second = cv2.cvtColor(input_image_no_bg_removed, cv2.COLOR_BGR2GRAY)

        # Apply Median Blur
        gray_median_blurred_second = cv2.medianBlur(gray_second, 5)

        # Display Binary image
        ret, binary_image = cv2.threshold(gray_median_blurred_second, 127, 255, cv2.THRESH_BINARY)

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
        predicted_weight_cerdas = estimate_weight(length_mm, breadth_mm)
        
        # Apply additional image processing (e.g., edge detection anad morphological operations)
        # get_kambing(id, predicted_weight, rotated_image)
        try:
            engine = use_engine()
            query = f"SELECT tanggal_lahir FROM kambing WHERE id_kambing= '{id}'"
            df = pd.read_sql(query, engine)

            if not df.empty:
                tanggal_lahir = df['tanggal_lahir'].iloc[0]

                tanggal_lahir_format = datetime.strptime(str(tanggal_lahir), '%Y-%m-%d')

                # Hitung umur dalam format bulan
                tanggal_hari_ini = datetime.now()
                umur_bulan = (tanggal_hari_ini.year - tanggal_lahir_format.year) * 12 + (tanggal_hari_ini.month - tanggal_lahir_format.month)
                
                # nextjs_api_url = f'http://localhost:3000/api/socket/image?id={id}&bobot={predicted_weight}&usia={umur_bulan}&deskripsi=oawkoakowak'
                # with io.BytesIO() as output:
                #     rotated_image_pil = Image.fromarray(rotated_image) 
                #     rotated_image_pil.save(output, format="JPEG")
                #     rotated_image_bytes = output.getvalue()

                # # Send the rotated image bytes to the API
                # files = {'filename': ('rotated_image.jpg', rotated_image_bytes, 'image/jpeg')}
                # response = requests.post(nextjs_api_url, files=files)

                # if response.status_code == 200:
                #     print('Data berhasil dikirim ke API Next.js')
                # else:
                #     print('Gagal mengirim data:', response.status_code, response.text)

                #     return {"id": id, "bobot": predicted_weight, "usia": umur_bulan, "deskripsi": "akwokaokwaokoa"}, 200
                # prediksi_bobot(id, umur_bulan, estimated_weight_second, rotated_image)
                if umur_bulan >= 0 and umur_bulan <= 6:
                    counter = 6
                elif umur_bulan>=7 and umur_bulan<=11:
                    counter = 6
                    if umur_bulan+counter > 12:
                        counter = 12 - umur_bulan
                else:
                    print("Umur bulan tidak valid.")
                deskripsi=""
            # 'D:/Backup agim/document/All Project/Kuliah/semester 5/Estimate Sheep Wight and Prediction
                dataset = pd.read_csv('D:/Backup agim/document/All Project/Kuliah/semester 5/Estimate Sheep Wight and Prediction/Prediksi/datadummy_kambingcerdas2.csv')
                for k in range(1, counter+1):
                    bobot_df = f"{umur_bulan}bulan"
                    if bobot_df not in dataset.columns:
                        print(f"Error: Kolom {bobot_df} tidak ditemukan dalam dataset.")
                        return
                    X = dataset[[bobot_df]]
                    y = dataset[[f'{umur_bulan+k}bulan']]
                    if X.empty or y.empty:
                        print("Error: Dataset tidak mencukupi untuk pelatihan model.")
                        return
                    X_kuadrat = X.apply(lambda x: x ** 2)
                    sum_x_kuadrat = X_kuadrat.sum().sum()
                    sum_x = X.sum().sum()
                    sum_y = y.sum().sum()
                    sum_x_kali_y = 0
                    for i in range(len(X)):
                        for j in range(len(y.columns)):
                            sum_x_kali_y += X.iloc[i, 0] * y.iloc[i, j]
                    n = X.count()
                    a = ((sum_y*sum_x_kuadrat) - (sum_x*sum_x_kali_y))/((n*(sum_x_kuadrat))-(pow(sum_x,2)))
                    b = ((n*sum_x_kali_y) - (sum_x*sum_y))/((n*sum_x_kuadrat) - ((pow(sum_x,2))))
                    persamaan = a + (b * predicted_weight_cerdas)
                    deskripsi += f"Bulan ke-{k+umur_bulan}: {persamaan[0]}|"
                    print(deskripsi)
                    bobot_df_standart = f"{umur_bulan}bulan"
                    X_standart = dataset[[bobot_df_standart]]
                    print(X_standart)
                    X_mean = X_standart.mean()
                    print(X_mean[0])
                    standart_bobot = X_mean[0] 
                    if predicted_weight_cerdas < standart_bobot:
                        keterangan = "Peternak harus waspada karena bobot saat ini tidak sesuai standart"
                    elif predicted_weight_cerdas > standart_bobot:
                        keterangan = "Bobot domba sangat baik dengan usia saat ini"
                    elif predicted_weight_cerdas + 2 > standart_bobot:
                        keterangan = "Bobot domba sudah sesai dengan standart usahakan pemberian pakan dan pemeriksaan kesehatan tetap teratur"
                nextjs_api_url = f'http://localhost:3000/api/socket/image?id={id}&bobot={predicted_weight_cerdas}&usia={umur_bulan}&deskripsi={deskripsi}&standart={standart_bobot}&keterangan={keterangan}'
                with io.BytesIO() as output:
                    rotated_image_pil = Image.fromarray(rotated_image) 
                    rotated_image_pil.save(output, format="JPEG")
                    rotated_image_bytes = output.getvalue()
                files = {'filename': ('rotated_image.jpg', rotated_image_bytes, 'image/jpeg')}
                response = requests.post(nextjs_api_url, files=files)
                if response.status_code == 200:
                    print('Data berhasil dikirim ke API Next.js')
                else:
                    print('Gagal mengirim data:', response.status_code, response.text)
                    return {"id": id, "bobot": predicted_weight_cerdas, "usia": umur_bulan, "deskripsi": "akwokaokwaokoa"}, 200
                return {"tanggal_lahir": str(tanggal_lahir)}
            else:
                return {"error": f"No data found for ID {id}"}, 404
        except Exception as e:
            return {"error": f"Error retrieving data: {str(e)}"}, 500
    except Exception as e:
        print(f"Error: {str(e)}")