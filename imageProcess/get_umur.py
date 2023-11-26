# sheepDetection/get_umur.py
import pandas as pd
from app.db import use_engine
from datetime import datetime
from Prediksi.regresi_bobot12dinamis import prediksi_bobot
import requests
import io
from PIL import Image


def get_kambing(id, predicted_weight, rotated_image):
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
            
            # nextjs_api_url = f'http://localhost:3000/api/socket/image?id={id}&bobot={predicted_weight}&usia={12}&deskripsi=oawkoakowak'
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
            prediksi_bobot(id, umur_bulan, predicted_weight, rotated_image)
            return {"tanggal_lahir": str(tanggal_lahir)}
        else:
            return {"error": f"No data found for ID {id}"}, 404
    except Exception as e:
        return {"error": f"Error retrieving data: {str(e)}"}, 500