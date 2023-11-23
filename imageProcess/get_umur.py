# sheepDetection/get_umur.py
import pandas as pd
from app.db import use_engine
from datetime import datetime
from Prediksi.prediksi_bobot import prediksi_bobot

def get_kambing(id_kambing, estimated_weight_second, rotated_image):
    try:
        engine = use_engine()
        query = f"SELECT tanggal_lahir FROM kambing WHERE id_kambing= '{id_kambing}'"
        df = pd.read_sql(query, engine)

        if not df.empty:
            tanggal_lahir = df['tanggal_lahir'].iloc[0]

            tanggal_lahir_format = datetime.strptime(tanggal_lahir, '%Y-%m-%d')

            # Hitung umur dalam format bulan
            tanggal_hari_ini = datetime.now()
            umur_bulan = (tanggal_hari_ini.year - tanggal_lahir_format.year) * 12 + (tanggal_hari_ini.month - tanggal_lahir_format.month)

            prediksi = prediksi_bobot(id_kambing,umur_bulan, estimated_weight_second, rotated_image)
            return {"tanggal_lahir": str(tanggal_lahir)}
        else:
            return {"error": f"No data found for ID {id_kambing}"}, 404
    except Exception as e:
        return {"error": f"Error retrieving data: {str(e)}"}, 500