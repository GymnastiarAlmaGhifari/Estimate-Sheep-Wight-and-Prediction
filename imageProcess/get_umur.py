# sheepDetection/get_umur.py
import pandas as pd
from app.db import use_engine

def get_kambing(id_kambing):
    try:
        engine = use_engine()
        query = f"SELECT tanggal_lahir FROM kambing WHERE id_kambing= '{id_kambing}'"
        df = pd.read_sql(query, engine)

        if not df.empty:
            tanggal_lahir = df['tanggal_lahir'].iloc[0]
            return {"tanggal_lahir": str(tanggal_lahir)}
        else:
            return {"error": f"No data found for ID {id_kambing}"}, 404
    except Exception as e:
        return {"error": f"Error retrieving data: {str(e)}"}, 500