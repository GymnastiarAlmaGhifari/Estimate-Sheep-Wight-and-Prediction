import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# from app.db import use_engine
import requests
def prediksi_bobot(id_kambing, umur_bulan, estimated_weight_second, rotated_image):
    try:
        umur_bulan = int(umur_bulan)
    except ValueError:
        print("Error: Umur bulan harus berupa angka.")
        return

    if umur_bulan < 0 or umur_bulan > 11:
        print("Error: Umur bulan tidak valid. Harus berada dalam rentang 0 hingga 11.")
        return

    dataset = pd.read_csv('datadummy_kambingcerdas2.csv')
    bobot_df = f"{umur_bulan}bulan"

    if bobot_df not in dataset.columns:
        print(f"Error: Kolom {bobot_df} tidak ditemukan dalam dataset.")
        return

    X = dataset[[bobot_df]]

    if umur_bulan >= 0 and umur_bulan <= 6:
        y = dataset.iloc[:, umur_bulan+2:umur_bulan+8]
    elif umur_bulan >= 7 and umur_bulan <= 11:
        y = dataset.iloc[:, umur_bulan+2:]
    else:
        print("Error: Umur bulan tidak valid.")
        return

    if X.empty or y.empty:
        print("Error: Dataset tidak mencukupi untuk pelatihan model.")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    data_kambing23 = pd.DataFrame({
        bobot_df: [estimated_weight_second]
    })

    try:
        predicted_weight = model.predict(data_kambing23)
    except Exception as e:
        print(f"Error during final prediction: {e}")
        return

    print("ID Kambing:", id_kambing)
    print("Umur_bulan:", umur_bulan)
    print("Estimated dari param:", estimated_weight_second)
    print("Rotated:", rotated_image)

    result_string = ""
    for i, prediksi in enumerate(predicted_weight[0], start=umur_bulan+1):
        formatted_prediksi = f'{prediksi:.2f}'
        result_string += f'{formatted_prediksi}|'
        print(f'Prediksi bobot kambing pada bulan ke-{i}: {formatted_prediksi}')
    # send_kambing(id_kambing, rotated_image, result_string, estimated_weight_second, umur_bulan)
    print(result_string)
    
def send_kambing(id_kambing, rotated_image, deskripsi, bobot, usia) :
    try:
        engine = use_engine()
        query_existence = f"SELECT 1 FROM kambing WHERE id_kambing = '{id_kambing}' LIMIT 1"
        result_existence = engine.execute(query_existence)

        if result_existence.fetchone():
            nextjs_api_url = f'http://localhost:3000/api/socket/image?id={id_kambing}&bobot={bobot}&usia={usia}&deskripsi={deskripsi}'

            # Buka file gambar
            with open(rotated_image, 'rb') as file_gambar:
                # Buat kamus untuk data formulir
                files = {'id': (None, id_kambing), 'image': (rotated_image, file_gambar)}
                # Kirim ID, tanggal_lahir, dan gambar ke API Next.js menggunakan multipart/form-data
                response = requests.post(nextjs_api_url, files=files)

            if response.status_code == 200:
                print('Data berhasil dikirim ke API Next.js')
            else:
                print('Gagal mengirim data:', response.status_code, response.text)

            return {"id": id_kambing, "bobot": bobot, "usia": usia, "deskripsi": deskripsi}, 200
        else:
            return {"error": f"Tidak ada data ditemukan untuk ID {id_kambing}"}, 404
    except Exception as e:
        return {"error": f"Error mengambil data: {str(e)}"}, 500