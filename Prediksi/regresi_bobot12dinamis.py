import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from app.db import use_engine
import requests
import io
from PIL import Image
def prediksi_bobot(id, umur_bulan, predicted_weight, rotated_image):
    
    if umur_bulan < 0 or umur_bulan > 11:
        print("Error: Umur bulan tidak valid. Harus berada dalam rentang 0 hingga 11.")
        return

    dataset = pd.read_csv('E:\coolyeah\Semester 5\Sistem Cerdas\Estimate-Sheep-Weight-and-Prediction\Estimate-Sheep-Wight-and-Prediction\Prediksi\datadummy_kambingcerdas2_22data.csv')
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



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    data_kambing23 = pd.DataFrame({
        bobot_df: [predicted_weight]
    })

    predicted_weight = model.predict(data_kambing23)
    print("ID Kambing:", id)
    print("Umur_bulan:", umur_bulan)
    print("Estimated dari param:", predicted_weight)
    print("Rotated:", rotated_image)

    deskripsi = ""
    for i, prediksi in enumerate(predicted_weight[0], start=umur_bulan+1):
        formatted_prediksi = f'{prediksi:.2f}'
        deskripsi += f'{formatted_prediksi}|'
        print(f'Prediksi bobot kambing pada bulan ke-{i}: {formatted_prediksi}')

    send_kambing(id, rotated_image, deskripsi, predicted_weight, umur_bulan)
    print(deskripsi)
    
def send_kambing(id, rotated_image, deskripsi, predicted_weight, umur_bulan) :
    try:
        engine = use_engine()
        query_existence = f"SELECT 1 FROM kambing WHERE id_kambing = '{id}' LIMIT 1"
        result_existence = engine.execute(query_existence)

        if result_existence.fetchone():
            nextjs_api_url = f'http://localhost:3000/api/socket/image?id={id}&bobot={predicted_weight}&usia={umur_bulan}&deskripsi={deskripsi}'
            with io.BytesIO() as output:
                rotated_image_pil = Image.fromarray(rotated_image) 
                rotated_image_pil.save(output, format="JPEG")
                rotated_image_bytes = output.getvalue()

            # Send the rotated image bytes to the API
            files = {'filename': ('rotated_image.jpg', rotated_image_bytes, 'image/jpeg')}
            response = requests.post(nextjs_api_url, files=files)

            if response.status_code == 200:
                print('Data berhasil dikirim ke API Next.js')
            else:
                print('Gagal mengirim data:', response.status_code, response.text)

                return {"id": id, "bobot": predicted_weight, "usia": umur_bulan, "deskripsi": deskripsi}, 200
        else:
            return {"error": f"Tidak ada data ditemukan untuk ID {id}"}, 404
    except Exception as e:
        return {"error": f"Error mengambil data: {str(e)}"}, 500