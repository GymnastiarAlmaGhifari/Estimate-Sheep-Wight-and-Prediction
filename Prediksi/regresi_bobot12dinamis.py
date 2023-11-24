import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# from app.db import use_engine
import requests

# dataset = pd.read_csv('datadummy_kambingcerdas2.csv')
# bobot_df = f"{umur_bulan}bulan"
# print(bobot_df)
# X = dataset[[bobot_df]]
# if umur_bulan == 0:
# y = dataset[['1bulan','2bulan', '3bulan', '4bulan', '5bulan', '6bulan', '7bulan', '8bulan', '9bulan', '10bulan', '11bulan','12bulan']]
# elif umur_bulan == 1:
# y = dataset[['2bulan', '3bulan', '4bulan', '5bulan', '6bulan', '7bulan', '8bulan', '9bulan', '10bulan', '11bulan','12bulan']]
# elif umur_bulan == 2:
# y = dataset[['3bulan', '4bulan', '5bulan', '6bulan', '7bulan', '8bulan', '9bulan', '10bulan', '11bulan','12bulan']]
# elif umur_bulan == 3:
# y = dataset[[ '4bulan', '5bulan', '6bulan', '7bulan', '8bulan', '9bulan', '10bulan', '11bulan','12bulan']]
# elif umur_bulan == 4:
# y = dataset[[ '5bulan', '6bulan', '7bulan', '8bulan', '9bulan', '10bulan', '11bulan','12bulan']]
# elif umur_bulan == 5:
# y = dataset[[ '6bulan', '7bulan', '8bulan', '9bulan', '10bulan', '11bulan','12bulan']]
# elif umur_bulan == 6:
# y = dataset[[ '7bulan', '8bulan', '9bulan', '10bulan', '11bulan','12bulan']]
# elif umur_bulan == 7:
# y = dataset[[ '8bulan', '9bulan', '10bulan', '11bulan','12bulan']]
# elif umur_bulan == 8:
# y = dataset[[ '9bulan', '10bulan', '11bulan','12bulan']]
# elif umur_bulan == 9:
# y = dataset[[ '10bulan', '11bulan','12bulan']]
# elif umur_bulan == 10:
# y = dataset[['11bulan','12bulan']]
# elif umur_bulan == 11:
# y = dataset[['12bulan']]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# data_kambing23 = pd.DataFrame({
# bobot_df: [estimated_weight_second]
# })
# predicted_weight = model.predict(data_kambing23)
# print("ID Kambing :", id_kambing)
# print("Umur_bulan : ", umur_bulan)
# print("estimated dari param : ", estimated_weight_second)
# print("rotated : ", rotated_image)
# for i, prediksi in enumerate(predicted_weight[0], start=umur_bulan+1):
#     print(f'Prediksi bobot kambing pada bulan ke-{i} : {prediksi}')
def prediksi_bobot(id_kambing, umur_bulan, estimated_weight_second, rotated_image):
        dataset = pd.read_csv('datadummy_kambingcerdas2.csv')
        bobot_df = f"{umur_bulan}bulan"
        print(bobot_df)
        X = dataset[[bobot_df]]
        if umur_bulan >= 0 and umur_bulan <= 12:
            y = dataset.iloc[:, umur_bulan+1:]
        else:
            print("Umur bulan tidak valid.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        data_kambing23 = pd.DataFrame({
            bobot_df: [estimated_weight_second]
        })
        predicted_weight = model.predict(data_kambing23)
        print("ID Kambing :", id_kambing)
        print("Umur_bulan : ", umur_bulan)
        print("estimated dari param : ", estimated_weight_second)
        print("rotated : ", rotated_image)
        result_string = ""
        for i, prediksi in enumerate(predicted_weight[0], start=umur_bulan):
            formatted_prediksi = f'{prediksi:.2f}'
            # result_string += f'bulan ke-{i} : {formatted_prediksi}|'
            result_string += f'{formatted_prediksi}|'
            print(f'Prediksi bobot kambing pada bulan ke-{i} : {formatted_prediksi}')
        # send_kambing (id_kambing, rotated_image, result_string)
        print(result_string)

# def send_kambing(id_kambing, rotated_image, result_string) :
#     try:
#         engine = use_engine()

#         # Mengecek apakah data dengan ID kambing tersebut ada
#         query_existence = f"SELECT 1 FROM kambing WHERE id_kambing = '{id_kambing}' LIMIT 1"
#         result_existence = engine.execute(query_existence)

#         if result_existence.fetchone():

#             # Endpoint API dari server Next.js dengan menyertakan ID sebagai parameter pada URL
#             nextjs_api_url = f'http://192.168.2.25:3000/api/socket/image?id={id_kambing}'

#             # Buka file gambar
#             with open(rotated_image, 'rb') as file_gambar:
#                 # Buat kamus untuk data formulir
#                 files = {'id': (None, id_kambing), 'image': (rotated_image, file_gambar), 'result': (None, result_string)}

#                 # Kirim ID, tanggal_lahir, dan gambar ke API Next.js menggunakan multipart/form-data
#                 response = requests.post(nextjs_api_url, files=files)

#             if response.status_code == 200:
#                 print('Data berhasil dikirim ke API Next.js')
#             else:
#                 print('Gagal mengirim data:', response.status_code, response.text)

#             return {"id": id_kambing, "tanggal_lahir": str(tanggal_lahir)}

#         else:
#             return {"error": f"Tidak ada data ditemukan untuk ID {id_kambing}"}, 404

#     except Exception as e:
#         return {"error": f"Error mengambil data: {str(e)}"}, 500
prediksi_bobot(123, 0, 4.2, "mbek.jpg")