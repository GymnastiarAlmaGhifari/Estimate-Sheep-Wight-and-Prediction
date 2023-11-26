from flask import Blueprint, render_template, request,jsonify, send_from_directory
from datetime import datetime
from imageProcess.double_detec import process_image
# from imageProcess.gambar import save_image
from imageProcess.get_umur import get_kambing

# from exponen.co2 import get_forecast_co2

bp = Blueprint('main', __name__)

# global variable to store the id of the current user
id_kambing_global = None

@bp.route('/api/python/image', methods=['POST'])
def receive_image():
    global id_kambing_global
    try:
        print("Request received")
        id = request.args.get('id')
        print(f"Received ID: {id}")
        if 'imageFile' not in request.files:
            print("No image file provided")
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['imageFile']
        
        if image_file.filename == '':
            print("No selected file")
            return jsonify({"error": "No selected file"}), 400

        # Log the size of the received image
        print(f"Received image file size: {len(image_file.read())} bytes")

        # Reset file position to the beginning before processing
        image_file.seek(0)

        result = process_image(image_file.read(), id)

        # Save the ID to the global variable
        id_kambing_global = id

        return jsonify({"result": result})
    except Exception as e:
        print(f'Error: {str(e)}')
        return jsonify({"error": f"500 Internal Server Error - {str(e)}"}), 500
    
@bp.route('/api/python/kambing', methods=['GET'])
def get_kambing():
    global id_kambing_global

    # gunakan variabel global id_kambing_global untuk mengambil id dari user
    id_kambing =request.args.get('id_kambing_global')

    if id_kambing is None:
        return jsonify({"error": "Missing 'id_kambing_global' parameter"}), 400
    
    # gunakan id_kambing untuk mengambil data kambing dari database
    kambing = get_kambing(id_kambing_global)
    return jsonify({"kambing": kambing})


    
# @bp.route('/index')
# def index():

#     result_path = 'uploads/images/your_id_your_timestamp/mask_result.jpeg'

#     data = {
#         'result_path': result_path
#     }
#     return render_template('index.html', data=data)





# @bp.route('/forecast_co2', methods=['POST'])
# def forecast_co2():
#     # Get the 'esp_id' parameter from the query string
#     esp_id = request.args.get('esp_id')

#     if esp_id is None:
#         return jsonify({"error": "Missing 'esp_id' parameter"}), 400

#     # Use the 'esp_id' in your get_forecast_co2 function
#     # forecast_co2 = get_forecast_co2(esp_id)
    # return jsonify({"Triple Exponential Smoothing Forecast": forecast_co2})

# @bp.route('/api/socket/iotimage', methods=['POST'])
# def receive_image():
#     try:
#         id = request.args.get('id')
#         image_file = request.files['imageFile']

#         # Generate a unique filename based on id and current date
#         current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"uploads/images/{id}_{current_date}.jpeg"

#         # Save the image file
#         image_file.save(filename)

#         # Optionally, you can perform further processing with the image here

#         return 'Image received successfully'
#     except Exception as e:
#         print(f'Error: {str(e)}')
#         return 'Error: 500 Internal Server Error' + str(e)