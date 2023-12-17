from flask import Blueprint, request,jsonify
from datetime import datetime
from imageProcess.double_detec import process_image


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
    