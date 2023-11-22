from app.init import create_app
# from waitress import serve

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    # host = "0.0.0.0"
    # port = 5000
    # print(f"Starting the application on http://{host}:{port}")
    # serve(app, host=host, port=port)
    # print("Application is running.")