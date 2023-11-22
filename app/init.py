from flask import Flask

def create_app():
    app = Flask(__name__)

    # Load configuration from config.py
    app.config.from_object('app.config')

    # Register routes
    from app import routes
    app.register_blueprint(routes.bp)

    return app