import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv

load_dotenv()  # Ini akan membaca file .env

db = SQLAlchemy()
bcrypt = Bcrypt()
jwt = JWTManager()

def create_app():
    app = Flask(__name__)

    # Konfigurasi database & secret
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URI")  # Sesuaikan dengan nama variabel di .env
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = os.getenv("JWT_SECRET_KEY")  # Gunakan os.getenv untuk mengambil nilai

    # Inisialisasi extensions
    db.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)

    # Register blueprint (routes)
    from .auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix="/api/auth")

    return app
