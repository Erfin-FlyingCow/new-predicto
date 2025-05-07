import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
from flask import Flask
from flask_mail import Mail

load_dotenv()  # Ini akan membaca file .env

db = SQLAlchemy()
bcrypt = Bcrypt()
jwt = JWTManager()
mail = Mail()

def create_app():
    app = Flask(__name__)

    # Konfigurasi mail
    app.config['MAIL_SERVER'] = 'smtp.gmail.com'
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USERNAME'] = os.getenv("MAIL")
    app.config['MAIL_PASSWORD'] = os.getenv("PASS_MAIL")

    mail.init_app(app)

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
    from .ml import ml_bp
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(ml_bp, url_prefix="/api/ml")

    return app
