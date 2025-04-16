from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from app.models import User
from . import db, bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from flask_mail import Message
from app import mail 
import uuid


auth_bp = Blueprint('auth', __name__)

# Endpoint untuk register pengguna baru
@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    # Validasi input
    if not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({"message": "Username, email, and password are required!"}), 400

    # Cek jika username atau email sudah ada
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"message": "Username already exists!"}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"message": "Email already exists!"}), 400

    # Buat pengguna baru (password langsung di-hash di dalam class User)
    new_user = User(
        id=str(uuid.uuid4()),  # Generate ID unik
        username=data['username'],
        email=data['email'],
        password=data['password']
    )
    
    # Simpan ke database
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User created successfully!"}), 201


# Endpoint untuk login pengguna
@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    # Validasi input
    if not data.get('username') or not data.get('password'):
        return jsonify({"message": "Username and password are required!"}), 400

    # Cari user berdasarkan username
    user = User.query.filter_by(username=data['username']).first()

    # Jika user tidak ditemukan atau password salah
    if user is None or not bcrypt.check_password_hash(user.password_hash, data['password']):
        return jsonify({"message": "Invalid username or password!"}), 401

    # Generate JWT token
    access_token = create_access_token(identity=user.id)

    return jsonify({"message": "Login successful", "access_token": access_token}), 200

# Endpoint untuk mendapatkan data pengguna yang sedang login
@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    # Mengambil ID pengguna yang sedang login (dari JWT token)
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)

    if not user:
        return jsonify({"message": "User not found!"}), 404

    return jsonify({
        "username": user.username,
        "email": user.email,
        "created_at": user.created_at
    }), 200


@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({"message": "Email is required!"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"message": "User with that email does not exist!"}), 404

    # Generate token (bisa disimpan ke DB kalau mau verifikasi nanti)
    reset_token = str(uuid.uuid4())

    # Kirim email
    msg = Message(
        subject="Password Reset Request",
        sender="emailmu@gmail.com",
        recipients=[email],
        body=f"Hi {user.username}, click the link to reset your password: http://localhost:5000/reset-password/{reset_token}"
    )
    mail.send(msg)

    return jsonify({"message": "Password reset email has been sent."}), 200
