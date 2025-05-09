from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from app.models import User, PasswordResetToken
from datetime import datetime, timedelta
from . import db, bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from flask_mail import Message
from app import mail 
import re 
import uuid
from flask_jwt_extended import get_jwt, get_jwt_identity
from app.models import TokenBlacklist


auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    # Validasi input dasar
    if not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({"message": "Username, email, and password are required!"}), 400

    password = data['password']

    # Validasi password: minimal 8 karakter, ada huruf besar dan angka
    if len(password) < 8 or not re.search(r'[A-Z]', password) or not re.search(r'\d', password):
        return jsonify({
            "message": "Password must be at least 8 characters long, contain at least one uppercase letter and one digit."
        }), 400

    # Cek jika username atau email sudah ada
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"message": "Username already exists!"}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"message": "Email already exists!"}), 400

    # Buat pengguna baru
    new_user = User(
        id=str(uuid.uuid4()),
        username=data['username'],
        email=data['email'],
        password=password  # Pastikan hashing dilakukan di dalam model User
    )

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

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"message": "User not found"}), 404

    # Generate token
    reset_token = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(hours=1)

    # Simpan ke tabel reset token
    token_entry = PasswordResetToken(
        id=str(uuid.uuid4()),
        user_id=user.id,
        token=reset_token,
        expires_at=expires_at
    )
    db.session.add(token_entry)
    db.session.commit()

    # Kirim email ke user
    reset_link = f"{request.host_url}reset-password/{reset_token}"
    msg = Message(
        subject="Reset Password",
        sender="youremail@gmail.com",
        recipients=[email],
         html=f"""
    <p>Click the button below to reset your password:</p>
    <a href="{reset_link}" 
       style="display:inline-block;padding:10px 20px;background-color:#007bff;color:#ffffff;text-decoration:none;border-radius:5px;">
       Reset Password
    </a>
    <p>If you didn’t request this, please ignore this email.</p>
    """
    )
    mail.send(msg)

    return jsonify({"message": "Password reset email sent. Check your email !"})



@auth_bp.route('/reset-password/<token>', methods=['POST'])
def reset_password(token):
    data = request.get_json()
    new_password = data.get('new_password')

    if not new_password:
        return jsonify({"message": "New password is required"}), 400

    # Validasi password: minimal 8 karakter, huruf besar, dan angka
    if len(new_password) < 8 or not re.search(r'[A-Z]', new_password) or not re.search(r'\d', new_password):
        return jsonify({
            "message": "Password must be at least 8 characters long, contain at least one uppercase letter and one digit."
        }), 400

    # Cek token di database
    token_entry = PasswordResetToken.query.filter_by(token=token, used=False).first()

    if not token_entry:
        return jsonify({"message": "Invalid or expired token"}), 400

    # Cek apakah token sudah kadaluarsa
    if token_entry.expires_at < datetime.utcnow():
        return jsonify({"message": "Token has expired"}), 400

    # Update password user
    user = User.query.get(token_entry.user_id)
    user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')

    # Tandai token sudah dipakai
    token_entry.used = True

    db.session.commit()

    return jsonify({"message": "Password has been successfully reset"}), 200



@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    jti = get_jwt()['jti']  # Unique ID dari JWT token
    db.session.add(TokenBlacklist(jti=jti))
    db.session.commit()
    return jsonify({"message": "Logout successful"}), 200

