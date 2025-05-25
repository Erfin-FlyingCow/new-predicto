from flask import Blueprint, request, jsonify, redirect, flash, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from app.models import User, PasswordResetToken
from datetime import datetime, timedelta
from . import db, bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from flask_mail import Message
from app import mail 
import re 
import uuid
from pytz import timezone, utc
from flask_jwt_extended import get_jwt, get_jwt_identity
from app.models import TokenBlacklist
import time


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
    access_token = create_access_token(identity=user.id, expires_delta=timedelta(hours=1))

    return jsonify({"message": "Login successful", "access_token": access_token}), 200

# Endpoint untuk mendapatkan data pengguna yang sedang login
@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)

    if not user:
        return jsonify({"message": "User not found!"}), 404

    # Ambil zona waktu dari header atau default ke UTC
    client_timezone_str = request.headers.get('X-Timezone', 'UTC')

    try:
        client_tz = timezone(client_timezone_str)
    except Exception:
        return jsonify({"message": "Invalid timezone format!"}), 400

    # Pastikan waktu created_at dalam UTC
    created_at_utc = user.created_at.replace(tzinfo=utc)

    # Konversi ke zona waktu klien
    created_at_local = created_at_utc.astimezone(client_tz)
    created_at_formatted = created_at_local.strftime('%d-%m-%Y / %H:%M:%S %Z')

    return jsonify({
        "username": user.username,
        "email": user.email,
        "created_at": created_at_formatted
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
    reset_link = f"{request.host_url}/reset-password-form/{reset_token}"
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
    # Deteksi tipe request
    if request.is_json:
        data = request.get_json()
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')
    else:
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

    # Validasi awal
    if not new_password or not confirm_password:
        message = "Semua kolom wajib diisi."
        if request.is_json:
            return jsonify({"message": message}), 400
        flash(message, "danger")
        return redirect(request.url)

    if new_password != confirm_password:
        message = "Password dan konfirmasi tidak cocok."
        if request.is_json:
            return jsonify({"message": message}), 400
        flash(message, "danger")
        return redirect(request.url)

    # Validasi password
    if len(new_password) < 8 or not re.search(r'[A-Z]', new_password) or not re.search(r'\d', new_password):
        message = "Password harus minimal 8 karakter, mengandung huruf besar dan angka."
        if request.is_json:
            return jsonify({"message": message}), 400
        flash(message, "danger")
        time.sleep(2)  # ⏱️ Delay 2 detik
        return redirect(request.url)

    # Validasi token
    token_entry = PasswordResetToken.query.filter_by(token=token, used=False).first()
    if not token_entry or token_entry.expires_at < datetime.utcnow():
        message = "Token tidak valid atau sudah kadaluarsa."
        if request.is_json:
            return jsonify({"message": message}), 400
        flash(message, "danger")
        time.sleep(2)  # ⏱️ Delay 2 detik
        return redirect(url_for('lupa_password'))

    # Update password
    user = User.query.get(token_entry.user_id)
    user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
    token_entry.used = True
    db.session.commit()

    message = "Password berhasil direset."
    if request.is_json:
        return jsonify({"message": message}), 200
    flash(message, "success")
    time.sleep(2)
    return redirect(url_for('login'))

    
@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    jti = get_jwt()['jti']  # Unique ID dari JWT token
    db.session.add(TokenBlacklist(jti=jti))
    db.session.commit()
    return jsonify({"message": "Logout successful"}), 200

