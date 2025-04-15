from datetime import datetime
from . import db
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()

class User(db.Model):
    __tablename__ = 'users'

    # Kolom-kolom pada tabel User
    id = db.Column(db.String(255), primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relasi dengan tabel History (one-to-many)
    histories = db.relationship('History', backref='user', lazy=True)

    def __init__(self, id, username, email, password):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    # Method untuk memverifikasi password
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    # Method untuk menghasilkan JWT token setelah login
    def generate_token(self):
        from flask_jwt_extended import create_access_token
        return create_access_token(identity=self.id)

    def __repr__(self):
        return f"<User {self.username}>"
    


from datetime import datetime
from . import db

class History(db.Model):
    __tablename__ = 'histories'

    # Kolom-kolom pada tabel History
    id = db.Column(db.String(255), primary_key=True)
    akun_id = db.Column(db.String(255), db.ForeignKey('users.id'), nullable=False)  # Foreign key ke tabel User
    result_prediction = db.Column(db.JSON, nullable=False)  # Hasil prediksi dalam format JSON
    prediction_at = db.Column(db.DateTime, default=datetime.utcnow)  # Tanggal dan waktu prediksi

    def __init__(self, id, akun_id, result_prediction):
        self.id = id
        self.akun_id = akun_id
        self.result_prediction = result_prediction

    def __repr__(self):
        return f"<History {self.id}>"

