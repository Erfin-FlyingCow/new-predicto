import os
from flask import Flask, render_template, request, redirect, url_for
from app import create_app, db

# Membuat instance aplikasi
app = create_app()

# Menyesuaikan lokasi folder template dan statis
app.template_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'frontend/templates')
app.static_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'frontend/static')

# Membuat tabel dalam konteks aplikasi
with app.app_context():
    db.create_all()

# Menjalankan aplikasi
@app.route('/')
def home():
    return render_template('landing-page/landing-page.html')

# Halaman Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Proses login di sini
        return redirect(url_for('dashboard'))
    return render_template('auth/login.html')

# Halaman Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Proses registrasi di sini
        return redirect(url_for('login'))
    return render_template('auth/register.html')

# Halaman Lupa Password
@app.route('/lupa-password', methods=['GET', 'POST'])
def lupa_password():
    if request.method == 'POST':
        # Proses lupa password di sini
        return redirect(url_for('login'))
    return render_template('auth/lupa-password.html')

# Halaman Dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('main-feature/dashboard.html')

# Halaman Prediksi
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Proses prediksi ML di sini
        return render_template('main-feature/prediction.html', hasil='Contoh hasil prediksi')
    return render_template('main-feature/prediction.html')

# Halaman History Prediksi
@app.route('/history')
def history():
    return render_template('main-feature/history-prediction.html')

# Halaman Detail History
@app.route('/history/<id>')
def detail_history(id):
    return render_template('main-feature/detail_history-prediction.html', id=id)

# Halaman Chatbot
@app.route('/chatbot')
def chatbot():
    return render_template('main-feature/chatbot.html')

# Halaman Setting Akun
@app.route('/setting', methods=['GET', 'POST'])
def setting():
    if request.method == 'POST':
        # Simpan perubahan setting
        return redirect(url_for('dashboard'))
    return render_template('main-feature/setting.html')

# Menjalankan aplikasi
if __name__ == "__main__":
    app.run(debug=True, port=5000)
