import os
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session
from app import create_app, db

# Membuat instance aplikasi
app = create_app()

# Tambahkan secret_key untuk session
app.secret_key = os.environ.get('JWT_SECRET_KEY')

# Menyesuaikan lokasi folder template dan statis
app.template_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'frontend/templates')
app.static_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'frontend/static')

# Base URL untuk API
base_url = "http://127.0.0.1:5000"

# Membuat tabel dalam konteks aplikasi
with app.app_context():
    db.create_all()

# ROUTES
@app.route('/')
def home():
    return render_template('landing-page/landing-page.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        response = requests.post(base_url + '/api/auth/login', json={
            'username': username,
            'password': password
        })

        if response.status_code == 200:
            access_token = response.json().get('access_token')
            session['access_token'] = access_token
            return redirect(url_for('dashboard'))
        else:
            flash(response.json().get('message', 'Login gagal'), 'error')
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        response = requests.post(base_url + '/api/auth/register', json={
            'username': username,
            'email': email,
            'password': password
        })

        if response.status_code == 201:
            flash("Registrasi berhasil, silakan login.", "success")
            return redirect(url_for('login'))
        else:
            flash(response.json().get('message', 'Registrasi gagal'), 'error')
    return render_template('auth/register.html')

@app.route('/lupa-password', methods=['GET', 'POST'])
def lupa_password():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('auth/lupa-password.html')

@app.route('/dashboard')
def dashboard():
    return render_template('main-feature/dashboard.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # TODO: Tambahkan logika prediksi
        return render_template('main-feature/prediction.html', hasil='Contoh hasil prediksi')
    return render_template('main-feature/prediction.html')

@app.route('/history')
def history():
    return render_template('main-feature/history-prediction.html')

@app.route('/history/<id>')
def detail_history(id):
    return render_template('main-feature/detail_history-prediction.html', id=id)

@app.route('/chatbot')
def chatbot():
    return render_template('main-feature/chatbot.html')

@app.route('/setting', methods=['GET', 'POST'])
def setting():
    if request.method == 'POST':
        return redirect(url_for('dashboard'))
    return render_template('main-feature/setting.html')

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
