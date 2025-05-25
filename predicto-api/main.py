import os
import requests
from flask_cors import CORS
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from app import create_app, db
from flask_jwt_extended import JWTManager
from app.models import TokenBlacklist
from flask_mail import Message
import uuid, json, datetime
from datetime import datetime, timedelta
from app.models import User, PasswordResetToken
from app import mail 


# Membuat instance aplikasi
app = create_app()
CORS(app, supports_credentials=True)

# Tambahkan secret_key untuk session
app.secret_key = os.environ.get('JWT_SECRET_KEY')

# Menyesuaikan lokasi folder template dan statis
app.template_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'frontend/templates')
app.static_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'frontend/static')

# Membuat tabel dalam konteks aplikasi
with app.app_context(): 
    db.create_all()


jwt = JWTManager(app)

@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    jti = jwt_payload["jti"]
    # Cek apakah token ini masuk blacklist
    return db.session.query(TokenBlacklist.id).filter_by(jti=jti).first() is not None

# Optional, agar pesan error saat akses dengan token blacklist jelas
@jwt.revoked_token_loader
def revoked_token_callback(jwt_header, jwt_payload):
    return jsonify({"message": "Token has been revoked"}), 401

# ROUTES
@app.route('/')
def home():
    return render_template('landing-page/landing-page.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if len(password) < 8:
            flash('Password must be at least 8 characters long, contain at least one uppercase letter and one digit.', 'error')
            return render_template('auth/login.html')
        try:
            response = requests.post(f'{request.host_url}/api/auth/login', json={
                'username': username,
                'password': password
            })
            if response.status_code == 200:
                access_token = response.json().get('access_token')
                session['access_token'] = access_token
                
                return redirect(url_for('dashboard'))
            else:
                error_message = response.json().get('message', 'Login gagal, periksa kembali username dan password')
                flash(error_message, 'error')
                return render_template('auth/login.html')
                
        except requests.exceptions.RequestException as e:
            flash('Terjadi kesalahan koneksi. Silakan coba lagi.', 'error')
            return render_template('auth/login.html')
        except Exception as e:
            flash('Terjadi kesalahan tidak terduga. Silakan coba lagi.', 'error')
            return render_template('auth/login.html')
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if len(password) < 8:
            flash("Password harus minimal 8 karakter", "error")
            return render_template('auth/register.html')
        
        if not any(c.isupper() for c in password) or not any(c.isdigit() for c in password):
            flash("Password harus mengandung minimal 1 huruf besar dan 1 angka", "error")
            return render_template('auth/register.html')
        
        try:
            response = requests.post(f'{request.host_url}api/auth/register', json={
                'username': username,
                'email': email,
                'password': password
            })

            if response.status_code == 201:
                flash("Akun berhasil dibuat! Silakan login.", "success")
                return redirect(url_for('login'))
            else:
                error_message = response.json().get('message', 'Registrasi gagal')
                
                if 'username' in error_message.lower() and 'already' in error_message.lower():
                    flash("Username sudah digunakan. Silakan pilih username lain.", "error")
                elif 'email' in error_message.lower() and 'already' in error_message.lower():
                    flash("Email sudah terdaftar. Silakan gunakan email lain atau login.", "error")
                else:
                    flash(error_message, "error")
                    
                return render_template('auth/register.html')
                
        except requests.exceptions.RequestException as e:
            flash("Terjadi kesalahan koneksi. Silakan coba lagi.", "error")
            return render_template('auth/register.html')
        except Exception as e:
            flash("Terjadi kesalahan tak terduga. Silakan coba lagi.", "error")
            return render_template('auth/register.html')
        
    return render_template('auth/register.html')

@app.route('/lupa-password', methods=['GET', 'POST'])
def lupa_password():
    if request.method == 'POST':
        email = request.form['email']
        
        response = requests.post(f'{request.host_url}/api/auth/forgot-password', json={
            'email': email
        })
        
        if response.status_code == 200:
            flash('Link reset password telah dikirim ke email Anda. Silakan periksa kotak masuk Anda.', 'success')
            return render_template('auth/lupa-password.html')
        else:
            flash(response.json().get('message', 'Email tidak ditemukan'), 'error')
            return render_template('auth/lupa-password.html')
    
    return render_template('auth/lupa-password.html')

@app.route('/dashboard')
def dashboard():
    if 'access_token' not in session:
        flash('Silakan login terlebih dahulu.', 'error')
        return redirect(url_for('login'))
    return render_template('main-feature/dashboard.html', token=session['access_token'])

@app.route('/prediction')
def prediction():
    if 'access_token' not in session:
        flash('Silakan login terlebih dahulu.', 'error')
        return redirect(url_for('login'))
    return render_template('main-feature/prediction.html', token=session['access_token'])

@app.route('/api/ml/predict')
def predict():
    if 'access_token' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    period = request.args.get('period', 'daily')
    
    frequency_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'ME'
    }
    
    frequency = frequency_map.get(period)  
    
    headers = {
        'Authorization': f'Bearer {session["access_token"]}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "frequency": frequency
    }
    
    try:
        
        response = requests.post(
            f'{request.host_url}/api/ml/predict',
            headers=headers,
            json=payload  
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            
            error_message = f"API Error: {response.status_code}"
            try:
                error_data = response.json()
                error_message = f"{error_message}, {json.dumps(error_data)}"
            except:
                error_message = f"{error_message}, {response.text}"
                
            print(error_message)
            return jsonify({'error': 'Failed to get prediction', 'status': response.status_code, 'details': response.text}), response.status_code
            
    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/ml/predict/save', methods=['POST'])
def save_prediction():
    if 'access_token' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        
        prediction_data = request.json
        token=session['access_token']
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        
        print(f"Sending prediction data to API: {json.dumps(prediction_data)}")
        
        
        response = requests.post(
            f'{request.host_url}/api/ml/predict/save',
            headers=headers,
            json=prediction_data,
            timeout=30
        )
        
        
        print(f"Response from API: Status {response.status_code}")
        try:
            response_data = response.json()
            print(f"Response data: {json.dumps(response_data)}")
        except:
            print(f"Raw response: {response.text}")
        
        if response.status_code == 201 or response.status_code == 200:
            return jsonify({'message': 'Prediction history saved successfully'}), 200
        else:
            return jsonify({
                'error': 'Error saving prediction', 
                'status': response.status_code, 
                'details': response.json() if response.content else 'No details available'
            }), response.status_code
            
    except Exception as e:
        print(f"Exception in save_prediction: {str(e)}")
        return jsonify({
            'error': f'Error saving prediction: {str(e)}'
        }), 500
    
@app.route('/api/ml/predict/histories', methods=['GET'])
def get_prediction_histories():
    if 'access_token' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        
        headers = {
            'X-Timezone': 'Asia/Jakarta',
            'Authorization': f'Bearer {session["access_token"]}',
            'Content-Type': 'application/json'
        }
        
        
        print(f"Making request to {request.host_url}/api/ml/predict/histories with token {session['access_token'][:10]}...")
        
        
        response = requests.get(
            f'{request.host_url}/api/ml/predict/histories',
            headers=headers,
            timeout=30
        )
        
        
        print(f"Received response with status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        if response.content:
            try:
                
                content_text = response.text
                print(f"Raw response content: {content_text[:1000]}...")  
                
                
                response_data = response.json()
                print(f"Parsed as JSON: {json.dumps(response_data)[:500]}...")  
                
                
                if isinstance(response_data, dict) and 'histories' in response_data:
                    print(f"Found 'histories' key with {len(response_data['histories'])} items")
                    if response_data['histories']:
                        first_item = response_data['histories'][0]
                        print(f"First item keys: {list(first_item.keys())}")
                elif isinstance(response_data, list):
                    print(f"Response is a list with {len(response_data)} items")
                    if response_data:
                        first_item = response_data[0]
                        print(f"First item keys: {list(first_item.keys())}")
                else:
                    print(f"Unknown response structure: {type(response_data)}")
                
            except Exception as e:
                print(f"Could not parse response as JSON. Error: {str(e)}")
                print(f"Raw content: {response.text[:500]}...")
        else:
            print("Response has no content")
        
        if response.status_code == 200:
            
            return response.text, 200, {'Content-Type': 'application/json'} 
        else:
            error_msg = f"Error fetching prediction histories: status {response.status_code}"
            print(error_msg)
            return jsonify({
                'error': error_msg, 
                'status': response.status_code, 
                'details': response.json() if response.content else 'No details available'
            }), response.status_code
            
    except Exception as e:
        error_msg = f"Exception in get_prediction_histories: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': error_msg
        }), 500


@app.route('/history')
def history():
    if 'access_token' not in session:
        flash('Silakan login terlebih dahulu.', 'error')
        return redirect(url_for('login'))
    return render_template('main-feature/history-prediction.html', token=session['access_token'])

@app.route('/history/detail')
def history_detail():
    if 'access_token' not in session:
        flash('Silakan login terlebih dahulu.', 'error')
        return redirect(url_for('login'))
    return render_template('main-feature/detail-history-prediction.html',token=session['access_token'])



@app.route('/api/ml/predict/histories/detail', methods=['POST'])
def get_prediction_detail():
    if 'access_token' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        
        data = request.json
        prediction_id = data.get('id')
        
        if not prediction_id:
            return jsonify({'error': 'No prediction ID provided'}), 400
        
        
        headers = {
            'X-Timezone': 'Asia/Jakarta',
            'Authorization': f'Bearer {session["access_token"]}',
            'Content-Type': 'application/json'
        }
        
        
        print(f"Fetching prediction detail for ID: {prediction_id}")
        
        
        response = requests.post(
            f'{request.host_url}/api/ml/predict/histories/detail',
            headers=headers,
            json={'id': prediction_id},
            timeout=30
        )
        
        
        print(f"API response status: {response.status_code}")
        print(f"API response headers: {response.headers}")
        
        if response.status_code == 200:
            try:
                
                response_data = response.json()
                
                
                print(f"API response data: {response_data}")
                
                
                
                return jsonify(response_data)
            except Exception as e:
                print(f"Error parsing response JSON: {str(e)}")
                
                return jsonify({
                    'prediction_date': datetime.now().strftime("%Y-%m-%d"),
                    'error': 'Invalid JSON from API',
                    'raw_response': response.text
                })
        else:
            
            print(f"Error response from API: {response.text}")
            return jsonify({
                'error': 'Failed to get prediction detail', 
                'status': response.status_code, 
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        print(f"Exception in get_prediction_detail: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Error fetching prediction detail: {str(e)}'
        }), 500


@app.route('/chatbot')
def chatbot():
    if 'access_token' not in session:
        flash('Silakan login terlebih dahulu.', 'error')
        return redirect(url_for('login'))
    return render_template('main-feature/chatbot.html')

@app.route('/api/chatbot/', methods=['POST'])
def api_chatbot():
    if 'access_token' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        
        message = request.json.get('message')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        
        headers = {
            'Authorization': f'Bearer {session["access_token"]}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{request.host_url}/api/chatbot/',
            headers=headers,
            json={'message': message},
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                'error': 'Error from API', 
                'status': response.status_code, 
                'response': 'Maaf, layanan chatbot sedang tidak tersedia.'
            }), response.status_code
    except Exception as e:
        return jsonify({
            'error': str(e),
            'response': 'Terjadi kesalahan saat menghubungi server.'
        }), 500


@app.route('/setting', methods=['GET', 'POST'])
def setting():
    if 'access_token' not in session:
        flash('Silakan login terlebih dahulu.', 'error')
        return redirect(url_for('login'))
    
    
    try:
        headers = {
            'X-Timezone': 'Asia/Jakarta',
            'Authorization': f'Bearer {session["access_token"]}'
        }
        response = requests.get(f'{request.host_url}/api/auth/me', headers=headers)
        
        if response.status_code == 200:
            user_data = response.json()
            
            if 'created_at' in user_data:
                
                from datetime import datetime
                try:
                    
                    created_at = datetime.fromisoformat(user_data['created_at'].replace('Z', '+00:00'))
                    user_data['formatted_date'] = created_at.strftime('%d-%m-%Y')
                except:
                   
                    user_data['formatted_date'] = user_data['created_at']
        else:
            flash('Failed to fetch user data', 'error')
            user_data = {
                'username': '',
                'email': '',
                'formatted_date': ''
            }
    except Exception as e:
        flash(f'Error fetching user data: {str(e)}', 'error')
        user_data = {
            'username': '',
            'email': '',
            'formatted_date': ''
        }
        
    if request.method == 'POST':
        return redirect(url_for('dashboard'))
        
    return render_template('main-feature/setting.html', user_data=user_data)

@app.route("/reset-password-form/<token>", methods=["GET"])
def reset_password_form(token):
    return render_template("auth/ubah-password.html", token=token)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
