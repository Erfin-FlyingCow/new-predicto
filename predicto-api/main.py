import os
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from app import create_app, db
from flask_jwt_extended import JWTManager
from app.models import TokenBlacklist
from flask_mail import Message
import uuid
from datetime import datetime, timedelta
from app.models import User, PasswordResetToken
from app import mail 
import json

# Membuat instance aplikasi
app = create_app()

# Tambahkan secret_key untuk session
app.secret_key = os.environ.get('JWT_SECRET_KEY')

# Menyesuaikan lokasi folder template dan statis
app.template_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'frontend/templates')
app.static_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'frontend/static')

# Base URL untuk API
API_BASE_URL = "http://127.0.0.1:5000"

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

        response = requests.post(API_BASE_URL + '/api/auth/login', json={
            'username': username,
            'password': password
        })

        if response.status_code == 200:
            access_token = response.json().get('access_token')
            session['access_token'] = access_token
            return redirect(url_for('dashboard'))
        else:
           flash(response.json().get('message', 'Login gagal, periksa kembali username dan password'), 'error')
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        response = requests.post(API_BASE_URL + '/api/auth/register', json={
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
        email = request.form.get('email')  # ambil dari form, bukan JSON

        user = User.query.filter_by(email=email).first()
        if not user:
            flash("Email tidak ditemukan. Silakan coba lagi.", "danger")
            return redirect(url_for('lupa_password'))

        # Generate reset token
        reset_token = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=1)

        # Simpan token ke DB
        token_entry = PasswordResetToken(
            id=str(uuid.uuid4()),
            user_id=user.id,
            token=reset_token,
            expires_at=expires_at
        )
        db.session.add(token_entry)
        db.session.commit()

        # Buat tautan reset password
        reset_link = API_BASE_URL+f"/reset-password/{reset_token}"

        # Kirim email
        msg = Message(
            subject="Reset Password",
            sender="youremail@gmail.com",  # ganti dengan email valid
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
        mail .send(msg)

        flash("Tautan reset password telah dikirim ke email Anda.", "success")
        return redirect(url_for('login'))

    return render_template('auth/lupa-password.html')


@app.route('/dashboard')
def dashboard():
    return render_template('main-feature/dashboard.html')

@app.route('/prediction')
def prediction():
    if 'access_token' not in session:
        flash('Silakan login terlebih dahulu.', 'error')
        return redirect(url_for('login'))
    return render_template('main-feature/prediction.html')

@app.route('/api/ml/predict')
def predict():
    if 'access_token' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get period from request arguments
    period = request.args.get('period', 'daily')
    
    # Map period to the API's expected frequency format
    frequency_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'ME'
    }
    
    frequency = frequency_map.get(period)  # Default to daily if invalid period
    
    headers = {
        'Authorization': f'Bearer {session["access_token"]}',
        'Content-Type': 'application/json'
    }
    
    # Create the JSON payload with frequency
    payload = {
        "frequency": frequency
    }
    
    try:
        # Use POST with JSON body instead of GET with query parameters
        response = requests.post(
            f'{API_BASE_URL}/ml/predict',
            headers=headers,
            json=payload  # This will be sent as raw JSON in the request body
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            # Log the error response for debugging
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
        # Get prediction data from request
        prediction_data = request.json
        
        # Forward request to backend API
        headers = {
            'Authorization': f'Bearer {session["access_token"]}',
            'Content-Type': 'application/json'
        }
        
        # Log the request data for debugging
        print(f"Sending prediction data to API: {json.dumps(prediction_data)}")
        
        # Send the prediction data to backend for saving
        response = requests.post(
            f'{API_BASE_URL}/ml/predict/save',
            headers=headers,
            json=prediction_data,
            timeout=30
        )
        
        # Log the response for debugging
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
        # Set up the headers with the access token
        headers = {
            'Authorization': f'Bearer {session["access_token"]}',
            'Content-Type': 'application/json'
        }
        
        # Log request for debugging
        print(f"Making request to {API_BASE_URL}/ml/predict/histories with token {session['access_token'][:10]}...")
        
        # Make request to backend API
        response = requests.get(
            f'{API_BASE_URL}/ml/predict/histories',
            headers=headers,
            timeout=30
        )
        
        # Log response for debugging
        print(f"Received response with status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        if response.content:
            try:
                # Get raw content first for logging
                content_text = response.text
                print(f"Raw response content: {content_text[:1000]}...")  # Print first 1000 chars
                
                # Try to parse as JSON
                response_data = response.json()
                print(f"Parsed as JSON: {json.dumps(response_data)[:500]}...")  # Print first 500 chars
                
                # Check structure
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
            # Try to extract the content directly without modifying
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
    return render_template('main-feature/history-prediction.html')

@app.route('/history/<id>')
def detail_history(id):
    if 'access_token' not in session:
        flash('Silakan login terlebih dahulu.', 'error')
        return redirect(url_for('login'))
    return render_template('main-feature/detail_history-prediction.html', id=id)

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
        # Get message from request
        message = request.json.get('message')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Forward request to backend API
        headers = {
            'Authorization': f'Bearer {session["access_token"]}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{API_BASE_URL}/chatbot/',
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
    
    # Get user data from API
    try:
        headers = {
            'Authorization': f'Bearer {session["access_token"]}'
        }
        response = requests.get(f'{API_BASE_URL}/auth/me', headers=headers)
        
        if response.status_code == 200:
            user_data = response.json()
            # Format the created_at date if needed
            if 'created_at' in user_data:
                # Assuming created_at is in ISO format
                from datetime import datetime
                try:
                    # Parse the datetime and format it as DD-MM-YYYY
                    created_at = datetime.fromisoformat(user_data['created_at'].replace('Z', '+00:00'))
                    user_data['formatted_date'] = created_at.strftime('%d-%m-%Y')
                except:
                    # Fallback if date parsing fails
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


@app.route("/reset-password/<token>", methods=["GET"])
def reset_password_form(token):
    return render_template("auth/new-password.html", token=token)


@app.route('/logout')
def logout():
    session.pop('access_token', None)
    return redirect(url_for('home'))



# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
