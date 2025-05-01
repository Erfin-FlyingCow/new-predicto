import subprocess

# Jalankan backend
backend = subprocess.Popen(["python", "predicto-api/main.py"])

# Jalankan frontend
frontend = subprocess.Popen(["python", "predicto-frontend/app.py"])

# Tunggu sampai kedua proses selesai (Ctrl+C untuk stop)
backend.wait()
frontend.wait()

