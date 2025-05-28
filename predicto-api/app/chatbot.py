from flask import Flask, request, jsonify, Blueprint
from flask_jwt_extended import jwt_required
import json
import numpy as np
import random
import re
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os
import pickle
import tensorflow as tf
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import traceback
from statsmodels.tsa.seasonal import STL
from keras._tf_keras.keras.models import load_model
import warnings
import spacy
warnings.filterwarnings('ignore')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rules_data_path=os.path.join(BASE_DIR,'chatbot_data','rule_data.json')
historical_data_path=os.path.join(BASE_DIR,'chatbot_data','customer_segmentation_result .csv')
daily_model=os.path.join(BASE_DIR,'chatbot_data','model_xgb_daily.pkl')
weekly_model=os.path.join(BASE_DIR,'chatbot_data','stl_attlstm_model_weekly(0.94).h5')
monthly_model=os.path.join(BASE_DIR,'chatbot_data','bilstm_monthly_model.h5')

chatbot_bp = Blueprint('chatbot', __name__)


with open(rules_data_path, 'r', encoding='utf-8') as f:
        rules_data = json.load(f)


    # Download necessary NLTK resources
# Download necessary NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, download_dir="/chatbot_data/nltk")

download_nltk_resources()

try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("SpaCy model not found. Using basic keyword matching.")
    nlp = None

# Initialize the SentenceTransformer model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Good for Indonesian and English

# Preprocessing functions
lemmatizer = WordNetLemmatizer()
stop_words_id = set([
    'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir',
    'akhiri', 'akhirnya', 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara',
    'antaranya', 'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'artinya', 'asal',
    'asalkan', 'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'bagai', 'bagaikan',
    'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya',
    'baik', 'bakal', 'bakalan', 'balik', 'banyak', 'bapak', 'baru', 'bawah', 'beberapa', 'begini',
    'beginian', 'beginikah', 'beginilah', 'begitu', 'begitukah', 'begitulah', 'begitupun', 'bekerja',
    'belakang', 'belakangan', 'belum', 'belumlah', 'benar', 'benarkah', 'benarlah', 'berada',
    'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 'berapakah', 'berapalah', 'berapapun',
    'berarti', 'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya',
    'berjumlah', 'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan',
    'berlalu', 'berlangsung', 'berlebihan', 'bermacam', 'bermacam-macam', 'bermaksud', 'bermula',
    'bersama', 'bersama-sama', 'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya', 'berturut',
    'berturut-turut', 'bertutur', 'berujar', 'berupa', 'besar', 'betul', 'betulkah', 'biasa',
    'biasanya', 'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'bolehlah', 'buat',
    'bukan', 'bukankah', 'bukanlah', 'bukannya', 'bulan', 'bung', 'cara', 'caranya', 'cukup',
    'cukupkah', 'cukuplah', 'cuma', 'dahulu', 'dalam', 'dan', 'dapat', 'dari', 'daripada', 'datang',
    'dekat', 'demi', 'demikian', 'demikianlah', 'dengan', 'depan', 'di', 'dia', 'diakhiri',
    'diakhirinya', 'dialah', 'diantara', 'diantaranya', 'diberi', 'diberikan', 'diberikannya',
    'dibuat', 'dibuatnya', 'didapat', 'didatangkan', 'digunakan', 'diibaratkan', 'diibaratkannya',
    'diingat', 'diingatkan', 'diinginkan', 'dijawab', 'dijelaskan', 'dijelaskannya', 'dikarenakan',
    'dikatakan', 'dikatakannya', 'dikerjakan', 'diketahui', 'diketahuinya', 'dikira', 'dilakukan',
    'dilalui', 'dilihat', 'dimaksud', 'dimaksudkan', 'dimaksudkannya', 'dimaksudnya', 'diminta',
    'dimintai', 'dimisalkan', 'dimulai', 'dimulailah', 'dimulainya', 'dimungkinkan', 'dini',
    'dipastikan', 'diperbuat', 'diperbuatnya', 'dipergunakan', 'diperkirakan', 'diperlihatkan',
    'diperlukan', 'diperlukannya', 'dipersoalkan', 'dipertanyakan', 'dipunyai', 'diri', 'dirinya',
    'disampaikan', 'disebut', 'disebutkan', 'disebutkannya', 'disini', 'disinilah', 'ditambahkan',
    'ditandaskan', 'ditanya', 'ditanyai', 'ditanyakan', 'ditegaskan', 'ditujukan', 'ditunjuk',
    'ditunjuki', 'ditunjukkan', 'ditunjukkannya', 'ditunjuknya', 'dituturkan', 'dituturkannya',
    'diucapkan', 'diucapkannya', 'diungkapkan', 'dong', 'dua', 'dulu', 'empat', 'enggak',
    'enggaknya', 'entah', 'entahlah', 'guna', 'gunakan', 'hal', 'hampir', 'hanya', 'hanyalah',
    'hari', 'harus', 'haruslah', 'harusnya', 'hendak', 'hendaklah', 'hendaknya', 'hingga', 'ia',
    'ialah', 'ibarat', 'ibaratkan', 'ibaratnya', 'ibu', 'ikut', 'ingat', 'ingat-ingat', 'ingin',
    'inginkah', 'inginkan', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jadilah',
    'jadinya', 'jangan', 'jangankan', 'janganlah', 'jauh', 'jawab', 'jawaban', 'jawabnya', 'jelas',
    'jelaskan', 'jelaslah', 'jelasnya', 'jika', 'jikalau', 'juga', 'jumlah', 'jumlahnya', 'justru',
    'kala', 'kalau', 'kalaulah', 'kalaupun', 'kalian', 'kami', 'kamilah', 'kamu', 'kamulah', 'kan',
    'kapan', 'kapankah', 'kapanpun', 'karena', 'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah',
    'katanya', 'ke', 'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 'keinginan', 'kelamaan',
    'kelihatan', 'kelihatannya', 'kelima', 'keluar', 'kembali', 'kemudian', 'kemungkinan',
    'kemungkinannya', 'kenapa', 'kepada', 'kepadanya', 'kesampaian', 'keseluruhan', 'keseluruhannya',
    'keterlaluan', 'ketika', 'khususnya', 'kini', 'kinilah', 'kira', 'kira-kira', 'kiranya', 'kita',
    'kitalah', 'kok', 'kurang', 'lagi', 'lagian', 'lah', 'lain', 'lainnya', 'lalu', 'lama',
    'lamanya', 'lanjut', 'lanjutnya', 'lebih', 'lewat', 'lima', 'luar', 'macam', 'maka', 'makanya',
    'makin', 'malah', 'malahan', 'mampu', 'mampukah', 'mana', 'manakala', 'manalagi', 'masa',
    'masalah', 'masalahnya', 'masih', 'masihkah', 'masing', 'masing-masing', 'mau', 'maupun',
    'melainkan', 'melakukan', 'melalui', 'melihat', 'melihatnya', 'memang', 'memastikan', 'memberi',
    'memberikan', 'membuat', 'memerlukan', 'memihak', 'meminta', 'memintakan', 'memisalkan',
    'memperbuat', 'mempergunakan', 'memperkirakan', 'memperlihatkan', 'mempersiapkan',
    'mempersoalkan', 'mempertanyakan', 'mempunyai', 'memulai', 'memungkinkan', 'menaiki',
    'menambahkan', 'menandaskan', 'menanti', 'menanti-nanti', 'menantikan', 'menanya', 'menanyai',
    'menanyakan', 'mendapat', 'mendapatkan', 'mendatang', 'mendatangi', 'mendatangkan',
    'menegaskan', 'mengakhiri', 'mengapa', 'mengatakan', 'mengatakannya', 'mengenai',
    'mengerjakan', 'mengetahui', 'menggunakan', 'menghendaki', 'mengibaratkan', 'mengibaratkannya',
    'mengingat', 'mengingatkan', 'menginginkan', 'mengira', 'mengucapkan', 'mengucapkannya',
    'mengungkapkan', 'menjadi', 'menjawab', 'menjelaskan', 'menuju', 'menunjuk', 'menunjuki',
    'menunjukkan', 'menunjuknya', 'menurut', 'menuturkan', 'menyampaikan', 'menyangkut',
    'menyatakan', 'menyebutkan', 'menyeluruh', 'menyiapkan', 'merasa', 'mereka', 'merekalah',
    'merupakan', 'meski', 'meskipun', 'meyakini', 'meyakinkan', 'minta', 'mirip', 'misal',
    'misalkan', 'misalnya', 'mula', 'mulai', 'mulailah', 'mulanya', 'mungkin', 'mungkinkah', 'nah',
    'naik', 'namun', 'nanti', 'nantinya', 'nyaris', 'nyatanya', 'oleh', 'olehnya', 'pada', 'padahal',
    'padanya', 'pak', 'paling', 'panjang', 'pantas', 'para', 'pasti', 'pastilah', 'penting', 'pentingnya', 'per', 'percuma',
    'perlu', 'perlukah', 'perlunya', 'pernah', 'persoalan', 'pertama', 'pertama-tama', 'pertanyaan',
    'pertanyakan', 'pihak', 'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'rasa', 'rasanya', 'rata',
    'rupanya', 'saat', 'saatnya', 'saja', 'sajalah', 'saling', 'sama', 'sama-sama', 'sambil', 'sampai',
    'sampai-sampai', 'sampaikan', 'sana', 'sangat', 'sangatlah', 'satu', 'saya', 'sayalah', 'se', 'sebab',
    'sebabnya', 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian', 'sebaik', 'sebaik-baiknya',
    'sebaiknya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu', 'sebelum', 'sebelumnya', 'sebenarnya',
    'seberapa', 'sebesar', 'sebetulnya', 'sebisanya', 'sebuah', 'sebut', 'sebutlah', 'sebutnya', 'secara',
    'secukupnya', 'sedang', 'sedangkan', 'sedemikian', 'sedikit', 'sedikitnya', 'seenaknya', 'segala',
    'segalanya', 'segera', 'seharusnya', 'sehingga', 'seingat', 'sejak', 'sejauh', 'sejenak', 'sejumlah',
    'sekadar', 'sekadarnya', 'sekali', 'sekali-kali', 'sekalian', 'sekaligus', 'sekalipun', 'sekarang',
    'sekecil', 'seketika', 'sekiranya', 'sekitar', 'sekitarnya', 'sekurang-kurangnya', 'sekurangnya',
    'sela', 'selain', 'selaku', 'selalu', 'selama', 'selama-lamanya', 'selamanya', 'selanjutnya',
    'seluruh', 'seluruhnya', 'semacam', 'semakin', 'semampu', 'semampunya', 'semasa', 'semasih',
    'semata', 'semata-mata', 'semaunya', 'sementara', 'semisal', 'semisalnya', 'sempat', 'semua',
    'semuanya', 'semula', 'sendiri', 'sendirian', 'sendirinya', 'seolah', 'seolah-olah', 'seorang',
    'sepanjang', 'sepantasnya', 'sepantasnyalah', 'seperlunya', 'seperti', 'sepertinya', 'sepihak',
    'sering', 'seringnya', 'serta', 'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera', 'sesekali',
    'seseorang', 'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah',
    'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidak-tidaknya', 'setidaknya', 'setinggi',
    'seusai', 'sewaktu', 'siap', 'siapa', 'siapakah', 'siapapun', 'sini', 'sinilah', 'soal', 'soalnya',
    'suatu', 'sudah', 'sudahkah', 'sudahlah', 'supaya', 'tadi', 'tadinya', 'tahu', 'tahun', 'tak',
    'tambah', 'tambahnya', 'tampak', 'tampaknya', 'tandas', 'tandasnya', 'tanpa', 'tanya', 'tanyakan',
    'tanyanya', 'tapi', 'tegas', 'tegasnya', 'telah', 'tempat', 'tengah', 'tentang', 'tentu', 'tentulah',
    'tentunya', 'tepat', 'terakhir', 'terasa', 'terbanyak', 'terdahulu', 'terdapat', 'terdiri',
    'terhadap', 'terhadapnya', 'teringat', 'teringat-ingat', 'terjadi', 'terjadilah', 'terjadinya',
    'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata', 'tersampaikan', 'tersebut',
    'tersebutlah', 'tertentu', 'tertuju', 'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba',
    'tiba-tiba', 'tidak', 'tidakkah', 'tidaklah', 'tiga', 'tinggi', 'toh', 'tunjuk', 'turut', 'tutur',
    'tuturnya', 'ucap', 'ucapnya', 'ujar', 'ujarnya', 'umum', 'umumnya', 'ungkap', 'ungkapnya', 'untuk',
    'usah', 'usai', 'waduh', 'wah', 'wahai', 'waktu', 'waktunya', 'walau', 'walaupun', 'wong', 'yaitu',
    'yakin', 'yakni', 'yang'
])

DOMAIN_KEYWORDS = {
    'sales': ['penjualan', 'sales', 'jual', 'revenue', 'pendapatan', 'omzet', 'transaksi', 'margin', 'lower bound', 'upper bound'],
    'prediction': ['prediksi', 'forecast', 'ramalan', 'perkiraan', 'estimasi'],
    'product': ['iphone', 'produk', 'barang', 'handphone', 'hp', 'smartphone'],
    'business': ['bisnis', 'usaha', 'toko', 'perusahaan', 'website', 'online', 'predicto'],
    'time': ['hari', 'minggu', 'bulan', 'harian', 'mingguan', 'bulanan', 'besok', 'depan'],
    'price': ['harga', 'price', 'cost', 'biaya'],
    'model': ['lstm', 'prophet', 'xgboost', 'ai', 'model', 'akurasi'],
    'platform': ['platform', 'api', 'integrasi', 'berlangganan', 'paket']
}

IRRELEVANT_KEYWORDS = {
    'automotive': ['mobil', 'motor', 'kendaraan', 'otomotif', 'transportasi'],
    'real_estate': ['rumah', 'properti', 'tanah', 'apartemen'],
    'food': ['makanan', 'restoran', 'kuliner', 'masak'],
    'travel': ['wisata', 'travel', 'liburan', 'hotel'],
    'weather': ['cuaca', 'weather', 'iklim', 'hujan', 'panas'],  
    'finance': ['saham', 'emas', 'gold', 'forex', 'crypto', 'bitcoin'],  
    'education': ['sekolah', 'universitas', 'kuliah', 'belajar'],
    'health': ['obat', 'dokter', 'rumah sakit', 'kesehatan'],
    'fashion': ['baju', 'sepatu', 'fashion', 'pakaian'],
    'politics': ['politik', 'pemilu', 'presiden', 'gubernur'] 
}

stop_words_en = set(stopwords.words('english'))
stop_words = stop_words_id.union(stop_words_en)

# Dictionary to store loaded ML models
ml_models = {
    'daily': None,
    'weekly': None,
    'monthly': None
}

# Dictionary to store model information
model_info = {
    'daily': {
        'file': daily_model, 
        'format': 'pkl',
        'description': 'XGBoost dengan fitur lag dan rolling statistics',
        'features': ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_month_start', 'is_month_end', 
                   'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', 'dayofyear', 
                   'weekofyear', 'is_weekend', 'is_holiday', 'year_sin', 'year_cos', 'month_sin', 
                   'month_cos', 'week_sin', 'week_cos', 'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 
                   'lag_21', 'lag_28', 'lag_30', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 
                   'rolling_std_14', 'rolling_mean_30', 'rolling_std_30', 'expanding_mean', 
                   'expanding_std', 'pct_change_1', 'pct_change_7', 'pct_change_28']
    },
    'weekly': {
        'file': weekly_model,
        'format': 'h5',
        'description': 'STT-ATTLSTM Hybrid dengan dekomposisi STL',
        'window_size': 16,
        'stl_period': 52
    },
    'monthly': {
        'file': monthly_model,
        'format': 'h5',
        'description': 'Bidirectional LSTM dengan normalisasi dan sequence prediction',
        'window_size': 12,
        'stl_period': 12
    }
}

# Data used for scaling and STL decomposition
historical_data = {
    'daily': None,
    'weekly': None,
    'monthly': None
}

# Store scalers for each model type
scalers = {
    'daily': None,
    'weekly': {
        'resid': StandardScaler(),
        'other': MinMaxScaler()
    },
    'monthly': MinMaxScaler()
}

# Helper function to create date features - Updated to match model's expected feature names
def add_date_features(df, date_col='date'):
    df = df.copy()
    
    # Using feature names required by the model
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['quarter'] = df[date_col].dt.quarter
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_col].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_col].dt.is_year_end.astype(int)
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Simple holiday detection
    df['is_holiday'] = ((df['month'] == 1) & (df['day'] == 1)).astype(int)

    # Cyclical features
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 2030)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 2030)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)

    return df


# Load historical data
def load_historical_data():
    try:
        df = pd.read_csv(historical_data_path)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df = df.sort_values('transaction_date')

        # Agregasi harian
        daily_df = df.groupby(pd.Grouper(key='transaction_date', freq='D'))['total'].sum().reset_index()
        daily_df.rename(columns={'transaction_date': 'date', 'total': 'sales'}, inplace=True)
        daily_df = add_date_features(daily_df)

        # Agregasi mingguan dan bulanan
        weekly_df = daily_df.resample('W-MON', on='date').sum().reset_index()
        monthly_df = daily_df.resample('M', on='date').sum().reset_index()

        # Simpan ke struktur chatbot
        historical_data['daily'] = daily_df
        historical_data['weekly'] = weekly_df
        historical_data['monthly'] = monthly_df

        print("✅ Data historis dari customer_segmentation_result.csv berhasil dimuat.")
        return True

    except Exception as e:
        print(f"❌ Gagal memuat data historis: {e}")
        return False


# Function to add lag features for daily model
def add_lag_features(df, lags=[1, 2, 3, 7, 14, 21, 28, 30]):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['sales'].shift(lag)
    
    # Add rolling statistics
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df['sales'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['sales'].rolling(window=window).std()
    
    # Add expanding stats
    df['expanding_mean'] = df['sales'].expanding().mean()
    df['expanding_std'] = df['sales'].expanding().std()
    
    # Add percentage changes
    df['pct_change_1'] = df['sales'].pct_change(periods=1)
    df['pct_change_7'] = df['sales'].pct_change(periods=7)
    df['pct_change_28'] = df['sales'].pct_change(periods=28)
    
    return df

# Function to create sequences for LSTM models
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Function to create dual sequences for STT-ATTLSTM model
def create_dual_sequences(resid, other, window_size):
    X_resid, X_other, y = [], [], []
    for i in range(len(resid) - window_size):
        X_resid.append(resid[i:i+window_size])
        X_other.append(other[i:i+window_size])
        y.append(resid[i+window_size])
    return np.expand_dims(np.array(X_resid), 2), np.array(X_other), np.array(y)

# Prepare models and scalers
def prepare_models():
    # First load historical data
    if not load_historical_data():
        return False

    try:
        ##### === DAILY MODEL === #####
        daily_df = historical_data['daily'].copy()

        # Tambahkan fitur tanggal terlebih dahulu
        daily_df = add_date_features(daily_df)

        # Tambahkan fitur lag dan rolling
        daily_df_with_lags = add_lag_features(daily_df)
        daily_df_with_lags = daily_df_with_lags.dropna()

        # Validasi semua fitur tersedia
        feature_cols = model_info['daily']['features']
        missing_cols = [col for col in feature_cols if col not in daily_df_with_lags.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in daily data: {missing_cols}")

        # Simpan nilai terakhir untuk prediksi
        latest_values = daily_df_with_lags.iloc[-1:][feature_cols]

        # Load model harian
        daily_model_path = model_info['daily']['file']
        if os.path.exists(daily_model_path):
            with open(daily_model_path, 'rb') as file:
                ml_models['daily'] = pickle.load(file)
            print(f"✅ Loaded daily model from {daily_model_path}")
        else:
            print(f"❌ Daily model file not found: {daily_model_path}")

        ##### === WEEKLY MODEL === #####
        weekly_df = historical_data['weekly']
        if weekly_df is not None and os.path.exists(model_info['weekly']['file']):
            window_size = model_info['weekly']['window_size']
            stl_period = model_info['weekly']['stl_period']

            stl = STL(weekly_df['sales'], period=stl_period)
            res = stl.fit()

            resid = res.resid
            trend = res.trend
            seasonal = res.seasonal

            resid_scaled = scalers['weekly']['resid'].fit_transform(resid.values.reshape(-1, 1)).flatten()
            other_scaled = scalers['weekly']['other'].fit_transform(np.vstack([trend, seasonal]).T)

            last_resid_seq = resid_scaled[-window_size:].reshape(1, window_size, 1)
            last_other_seq = other_scaled[-window_size:].reshape(1, window_size, 2)
            last_resid_input = last_resid_seq[:, -1, 0].reshape(-1, 1)

            historical_data['weekly_stl'] = {
                'resid': resid,
                'trend': trend,
                'seasonal': seasonal,
                'last_resid_seq': last_resid_seq,
                'last_other_seq': last_other_seq,
                'last_resid_input': last_resid_input
            }

            ml_models['weekly'] = load_model(model_info['weekly']['file'], compile=False)
            print(f"✅ Loaded weekly model from {model_info['weekly']['file']}")
        else:
            print(f"❌ Weekly model file not found or data missing")

        ##### === MONTHLY MODEL === #####
        monthly_df = historical_data['monthly']
        if monthly_df is not None and os.path.exists(model_info['monthly']['file']):
            window_size = model_info['monthly']['window_size']
            stl_period = model_info['monthly']['stl_period']

            stl = STL(monthly_df['sales'], period=stl_period)
            result = stl.fit()

            trend_monthly = result.trend
            seasonal_monthly = result.seasonal
            resid_monthly = result.resid

            resid_scaled_monthly = scalers['monthly'].fit_transform(resid_monthly.values.reshape(-1, 1))
            last_seq = resid_scaled_monthly[-window_size:].reshape(1, window_size, 1)

            historical_data['monthly_stl'] = {
                'trend': trend_monthly,
                'seasonal': seasonal_monthly,
                'resid': resid_monthly,
                'last_seq': last_seq
            }

            ml_models['monthly'] = load_model(model_info['monthly']['file'], custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
            print(f"✅ Loaded monthly model from {model_info['monthly']['file']}")
        else:
            print(f"❌ Monthly model file not found or data missing")

        return True

    except Exception as e:
        print(f"❌ Error preparing models: {e}")
        traceback.print_exc()
        return False


# Try to prepare models when starting the application
try:
    prepare_models()
except Exception as e:
    print(f"❌ Could not prepare ML models: {e}")
    traceback.print_exc()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{string.punctuation}]", " ", text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    # Join back to string
    return " ".join(tokens)

# Calculate embeddings for all patterns in the rules
rule_embeddings = {}
rule_patterns = {}

for rule_name, rule_content in rules_data.items():
    patterns = rule_content['patterns']
    if patterns:  # Only process rules with patterns
        # Preprocess patterns
        preprocessed_patterns = [preprocess_text(pattern) for pattern in patterns]
        # Calculate embeddings
        embeddings = model.encode(preprocessed_patterns)
        rule_embeddings[rule_name] = embeddings
        rule_patterns[rule_name] = preprocessed_patterns

# Add special prediction rules
prediction_rules = {
    'daily_prediction': {
        'patterns': [
            "prediksi penjualan harian",
            "prediksi besok",
            "prediksi hari ini",
            "prediksi harian",
            "forecast besok",
            "forecast hari ini",
            "bagaimana prediksi besok",
            "bagaimana prediksi hari ini",
            "perkiraan penjualan besok",
            "forecast harian",
            "ramalan penjualan besok"
        ]
    },
    'weekly_prediction': {
        'patterns': [
            "prediksi penjualan mingguan"
            "prediksi minggu depan",
            "prediksi mingguan",
            "forecast minggu ini",
            "forecast mingguan",
            "bagaimana prediksi minggu depan",
            "perkiraan penjualan minggu depan",
            "ramalan penjualan seminggu ke depan"
        ]
    },
    'monthly_prediction': {
        'patterns': [
            "prediksi penjualan bulanan",
            "prediksi bulan depan",
            "prediksi bulanan",
            "forecast bulan ini",
            "forecast bulanan",
            "bagaimana prediksi bulan depan",
            "perkiraan penjualan bulan depan",
            "ramalan penjualan sebulan ke depan"
        ]
    }
}

# Add prediction rules to rule_embeddings and rule_patterns
for rule_name, rule_content in prediction_rules.items():
    patterns = rule_content['patterns']
    if patterns:
        preprocessed_patterns = [preprocess_text(pattern) for pattern in patterns]
        embeddings = model.encode(preprocessed_patterns)
        rule_embeddings[rule_name] = embeddings
        rule_patterns[rule_name] = preprocessed_patterns

def extract_keywords(text):
    """Extract keywords dari text"""
    keywords = []
    
    if nlp:
        doc = nlp(text.lower())
        keywords = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and len(token.text) > 2]
    else:
        # Basic keyword extraction
        keywords = [word for word in word_tokenize(text.lower()) if len(word) > 2]
    
    return list(set(keywords))

def check_domain_relevance(user_input):
    """Check apakah input user relevan dengan domain bisnis"""
    keywords = extract_keywords(user_input)
    
    # Hitung score untuk domain yang relevan
    relevant_score = 0
    for domain, domain_keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if any(dk in keyword or keyword in dk for dk in domain_keywords):
                relevant_score += 1
    
    # Hitung score untuk domain yang tidak relevan
    irrelevant_score = 0
    for domain, irrelevant_keywords in IRRELEVANT_KEYWORDS.items():
        for keyword in keywords:
            if any(ik in keyword or keyword in ik for ik in irrelevant_keywords):
                irrelevant_score += 2  # Beri weight lebih tinggi untuk irrelevant
    
    return relevant_score, irrelevant_score, keywords

def find_best_rule_match(user_input):
    """Enhanced rule matching dengan context awareness"""
    
    # 1. Check domain relevance terlebih dahulu
    relevant_score, irrelevant_score, extracted_keywords = check_domain_relevance(user_input)
    
    print(f"\033[90m[Debug] Keywords: {extracted_keywords}\033[0m")
    print(f"\033[90m[Debug] Relevant score: {relevant_score}, Irrelevant score: {irrelevant_score}\033[0m")

    out_of_context_keywords = [
    # Cuaca & Alam
    'cuaca', 'weather', 'hujan', 'panas', 'dingin', 'angin', 'badai', 'topan',
    'gempa', 'tsunami', 'banjir', 'kekeringan', 'musim', 'iklim', 'suhu',
    
    # Keuangan & Investasi (Non-iPhone)
    'emas', 'gold', 'perak', 'silver', 'saham', 'stock', 'crypto', 'bitcoin',
    'ethereum', 'forex', 'mata uang', 'currency', 'rupiah', 'dollar', 'euro',
    'obligasi', 'reksadana', 'deposito', 'investasi emas', 'trading', 'bursa',
    'ihsg', 'nasdaq', 'dow jones', 'nikkei', 'hang seng',
    
    # Properti & Real Estate
    'properti', 'real estate', 'tanah', 'rumah', 'apartemen', 'villa', 'ruko',
    'gedung', 'perkantoran', 'mall', 'hotel', 'resort', 'kost', 'kontrakan',
    'sewa', 'beli rumah', 'kredit rumah', 'kpr', 'pajak bumi', 'sertifikat',
    
    # Otomotif
    'mobil', 'motor', 'kendaraan', 'otomotif', 'toyota', 'honda', 'suzuki',
    'yamaha', 'kawasaki', 'bmw', 'mercedes', 'audi', 'ferrari', 'lamborghini',
    'truck', 'bus', 'sepeda', 'vespa', 'harley', 'bensin', 'solar', 'oli',
    'ban', 'velg', 'sparepart', 'service', 'bengkel', 'showroom', 'kredit mobil',
    
    # Makanan & Kuliner
    'makanan', 'restoran', 'cafe', 'warung', 'kedai', 'bakso', 'soto', 'rendang',
    'nasi goreng', 'mie ayam', 'gado-gado', 'satay', 'gudeg', 'rawon', 'sate',
    'pizza', 'burger', 'sushi', 'ramen', 'pasta', 'steak', 'seafood', 'dimsum',
    'kopi', 'teh', 'jus', 'smoothie', 'ice cream', 'cake', 'donut', 'cookies',
    'masak', 'recipe', 'bahan', 'bumbu', 'resep', 'chef', 'cooking',
    
    # Travel & Pariwisata
    'wisata', 'travel', 'liburan', 'vacation', 'holiday', 'pantai', 'gunung',
    'danau', 'air terjun', 'candi', 'museum', 'taman', 'kebun binatang',
    'bali', 'jogja', 'bandung', 'malang', 'lombok', 'batam', 'medan', 'surabaya',
    'singapore', 'malaysia', 'thailand', 'jepang', 'korea', 'eropa', 'amerika',
    'tiket', 'booking', 'hotel', 'penginapan', 'homestay', 'villa', 'resort',
    'pesawat', 'kereta', 'bus', 'kapal', 'cruise', 'backpacker', 'tour',
    
    # Kesehatan & Medis
    'kesehatan', 'obat', 'dokter', 'rumah sakit', 'klinik', 'puskesmas',
    'sakit', 'penyakit', 'flu', 'demam', 'batuk', 'pilek', 'covid', 'virus',
    'vaksin', 'vitamin', 'suplemen', 'therapy', 'fisioterapi', 'psikolog',
    'diet', 'olahraga', 'fitness', 'gym', 'yoga', 'senam', 'lari', 'renang',
    'jantung', 'diabetes', 'hipertensi', 'kolesterol', 'asam urat', 'maag',
    
    # Politik & Pemerintahan
    'politik', 'pemilu', 'pilpres', 'pileg', 'pilkada', 'presiden', 'menteri',
    'gubernur', 'bupati', 'walikota', 'dpr', 'dprd', 'mpr', 'mahkamah',
    'pemerintah', 'negara', 'undang-undang', 'peraturan', 'kebijakan',
    'korupsi', 'kpk', 'polisi', 'tentara', 'tni', 'polri',
    
    # Pendidikan (Non-Teknis)
    'mulyono', 'ukt', 'pnj', 'politeknik negeri jakarta', 'universitas',
    'sekolah', 'sma', 'smk', 'smp', 'sd', 'tk', 'kuliah', 'kampus',
    'dosen', 'guru', 'mahasiswa', 'siswa', 'ujian', 'tes', 'skripsi',
    'thesis', 'nilai', 'ipk', 'beasiswa', 'lomba', 'olimpiade',
    
    # Waktu & Tanggal
    'jam', 'waktu', 'tanggal', 'tahun',
    'lebaran', 'natal', 'nyepi', 'waisak', 'ramadan', 'puasa', 'sahur',
    'berbuka', 'mudik', 'libur', 'cuti', 'weekend', 'senin', 'selasa',
    'rabu', 'kamis', 'jumat', 'sabtu',
    
    # Fashion & Clothing
    'jaket', 'sepatu', 'baju', 'celana', 'rok', 'dress', 'kemeja', 'kaos',
    'hoodie', 'sweater', 'blazer', 'jas', 'topi', 'tas', 'dompet', 'jam tangan',
    'gelang', 'kalung', 'cincin', 'anting', 'kacamata', 'sandal', 'boot',
    'sneakers', 'heels', 'flat shoes', 'brand', 'fashion', 'style', 'trend',
    'zara', 'h&m', 'uniqlo', 'nike', 'adidas', 'puma', 'converse', 'vans',
    
    # Nama Orang (Contoh)
    'haidar', 'hilmi', 'rayhan', 'budi', 'siti', 'ahmad', 'fitri', 'andi',
    'dewi', 'rini', 'agus', 'tono', 'wati', 'joko', 'sri', 'dian', 'rina',
    
    # Hiburan & Entertainment
    'spotify', 'netflix', 'youtube', 'instagram', 'facebook', 'twitter',
    'tiktok', 'whatsapp', 'telegram', 'discord', 'music', 'lagu', 'band',
    'singer', 'artist', 'concert', 'konser', 'film', 'movie', 'cinema',
    'bioskop', 'drama', 'series', 'anime', 'manga', 'novel', 'buku',
    'game', 'gaming', 'ps5', 'xbox', 'nintendo', 'mobile legends', 'pubg',
    'free fire', 'valorant', 'dota', 'lol', 'fifa', 'pes', 'streaming',
    
    # Gadgets & Electronics (Non-iPhone Prediction)
    'laptop', 'hp', 'handphone', 'samsung', 'xiaomi', 'oppo', 'vivo', 'realme',
    'huawei', 'oneplus', 'google pixel', 'sony', 'lg', 'nokia', 'blackberry',
    'tablet', 'ipad', 'computer', 'pc', 'monitor', 'keyboard', 'mouse',
    'headphone', 'earphone', 'speaker', 'tv', 'smart tv', 'camera', 'drone',
    'smartwatch', 'fitness tracker', 'powerbank', 'charger', 'cable',
    'psp', 'nintendo switch', 'playstation', 'xbox', 'gaming chair',
    
    # Hobi & Lifestyle
    'resep', 'ramuan', 'cinta', 'sayang', 'pacaran', 'jodoh', 'nikah',
    'kawin', 'menikah', 'wedding', 'lamaran', 'tunangan', 'valentine',
    'anniversary', 'ultah', 'birthday', 'party', 'pesta', 'acara',
    'foto', 'selfie', 'photography', 'photoshoot', 'video', 'vlog',
    'blog', 'content', 'influencer', 'selebgram', 'youtuber', 'tiktoker',
    
    # Pekerjaan & Profesi
    'kerja', 'pekerjaan', 'karir', 'job', 'career', 'interview', 'cv',
    'resume', 'lowongan', 'vacancy', 'hiring', 'recruitment', 'hr',
    'manager', 'direktur', 'ceo', 'staff', 'karyawan', 'pegawai',
    'freelance', 'part time', 'full time', 'intern', 'magang', 'gaji',
    'salary', 'bonus', 'thr', 'cuti', 'resign', 'pensiun',
    
    # Agama & Spiritual
    'agama', 'islam', 'kristen', 'katolik', 'hindu', 'buddha', 'konghucu',
    'sholat', 'doa', 'dzikir', 'quran', 'bible', 'gereja', 'masjid',
    'vihara', 'pura', 'klenteng', 'ustadz', 'pastor', 'pendeta', 'biksu',
    'rohani', 'spiritual', 'meditasi', 'taubat', 'tobat', 'berkah',
    
    # Olahraga & Sports
    'sepakbola', 'basket', 'voli', 'badminton', 'tenis', 'golf', 'boxing',
    'mma', 'martial arts', 'karate', 'taekwondo', 'silat', 'judo',
    'atletik', 'renang', 'diving', 'surfing', 'skateboard', 'cycling',
    'marathon', 'triathlon', 'olimpiade', 'asian games', 'sea games',
    'liga', 'pertandingan', 'match', 'tournament', 'championship',
    
    # Sains & Pengetahuan Umum
    'fisika', 'kimia', 'biologi', 'matematika', 'sejarah', 'geografi',
    'astronomi', 'planet', 'bintang', 'galaxy', 'alien', 'ufo',
    'dinosaurus', 'evolusi', 'dna', 'gen', 'sel', 'virus', 'bakteri',
    'atom', 'molekul', 'energi', 'gravitasi', 'relativitas', 'quantum',
    
    # Supernatural & Mistis
    'hantu', 'setan', 'jin', 'roh', 'arwah', 'pocong', 'kuntilanak',
    'sundel bolong', 'leak', 'penanggalan', 'mistis', 'supranatural',
    'paranormal', 'dukun', 'santet', 'guna-guna', 'pelet', 'susuk',
    'amulet', 'jimat', 'keris', 'mantra', 'ritual', 'sesajen',
    
    # Ekonomi Makro
    'inflasi', 'deflasi', 'resesi', 'ekonomi', 'pdb', 'gdp', 'imf',
    'world bank', 'bank indonesia', 'bi rate', 'suku bunga', 'kredit',
    'kurs', 'ekspor', 'impor', 'neraca', 'surplus', 'defisit',
    
    # Transportasi
    'ojek', 'gojek', 'grab', 'uber', 'taxi', 'angkot', 'metromini',
    'transjakarta', 'mrt', 'lrt', 'krl', 'commuter line', 'airport',
    'bandara', 'terminal', 'stasiun', 'pelabuhan', 'toll', 'macet',
    'traffic', 'parkir', 'bensin', 'premium', 'pertamax', 'solar',
    
    # Teknologi Lain (Bukan AI/Prediksi)
    'blockchain', 'nft', 'metaverse', 'vr', 'iot', 'cloud',
    'server', 'database', 'sql', 'nosql', 'api', 'backend', 'frontend',
    'ui', 'ux', 'design', 'photoshop', 'illustrator', 'figma',
    'wordpress', 'laravel', 'react', 'angular', 'vue', 'node', 'python',
    'java', 'php', 'javascript', 'html', 'css', 'bootstrap', 'tailwind',
    
    # Bahasa Gaul & Slang
    'anjay', 'mantap', 'keren', 'asik', 'seru', 'gokil', 'ngakak',
    'baper', 'gabut', 'kepo', 'julid', 'toxic', 'salty', 'haters',
    'fans', 'stan', 'ship', 'otp', 'bias', 'visual', 'vocal', 'rapper',
    'dancer', 'idol', 'kpop', 'jpop', 'cpop', 'comeback', 'debut',
    
    # Lain-lain
    'cuci', 'nyuci', 'setrika', 'pel', 'sapu', 'vacuum', 'deterjen',
    'sabun', 'shampoo', 'pasta gigi', 'sikat gigi', 'tissue', 'tisu',
    'popok', 'diapers', 'susu', 'bubur', 'biskuit', 'roti', 'mentega',
    'jam', 'menit', 'detik', 'alarm', 'timer', 'stopwatch', 'kalender',
    'agenda', 'jadwal', 'appointment', 'meeting', 'rapat', 'presentasi'
]

    user_lower = user_input.lower()
    for keyword in out_of_context_keywords:
        if keyword in user_lower:
            print(f"\033[90m[Debug] Found out-of-context keyword: {keyword}\033[0m")
            return 'out_of_scope', 0.0, f"Detected out-of-context keyword: {keyword}"
    
    # Jika input jelas-jelas tidak relevan dengan domain bisnis
    if irrelevant_score > relevant_score and irrelevant_score > 0:
        return 'out_of_scope', 0.0, "Input tidak relevan dengan domain bisnis"
    
    # 2. Preprocess user input
    preprocessed_input = preprocess_text(user_input)
    
    # 3. Calculate embedding for user input
    input_embedding = model.encode([preprocessed_input])[0]
    
    # 4. Find the rule with the highest similarity
    best_rule = None
    best_similarity = -1
    best_pattern = None
    
    # First check for exact matches
    for rule_name, patterns in rule_patterns.items():
        for i, pattern in enumerate(patterns):
            if preprocessed_input == pattern:
                return rule_name, 1.0, pattern
    
    # Then check for semantic similarity dengan context filtering
    candidates = []
    
    for rule_name, embeddings in rule_embeddings.items():
        for i, embedding in enumerate(embeddings):
            similarity = np.dot(input_embedding, embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(embedding)
            )
            
            # Context-aware filtering
            pattern_text = rule_patterns[rule_name][i]
            pattern_keywords = extract_keywords(pattern_text)
            
            # Hitung keyword overlap
            keyword_overlap = len(set(extracted_keywords) & set(pattern_keywords))
            
            # Boost similarity jika ada keyword overlap
            if keyword_overlap > 0:
                similarity += 0.05 * keyword_overlap  # Small boost
            
            candidates.append((rule_name, similarity, rule_patterns[rule_name][i]))
    
    # Sort candidates by similarity
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    if candidates:
        best_rule, best_similarity, best_pattern = candidates[0]
        
        # Additional context check untuk prediksi
        if 'prediction' in best_rule and relevant_score == 0:
            # Jika match prediction tapi tidak ada context bisnis yang relevan
            best_similarity *= 0.7  # Reduce confidence
    
    return best_rule, best_similarity, best_pattern

# Prediction Functions
def predict_daily():
    """Generate daily sales prediction for next 7 days using current date"""
    try:
        if ml_models['daily'] is None:
            print("❌ Error in daily prediction: model belum tersedia")
            return "Model harian belum tersedia."
        
        daily_df = historical_data['daily']
        model = ml_models['daily']
        
        # Get current date instead of last date in data
        current_date = datetime.datetime.now().date()
        
        # Generate prediction dates starting from tomorrow
        future_dates = [current_date + datetime.timedelta(days=i+1) for i in range(7)]
        
        # Initialize prediction results
        predictions = []
        lower_bounds = []
        upper_bounds = []
        
        # Create base dataframe with date features
        future_df = pd.DataFrame({'date': future_dates})
        future_df['date'] = pd.to_datetime(future_df['date']) 
        future_df = add_date_features(future_df)
        
        # Get latest data with lags - need to adjust this if there's a gap between data and current date
        latest_data = add_lag_features(daily_df.iloc[-30:].copy())
        
        # Iteratively predict next 7 days
        for i in range(7):
            # Prepare next row
            next_row = future_df.iloc[i:i+1].copy()
            
            # Add lag features
            if i == 0:
                # First prediction uses known historical data
                for lag in [1, 2, 3, 7, 14, 21, 28, 30]:
                    lag_idx = -lag
                    next_row[f'lag_{lag}'] = daily_df['sales'].iloc[lag_idx]
                
                # Add rolling statistics
                for window in [7, 14, 30]:
                    next_row[f'rolling_mean_{window}'] = daily_df['sales'].iloc[-window:].mean()
                    next_row[f'rolling_std_{window}'] = daily_df['sales'].iloc[-window:].std()
                
                # Add expanding stats
                next_row['expanding_mean'] = daily_df['sales'].mean()
                next_row['expanding_std'] = daily_df['sales'].std()
                
                # Add percentage changes
                next_row['pct_change_1'] = daily_df['sales'].iloc[-1] / daily_df['sales'].iloc[-2] - 1
                next_row['pct_change_7'] = daily_df['sales'].iloc[-1] / daily_df['sales'].iloc[-8] - 1
                next_row['pct_change_28'] = daily_df['sales'].iloc[-1] / daily_df['sales'].iloc[-29] - 1
            else:
                # Subsequent predictions use previous predictions
                for lag in [1, 2, 3, 7, 14, 21, 28, 30]:
                    if lag <= i:
                        # Use predicted values
                        next_row[f'lag_{lag}'] = predictions[i-lag]
                    else:
                        # Use known values
                        next_row[f'lag_{lag}'] = daily_df['sales'].iloc[-(lag-i)]
                
                # Add rolling statistics - mix of predictions and historical
                for window in [7, 14, 30]:
                    if i >= window:
                        # All predictions
                        next_row[f'rolling_mean_{window}'] = np.mean(predictions[i-window:i])
                        next_row[f'rolling_std_{window}'] = np.std(predictions[i-window:i])
                    else:
                        # Mix of historical and predictions
                        hist_needed = window - i
                        all_vals = list(daily_df['sales'].iloc[-hist_needed:]) + predictions[:i]
                        next_row[f'rolling_mean_{window}'] = np.mean(all_vals)
                        next_row[f'rolling_std_{window}'] = np.std(all_vals)
                
                # Add expanding stats
                all_vals = list(daily_df['sales']) + predictions[:i]
                next_row['expanding_mean'] = np.mean(all_vals)
                next_row['expanding_std'] = np.std(all_vals)
                
                # Add percentage changes
                if i >= 1:
                    next_row['pct_change_1'] = predictions[i-1] / (predictions[i-2] if i >= 2 else daily_df['sales'].iloc[-1]) - 1
                else:
                    next_row['pct_change_1'] = daily_df['sales'].pct_change().iloc[-1]
                
                if i >= 7:
                    next_row['pct_change_7'] = predictions[i-1] / predictions[i-8] - 1
                else:
                    value = predictions[i-1] if i > 0 else daily_df['sales'].iloc[-1]
                    prev_value = daily_df['sales'].iloc[-(8-i)]
                    next_row['pct_change_7'] = value / prev_value - 1
                
                if i >= 28:
                    next_row['pct_change_28'] = predictions[i-1] / predictions[i-29] - 1
                else:
                    value = predictions[i-1] if i > 0 else daily_df['sales'].iloc[-1]
                    prev_value = daily_df['sales'].iloc[-(29-i)]
                    next_row['pct_change_28'] = value / prev_value - 1
            
            # Make prediction
            feature_cols = model_info['daily']['features']
            X_next = next_row[feature_cols]
            pred = model.predict(X_next)[0]
            
            # Store prediction
            predictions.append(pred)
            
            # Add confidence bounds (±10%)
            margin = 0.10
            lower_bounds.append(pred * (1 - margin))
            upper_bounds.append(pred * (1 + margin))
        
        # Format results
        results = {
            'date': [d.strftime('%Y-%m-%d') for d in future_dates],
            'prediction': predictions,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds
        }
        
        return results
    except Exception as e:
        print(f"❌ Error in daily prediction: {e}")
        return "Terjadi kesalahan saat memproses prediksi harian."                  


def predict_weekly():
    """Generate weekly sales prediction for next 4 weeks"""
    try:
        if ml_models['weekly'] is None or 'weekly_stl' not in historical_data:
              print("❌ Error in weekly prediction: model belum tersedia")
              return "Model mingguan belum tersedia."
        
        weekly_df = historical_data['weekly']
        stl_data = historical_data['weekly_stl']
        model = ml_models['weekly']
        
        # Get STL components
        resid = stl_data['resid']
        trend = stl_data['trend']
        seasonal = stl_data['seasonal']

        current_date = datetime.datetime.now().date()
        
        # Extract model input shape from model object
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            # If model expects multiple inputs
            print(f"Debug - Model expects multiple inputs: {input_shape}")
            input_shapes = input_shape
        else:
            # If model expects a single input
            print(f"Debug - Model expects single input: {input_shape}")
            input_shapes = [input_shape]
        
        # Check model input structure
        num_inputs = len(model.inputs)
        print(f"Debug - Model expects {num_inputs} input(s)")
        
        # Get last date
        last_date = weekly_df['date'].iloc[-1]
        
        # Generate prediction dates (weekly, starting from next Monday)
        future_dates = []
        for i in range(4):
            days_until_monday = (7 - current_date.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7  # If today is Monday, go to next Monday
            
            next_date = current_date + datetime.timedelta(days=days_until_monday + (7 * i))
            future_dates.append(next_date)
        
        # Future predictions
        future_resid_preds = []
        
        # Based on model_weekly.txt, we need to adapt to the correct input structure
        # The STT-ATTLSTM model expects 3 inputs:
        # 1. Residual sequence (shape: batch, window_size, 1)
        # 2. Other components (trend+seasonal) sequence (shape: batch, window_size, 2)
        # 3. Last residual value (shape: batch, 1)
        
        # Extract the window size from the model input shape
        window_size = input_shapes[0][1] if num_inputs > 0 else 16
        print(f"Debug - Using window size: {window_size}")
        
        # Prepare sequences with correct shapes
        if num_inputs == 3:
            # Get sequences for residual and other components
            last_resid_seq = stl_data['last_resid_seq']
            last_other_seq = stl_data['last_other_seq']
            last_resid_input = stl_data['last_resid_input']
            
            # Reshape if needed to match expected dimensions
            if last_resid_seq.shape[1] != window_size:
                print(f"Debug - Reshaping residual sequence from {last_resid_seq.shape} to match window size {window_size}")
                # Use the last 'window_size' elements
                last_resid_seq = last_resid_seq[:, -window_size:, :]
                
            if last_other_seq.shape[1] != window_size:
                print(f"Debug - Reshaping other sequence from {last_other_seq.shape} to match window size {window_size}")
                # Use the last 'window_size' elements
                last_other_seq = last_other_seq[:, -window_size:, :]
                
            # Generate predictions
            for i in range(4):
                # Make prediction with triple input
                pred_scaled = model.predict(
                    [last_resid_seq, last_other_seq, last_resid_input],
                    verbose=0
                ).flatten()[0]
                
                # Inverse transform
                pred_resid = scalers['weekly']['resid'].inverse_transform([[pred_scaled]])[0, 0]
                future_resid_preds.append(pred_resid)
                
                # Update sequences for next prediction
                last_resid_seq = np.roll(last_resid_seq, -1, axis=1)
                last_resid_seq[0, -1, 0] = pred_scaled
                
                # Update other sequence (trend + seasonal)
                # Calculate next week's position
                current_week = (len(resid) + i) % 52
                trend_val = trend.values[-1]  # Assume constant trend
                seasonal_val = seasonal.values[current_week]
                
                # Transform and update
                other_vals = np.array([[trend_val, seasonal_val]])
                other_scaled = scalers['weekly']['other'].transform(other_vals)
                last_other_seq = np.roll(last_other_seq, -1, axis=1)
                last_other_seq[0, -1, 0] = other_scaled[0, 0]  # trend
                last_other_seq[0, -1, 1] = other_scaled[0, 1]  # seasonal
                
                # Update last residual input
                last_resid_input = np.array([[pred_scaled]])
                
        elif num_inputs == 1:
            # Model expects a single input with shape (batch, window_size, 3)
            # We need to create a combined input with residual, trend, and seasonal
            
            # Extract sequences and reshape them
            residual_seq = stl_data['last_resid_seq']
            trend_seq = stl_data['last_other_seq'][:, :, 0:1]
            seasonal_seq = stl_data['last_other_seq'][:, :, 1:2]
            
            # Combine into a single sequence with 3 channels
            combined_seq = np.concatenate([residual_seq, trend_seq, seasonal_seq], axis=2)
            
            # Ensure correct window size
            if combined_seq.shape[1] != window_size:
                print(f"Debug - Reshaping combined sequence from {combined_seq.shape} to match window size {window_size}")
                combined_seq = combined_seq[:, -window_size:, :]
            
            # Generate predictions
            for i in range(4):
                # Make prediction with single input
                pred_scaled = model.predict(combined_seq, verbose=0).flatten()[0]
                
                # Inverse transform
                pred_resid = scalers['weekly']['resid'].inverse_transform([[pred_scaled]])[0, 0]
                future_resid_preds.append(pred_resid)
                
                # Update sequence for next prediction
                combined_seq = np.roll(combined_seq, -1, axis=1)
                
                # Calculate next week's trend and seasonal components
                current_week = (len(resid) + i) % 52
                trend_val = trend.values[-1]
                seasonal_val = seasonal.values[current_week]
                
                # Transform and update
                trend_scaled = scalers['weekly']['other'].transform([[trend_val, seasonal_val]])[0, 0]
                seasonal_scaled = scalers['weekly']['other'].transform([[trend_val, seasonal_val]])[0, 1]
                
                # Update combined sequence
                combined_seq[0, -1, 0] = pred_scaled       # residual
                combined_seq[0, -1, 1] = trend_scaled      # trend
                combined_seq[0, -1, 2] = seasonal_scaled   # seasonal
        
        else:
            # Fallback to a simpler approach
            print("Debug - Using fallback approach with sequence reshaping")
            
            # Create a reshaped sequence matching what the model expects
            if hasattr(model, 'input_shape') and model.input_shape is not None:
                expected_shape = model.input_shape
                if isinstance(expected_shape, tuple):
                    # Single input expected
                    batch_size, seq_len, features = 1, expected_shape[1], expected_shape[2]
                    
                    # Create a sequence with zeros of the right shape
                    sequence = np.zeros((batch_size, seq_len, features))
                    
                    # Fill with the residual data we have
                    resid_scaled = scalers['weekly']['resid'].transform(resid.values[-seq_len:].reshape(-1, 1))
                    for i in range(min(seq_len, len(resid_scaled))):
                        sequence[0, i, 0] = resid_scaled[i]
                    
                    # Fill with trend and seasonal data if we have more features
                    if features > 1:
                        trend_seasonal = np.vstack([
                            trend.values[-seq_len:],
                            seasonal.values[-(seq_len % 52):(-(seq_len % 52) + seq_len)]
                        ]).T
                        trend_seasonal_scaled = scalers['weekly']['other'].transform(trend_seasonal)
                        
                        for i in range(min(seq_len, len(trend_seasonal_scaled))):
                            if features >= 2:
                                sequence[0, i, 1] = trend_seasonal_scaled[i, 0]  # trend
                            if features >= 3:
                                sequence[0, i, 2] = trend_seasonal_scaled[i, 1]  # seasonal
                    
                    # Generate predictions
                    for i in range(4):
                        pred_scaled = model.predict(sequence, verbose=0).flatten()[0]
                        pred_resid = scalers['weekly']['resid'].inverse_transform([[pred_scaled]])[0, 0]
                        future_resid_preds.append(pred_resid)
                        
                        # Update sequence for next prediction
                        sequence = np.roll(sequence, -1, axis=1)
                        sequence[0, -1, 0] = pred_scaled
                        
                        # Update other features if we have them
                        if features > 1:
                            current_week = (len(resid) + i) % 52
                            trend_val = trend.values[-1]
                            seasonal_val = seasonal.values[current_week]
                            
                            # Transform and update
                            ts_scaled = scalers['weekly']['other'].transform([[trend_val, seasonal_val]])
                            if features >= 2:
                                sequence[0, -1, 1] = ts_scaled[0, 0]  # trend
                            if features >= 3:
                                sequence[0, -1, 2] = ts_scaled[0, 1]  # seasonal
                else:
                    # Multiple inputs expected (fallback)
                    print("Debug - Complex input shape, falling back to simple prediction")
                    # Simple prediction logic for fallback
                    for i in range(4):
                        # Just estimate based on previous data patterns
                        last_4_resid = resid.values[-4:]
                        pred_resid = np.mean(last_4_resid)  # Simple average forecast
                        future_resid_preds.append(pred_resid)
            else:
                # Very simple fallback
                print("Debug - No input shape detected, using simple average")
                for i in range(4):
                    last_4_resid = resid.values[-4:]
                    pred_resid = np.mean(last_4_resid)  # Simple average forecast
                    future_resid_preds.append(pred_resid)
        
        # Reconstruct predictions (residual + trend + seasonal)
        trend_forecast = [trend.values[-1]] * 4  # Assume constant trend
        seasonal_forecast = [seasonal.values[(len(resid) + i) % 52] for i in range(4)]
        
        # Final predictions
        predictions = np.array(future_resid_preds) + np.array(trend_forecast) + np.array(seasonal_forecast)
        
        # Add confidence bounds (±10%)
        margin = 0.10
        lower_bounds = predictions * (1 - margin)
        upper_bounds = predictions * (1 + margin)
        
        # Format results
        results = {
            'date': [d.strftime('%Y-%m-%d') for d in future_dates],
            'prediction': predictions.tolist(),
            'lower_bound': lower_bounds.tolist(),
            'upper_bound': upper_bounds.tolist()
        }
        
        return results
    except Exception as e:
        print(f"❌ Error in weekly prediction: {str(e)}")
        traceback.print_exc()
        return "Terjadi kesalahan saat memproses prediksi mingguan."

def predict_monthly():
    """Generate monthly sales prediction for next 3 months starting from the next month"""
    try:
        if ml_models['monthly'] is None or 'monthly_stl' not in historical_data:
            print("❌ Error in monthly prediction: model belum tersedia")
            return "Model bulanan belum tersedia."
        
        current_date = datetime.datetime.now().date()
        
        monthly_df = historical_data['monthly']
        stl_data = historical_data['monthly_stl']
        model = ml_models['monthly']
        
        # Get STL components
        trend = stl_data['trend']
        seasonal = stl_data['seasonal']
        resid = stl_data['resid']
        
        # Get last sequence
        last_seq = stl_data['last_seq']
        
        # Get last date
        last_date = monthly_df['date'].iloc[-1]
        
        # Generate prediction dates (next 3 months, starting from next month)
        future_dates = []
        # Mulai dari bulan depan, bukan bulan berjalan
        if current_date.month == 12:
            next_month_date = datetime.date(current_date.year + 1, 1, 1)
        else:
            next_month_date = datetime.date(current_date.year, current_date.month + 1, 1)
        
        for i in range(3):
            # Kalkulasi tahun dan bulan untuk tanggal yang akan datang
            year = next_month_date.year + ((next_month_date.month + i - 1) // 12)
            month = ((next_month_date.month + i - 1) % 12) + 1
            future_date = datetime.date(year, month, 1)
            future_dates.append(future_date)
        
        # Generate predictions
        predictions = []
        current_seq = last_seq.copy()
        
        for i in range(3):
            # Predict next value
            pred_scaled = model.predict(current_seq)[0, 0]
            
            # Inverse transform
            pred_resid = scalers['monthly'].inverse_transform([[pred_scaled]])[0, 0]
            
            # Get trend and seasonal for this month
            # Hitung bulan yang sesuai dengan urutan prediksi
            pred_month = (next_month_date.month + i - 1) % 12 + 1
            
            trend_val = trend.iloc[-1]  # Assume constant trend
            seasonal_idx = pred_month - 1  # Adjust for 0-based indexing in seasonal component
            seasonal_val = seasonal.iloc[seasonal_idx]
            
            # Reconstruct prediction
            prediction = pred_resid + trend_val + seasonal_val
            predictions.append(prediction)
            
            # Update sequence for next prediction
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, 0] = pred_scaled
        
        # Tambahkan confidence bounds (±10%)
        margin = 0.10
        lower_bounds = [pred * (1 - margin) for pred in predictions]
        upper_bounds = [pred * (1 + margin) for pred in predictions]
        
        # Format hasil dalam dictionary
        results = {
            'date': [d.strftime('%Y-%m-%d') for d in future_dates],
            'prediction': [float(p) for p in predictions],
            'lower_bound': [float(lb) for lb in lower_bounds],
            'upper_bound': [float(ub) for ub in upper_bounds]
        }
        
        return results

    except Exception as e:
        print(f"❌ Error in monthly prediction: {e}")
        traceback.print_exc()
        return "Terjadi kesalahan saat memproses prediksi bulanan."
    

@chatbot_bp.route('/',methods=['POST'])
@jwt_required()


def chatbot():
    try:
        # Mapping rule ke threshold masing-masing
        thresholds = {
        'daily_prediction': 0.68,
        'weekly_prediction': 0.70,
        'monthly_prediction': 0.72,
        'greeting': 0.60,
        'thanks': 0.60,
        'fallback': 0.0,
        'out_of_scope': 0.0
        }

        data = request.get_json(force=True)
        if not data or 'message' not in data:
            return jsonify({'error': 'Permintaan harus berupa JSON dengan kunci "message"'}), 400

        user_input = data.get('message', '').strip()
        if not user_input:
            return jsonify({'error': 'Pesan tidak boleh kosong'}), 400

        if user_input.lower() in ['exit', 'keluar', 'quit', 'selesai']:
            return jsonify({'response': 'Terima kasih telah menggunakan PredictoBot. Sampai jumpa!'})

        rule_name, similarity, matched_pattern = find_best_rule_match(user_input)
        threshold = thresholds.get(rule_name, 0.68)  # nilai default jika tidak ada di mapping

        print(f"[Debug] Matched rule: {rule_name}, Similarity: {similarity:.4f}, Pattern: {matched_pattern}")

        if rule_name in ['daily_prediction', 'weekly_prediction', 'monthly_prediction'] and similarity >= threshold:
            timeframe = rule_name.split('_')[0]

            if timeframe == 'daily':
                result = predict_daily()
                if isinstance(result, str):
                    return jsonify({'response': result})
                response_data = [
                    {
                        'date': date,
                        'prediction': int(pred),
                        'margin': int(abs(ub - lb) / 2)
                    }
                    for date, pred, lb, ub in zip(result['date'], result['prediction'], result['lower_bound'], result['upper_bound'])
                ]
                return jsonify({'type': 'daily', 'response': response_data})

            elif timeframe == 'weekly':
                result = predict_weekly()
                if isinstance(result, str):
                    return jsonify({'response': result})
                response_data = [
                    {
                        'week': date,
                        'prediction': int(pred),
                        'margin': int(abs(ub - lb) / 2)
                    }
                    for date, pred, lb, ub in zip(result['date'], result['prediction'], result['lower_bound'], result['upper_bound'])
                ]
                return jsonify({'type': 'weekly', 'response': response_data})

            elif timeframe == 'monthly':
                result = predict_monthly()
                if isinstance(result, str):
                    return jsonify({'response': result})
                response_data = [
                    {
                        'month': date,
                        'prediction': int(pred),
                        'margin': int(abs(ub - lb) / 2)
                    }
                    for date, pred, lb, ub in zip(result['date'], result['prediction'], result['lower_bound'], result['upper_bound'])
                ]
                return jsonify({'type': 'monthly', 'response': response_data})

        elif rule_name == 'out_of_scope' and similarity >= threshold:
            response = random.choice([
                "Maaf, pertanyaan Anda sepertinya di luar cakupan bisnis prediksi penjualan iPhone kami.",
                "Saya khusus membantu dengan prediksi penjualan iPhone. Pertanyaan Anda tidak berkaitan dengan hal tersebut.",
                "Sepertinya pertanyaan Anda tidak terkait dengan prediksi penjualan iPhone. Silahkan ajukan pertanyaan yang sesuai."
            ])
            return jsonify({'response': response})

        elif rule_name and similarity >= threshold:
            response = random.choice(rules_data[rule_name]['responses'])
            return jsonify({'response': response})

        else:
            fallback = random.choice(rules_data['fallback']['responses'])
            return jsonify({'response': fallback})

    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500
