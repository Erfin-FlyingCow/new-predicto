from flask import Blueprint, request, jsonify, send_file
from flask_jwt_extended import jwt_required
import pandas as pd
import numpy as np
import traceback
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import datetime
import tensorflow as tf
import os
import pickle
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras._tf_keras.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
historical_data_path=os.path.join(BASE_DIR,'ml_data','customer_segmentation_result .csv')
daily_model=os.path.join(BASE_DIR,'ml_data','model_xgb_daily.pkl')
weekly_model=os.path.join(BASE_DIR,'ml_data','stl_attlstm_model_weekly(0.94).h5')
monthly_model=os.path.join(BASE_DIR,'ml_data','bilstm_monthly_model.h5')

ml_bp = Blueprint('ml', __name__)


ml_models = {
    'daily': None,
    'weekly': None,
    'monthly': None
}

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
        monthly_df = daily_df.resample('ME', on='date').sum().reset_index()

        # Simpan ke struktur chatbot
        historical_data['daily'] = daily_df
        historical_data['weekly'] = weekly_df
        historical_data['monthly'] = monthly_df

        print("✅ Data historis dari customer_segmentation_result.csv berhasil dimuat.")
        return True

    except Exception as e:
        print(f"❌ Gagal memuat data historis: {e}")
        return False


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
        
        nested_results = []

        for i in range(len(results['date'])):
            data_point = {
                'date': results['date'][i],
                'prediction': float(results['prediction'][i]),
                'lower_bound': float(results['lower_bound'][i]),
                'upper_bound': float(results['upper_bound'][i])
            }
            nested_results.append(data_point)
        
        return nested_results
    

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
            'prediction': predictions,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds
        }
        nested_results = []

        for i in range(len(results['date'])):
            data_point = {
                'date': results['date'][i],
                'prediction': results['prediction'][i],
                'lower_bound': results['lower_bound'][i],
                'upper_bound': results['upper_bound'][i]
            }
            nested_results.append(data_point)
        
        return nested_results
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

        nested_results = []

        for i in range(len(results['date'])):
            data_point = {
                'date': results['date'][i],
                'prediction': results['prediction'][i],
                'lower_bound': results['lower_bound'][i],
                'upper_bound': results['upper_bound'][i]
            }
            nested_results.append(data_point)
        
        return nested_results

    except Exception as e:
        print(f"❌ Error in monthly prediction: {e}")
        traceback.print_exc()
        return "Terjadi kesalahan saat memproses prediksi bulanan ."

@ml_bp.route('/predict',methods=['POST'])
@jwt_required()

def predict():

    data = request.get_json()

    valid_frequencies = {'d', 'D', 'w', 'W', 'ME', 'me'}

    frequency = data.get('frequency')
    if not frequency or frequency not in valid_frequencies:
        return jsonify({"error": "Valid frequency is required! (d, D, w, W, ME, me)"}), 400
        
        
    if frequency.upper() == 'D' :
        hasil_prediksi = predict_daily()
    elif frequency.upper()=='W':
        hasil_prediksi = predict_weekly()
    elif frequency.upper()=='ME':
        hasil_prediksi = predict_monthly()


    return jsonify({
        'status': 'success',
        'data': hasil_prediksi
    }), 200