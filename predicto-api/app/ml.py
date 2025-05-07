from flask import Blueprint, request, jsonify, send_file
from flask_jwt_extended import jwt_required
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime, timedelta
import holidays
from xgboost import XGBRegressor
import joblib 
import os
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras._tf_keras.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path_daily = os.path.join(BASE_DIR, 'model_ml', 'xgboost_daily_model.pkl')
model_path_weakly = os.path.join(BASE_DIR, 'model_ml', 'model_stt_attlstm_weekly.keras')

ml_bp = Blueprint('ml', __name__)

@ml_bp.route('/predict',methods=['POST'])
@jwt_required()

def predict():
    if 'file' not in request.files:
        return jsonify({"message": "CSV file is required"}), 400
    if 'frequency' not in request.form:
        return jsonify({"message": "Frequency is required"}), 400
    
    csv_data = request.files['file']
    freq_data = request.form['frequency']

     # Validasi frekuensi
    if freq_data not in ['D', 'W', 'ME','d','w','me']:
        return jsonify({"message": "Invalid frequency, choose 'D', 'W', or 'ME'"}), 400

    

    def to_millions(x):
    # Handle different types of input
        if isinstance(x, list):
            return np.array(x) / 1000000
        else:
            return x / 1000000
    


    def check_stationarity(time_series, title='', signif=0.05):
        # Ensure time_series is appropriate length
        if len(time_series) < 10:
            print(f"Warning: Time series for {title} is too short for reliable stationarity testing.")
            return False

        # Augmented Dickey-Fuller test
        result_adf = adfuller(time_series, autolag='AIC')
        adf_stat, adf_pvalue = result_adf[0], result_adf[1]

        # KPSS test
        result_kpss = kpss(time_series, regression='c', nlags="auto")
        kpss_stat, kpss_pvalue = result_kpss[0], result_kpss[1]

        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        # Plot time series
        ax1.plot(time_series)
        ax1.set_title(f'Time Series {title}')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')

        # Calculate appropriate number of lags (not exceeding 50% of data length as required by statsmodels)
        max_lags = min(40, int(len(time_series) * 0.4))
        if max_lags < 1:
            max_lags = 1

        print(f"Using {max_lags} lags for ACF/PACF calculation (data length: {len(time_series)})")

        # Plot ACF with appropriate lags
        plot_acf(time_series, ax=ax2, lags=max_lags)
        ax2.set_title('Autocorrelation Function')

        # Plot PACF with appropriate lags
        plot_pacf(time_series, ax=ax3, lags=max_lags)
        ax3.set_title('Partial Autocorrelation Function')

        # Print results
        print(f"\nStationarity Test Results for {title}:")
        print(f"ADF Statistic: {adf_stat:.4f}, p-value: {adf_pvalue:.4f}")
        print(f"KPSS Statistic: {kpss_stat:.4f}, p-value: {kpss_pvalue:.4f}")

        # Interpretation
        is_stationary = (adf_pvalue < signif) and (kpss_pvalue >= signif)
        print(f"Series is {'stationary' if is_stationary else 'non-stationary'}")
        print(f"ADF Test: {'Rejects' if adf_pvalue < signif else 'Does not reject'} null hypothesis of non-stationarity")
        print(f"KPSS Test: {'Rejects' if kpss_pvalue < signif else 'Does not reject'} null hypothesis of stationarity")

        return is_stationary

    def add_features(df):
        df_feat = df.copy()

        # Date components
        df_feat['year'] = df_feat['date'].dt.year
        df_feat['month'] = df_feat['date'].dt.month
        df_feat['day'] = df_feat['date'].dt.day
        df_feat['dayofweek'] = df_feat['date'].dt.dayofweek
        df_feat['quarter'] = df_feat['date'].dt.quarter
        df_feat['is_month_start'] = df_feat['date'].dt.is_month_start.astype(int)
        df_feat['is_month_end'] = df_feat['date'].dt.is_month_end.astype(int)
        df_feat['is_quarter_start'] = df_feat['date'].dt.is_quarter_start.astype(int)
        df_feat['is_quarter_end'] = df_feat['date'].dt.is_quarter_end.astype(int)
        df_feat['is_year_start'] = df_feat['date'].dt.is_year_start.astype(int)
        df_feat['is_year_end'] = df_feat['date'].dt.is_year_end.astype(int)

        # Day of the year and week of the year
        df_feat['dayofyear'] = df_feat['date'].dt.dayofyear
        df_feat['weekofyear'] = df_feat['date'].dt.isocalendar().week

        # Weekend indicator
        df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)

        # Holidays (Indonesian holidays)
        indo_holidays = holidays.Indonesia()
        df_feat['is_holiday'] = df_feat['date'].apply(lambda x: x in indo_holidays).astype(int)

        # Create seasonal features using sine and cosine transforms
        # These capture cyclical patterns better than categorical variables

        # Yearly seasonality
        days_in_year = 365.25
        df_feat['year_sin'] = np.sin(2 * np.pi * df_feat['dayofyear'] / days_in_year)
        df_feat['year_cos'] = np.cos(2 * np.pi * df_feat['dayofyear'] / days_in_year)

        # Monthly seasonality
        days_in_month = 30.44
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['day'] / days_in_month)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['day'] / days_in_month)

        # Weekly seasonality
        days_in_week = 7
        df_feat['week_sin'] = np.sin(2 * np.pi * df_feat['dayofweek'] / days_in_week)
        df_feat['week_cos'] = np.cos(2 * np.pi * df_feat['dayofweek'] / days_in_week)

        return df_feat

    def remove_outliers(df, column='sales', lower_quantile=0.05, upper_quantile=0.95, iqr_multiplier=1.5):

        Q1 = df[column].quantile(lower_quantile)
        Q3 = df[column].quantile(upper_quantile)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        print(f"Removing outliers from {column}:")
        print(f"Lower bound: {to_millions(lower_bound):.2f} million")
        print(f"Upper bound: {to_millions(upper_bound):.2f} million")
        print(f"Original shape: {df.shape[0]}")

        df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        print(f"Shape after removing outliers: {df_clean.shape[0]}")
        print(f"Removed {df.shape[0] - df_clean.shape[0]} rows ({(df.shape[0] - df_clean.shape[0])/df.shape[0]*100:.2f}%)")

        return df_clean

    def add_lag_features(df, target_col='sales', lags=[1, 2, 3, 7, 14, 21, 28, 30]):

        df_lag = df.copy()

        # Add lag features
        for lag in lags:
            df_lag[f'lag_{lag}'] = df_lag[target_col].shift(lag)

        # Add rolling mean/std features
        for window in [7, 14, 30]:
            df_lag[f'rolling_mean_{window}'] = df_lag[target_col].rolling(window=window).mean()
            df_lag[f'rolling_std_{window}'] = df_lag[target_col].rolling(window=window).std()

        # Add expanding mean/std features
        df_lag['expanding_mean'] = df_lag[target_col].expanding().mean()
        df_lag['expanding_std'] = df_lag[target_col].expanding().std()

        # Add growth rate features (percent change)
        df_lag['pct_change_1'] = df_lag[target_col].pct_change(periods=1)
        df_lag['pct_change_7'] = df_lag[target_col].pct_change(periods=7)
        df_lag['pct_change_28'] = df_lag[target_col].pct_change(periods=28)

        # Drop rows with NaN values from lag features
        # Instead of dropping, we could also impute them
        df_lag = df_lag.dropna()

        return df_lag
    
    def preprocess_and_analyze_sales(
        csv_data,
        date_col='transaction_date',
        target_col='total',
        freq=freq_data
    ):
        try:
            print("Memuat dataset...")
            df = pd.read_csv(csv_data)

        except FileNotFoundError:
            print(f"Error: File '{csv_data}' tidak ditemukan.")
            return None

        except Exception as e:
            print(f"Terjadi kesalahan saat memuat file: {e}")
            return None

        try:
            print("Memproses data...")
            print(f"Dataset mentah memiliki {df.shape[0]} baris dan {df.shape[1]} kolom")
            print("Mengecek data yang hilang:")
            print(df.isnull().sum().sum(), "nilai yang hilang")
            print("\nInformasi kolom:")
            print(df.dtypes)

            if date_col not in df.columns or target_col not in df.columns:
                raise ValueError(f"Kolom '{date_col}' atau '{target_col}' tidak ditemukan dalam dataset.")

            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            if df[date_col].isnull().any():
                print("Warning: Beberapa nilai dalam kolom tanggal tidak valid dan telah diubah menjadi NaT.")

            df = df.sort_values(by=date_col)

            print("\nMembuat agregasi data...")
            freq_label = {'D': 'harian', 'W': 'mingguan', 'ME': 'bulanan'}
            if freq.upper() == 'ME':
                df_agg = df.groupby(pd.Grouper(key=date_col, freq='ME')).agg({target_col: 'sum'}).reset_index()
            elif freq.upper() == 'W':
                df_agg = df.groupby(pd.Grouper(key=date_col, freq='W')).agg({target_col: 'sum'}).reset_index()
            elif freq.upper() == 'D':
                df_agg = df.groupby(pd.Grouper(key=date_col, freq='D')).agg({target_col: 'sum'}).reset_index()

            df_agg.rename(columns={date_col: 'date', target_col: 'sales'}, inplace=True)
            print(f"\n============ ANALISIS SKALA {freq_label.get(freq.upper(), freq)} ============")

        except ValueError as ve:
            print(f"Error: {ve}")
            return None
        except Exception as e:
            print(f"Terjadi kesalahan saat memproses data: {e}")
            return None

        try:
            # Hapus outlier
            df_clean = remove_outliers(df_agg)
            print("\nDeskripsi Data (dalam jutaan):")
            print(to_millions(df_clean['sales']).describe())

            # ======== NEW: Check stationarity and apply differencing if needed ========
            print("\nMenguji stasioneritas data...")
            is_stationary = check_stationarity(df_clean['sales'], title=f"Frekuensi: {freq_label.get(freq.upper(), freq)}")

            diff_order = 0
            sales_series = df_clean['sales'].copy()

            if not is_stationary:
                print("Data tidak stasioner, melakukan differencing orde pertama...")
                diff_order = 1
                sales_series = sales_series.diff().dropna()
                is_stationary = check_stationarity(sales_series.values, title=f"Frekuensi: {freq_label.get(freq.upper(), freq)} (Diff 1)")

                if not is_stationary:
                    print("Data masih tidak stasioner, melakukan differencing orde kedua...")
                    diff_order = 2
                    sales_series = sales_series.diff().dropna()
                    is_stationary = check_stationarity(sales_series.values, title=f"Frekuensi: {freq_label.get(freq.upper(), freq)} (Diff 2)")

                    if not is_stationary:
                        print("Data tetap tidak stasioner setelah dua kali differencing. Menggunakan data asli.")
                        sales_series = df_clean['sales']
                        diff_order = 0
            else:
                print("Data sudah stasioner, tidak perlu differencing.")

            print(f"Orde differencing yang digunakan: {diff_order}")

            # Karena differencing menghilangkan baris awal, sesuaikan indeks dataframe
            df_clean = df_clean.iloc[-len(sales_series):].copy()
            df_clean['sales'] = sales_series.values  # ganti kolom 'sales' dengan hasil differencing

            # Tambahkan fitur setelah data stasioner
            df_features = add_features(df_clean)
            df_features = add_lag_features(df_features)

            return df_features  # untuk pelatihan model selanjutnya

        except Exception as e:
            print(f"Terjadi kesalahan dalam analisis data atau fitur: {e}")
            return None

        

    df_result = preprocess_and_analyze_sales(
        csv_data,
        date_col='transaction_date',
        target_col='total',
        freq=freq_data
    )

    if freq_data.upper()=='D':
            try:
                xgb_model = joblib.load(model_path_daily)
                feature_cols = [col for col in df_result if col not in ['date', 'sales']]

                # Initialize the future predictions list
                future_pred_xgb = []

                # Forecast future values
                future_steps = 30

                # Generate future dates
                last_date = df_result['date'].iloc[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(future_steps)]
                last_known_values = df_result.iloc[-30:].copy()  # Get last month of data

                # Create a dataframe for future dates
                conf_level = 0.90
                margin = (1 - conf_level) / 2
                future_dates_df = pd.DataFrame({'date': future_dates})
                future_df = add_features(future_dates_df)  # Add date features

                # Initialize with known values
                future_df_with_values = future_df.copy()

                # Assuming xgb_model is your trained XGBoost model and feature_cols is your list of features
                for i in range(future_steps):
                    # Get the latest available data (either original or predicted)
                    if i == 0:
                        latest_data = last_known_values.iloc[-30:].copy()  # Start with the last 30 days of data
                    else:
                        # Update with new predictions
                        latest_data = pd.concat([
                            latest_data.iloc[1:],  # Remove the oldest day (shift)
                            future_rows.iloc[i-1:i][['sales']]  # Add the newest prediction
                        ])

                    # Create row for next date with date features
                    next_row = future_df.iloc[i:i+1].copy()

                    # Add lag features based on available data
                    for lag in [1, 2, 3, 7, 14, 21, 28, 30]:
                        if lag <= i:
                            # Use predicted values
                            lag_idx = i - lag
                            next_row[f'lag_{lag}'] = future_pred_xgb[lag_idx]
                        else:
                            # Use known values from historical data
                            lag_idx = -(lag - i)
                            if abs(lag_idx) <= len(latest_data):
                                next_row[f'lag_{lag}'] = latest_data['sales'].iloc[lag_idx]
                            else:
                                next_row[f'lag_{lag}'] = latest_data['sales'].iloc[0]

                    # Add rolling statistics
                    for window in [7, 14, 30]:
                        if i >= window - 1:
                            # We have enough predictions to calculate rolling window
                            window_data = future_pred_xgb[max(0, i-window+1):i] + [latest_data['sales'].iloc[-1]]
                            next_row[f'rolling_mean_{window}'] = np.mean(window_data)
                            next_row[f'rolling_std_{window}'] = np.std(window_data) if len(window_data) > 1 else 0
                        else:
                            # Not enough predictions, use available data
                            available_preds = future_pred_xgb[:i] if i > 0 else []
                            needed_hist = window - len(available_preds)
                            hist_data = latest_data['sales'].iloc[-needed_hist:].values if needed_hist > 0 else []
                            window_data = np.concatenate([hist_data, available_preds])
                            next_row[f'rolling_mean_{window}'] = np.mean(window_data)
                            next_row[f'rolling_std_{window}'] = np.std(window_data) if len(window_data) > 1 else 0

                    # Add expanding stats (simplified)
                    next_row['expanding_mean'] = np.mean(latest_data['sales'])
                    next_row['expanding_std'] = np.std(latest_data['sales'])

                    # Add percentage change features
                    if i == 0:
                        next_row['pct_change_1'] = latest_data['sales'].pct_change(periods=1).iloc[-1]
                        next_row['pct_change_7'] = latest_data['sales'].pct_change(periods=7).iloc[-1]
                        next_row['pct_change_28'] = latest_data['sales'].pct_change(periods=28).iloc[-1]
                    else:
                        prev_value = future_pred_xgb[i-1]
                        next_row['pct_change_1'] = (prev_value / latest_data['sales'].iloc[-1]) - 1 if latest_data['sales'].iloc[-1] != 0 else 0

                        if i >= 7:
                            prev7_value = future_pred_xgb[i-7]
                            next_row['pct_change_7'] = (prev_value / prev7_value) - 1 if prev7_value != 0 else 0
                        else:
                            next_row['pct_change_7'] = latest_data['pct_change_7'].iloc[-1]

                        if i >= 28:
                            prev28_value = future_pred_xgb[i-28]
                            next_row['pct_change_28'] = (prev_value / prev28_value) - 1 if prev28_value != 0 else 0
                        else:
                            next_row['pct_change_28'] = latest_data['pct_change_28'].iloc[-1]

                    # Prepare the feature vector for the prediction
                    X_next = next_row[feature_cols]
                    
                    # Predict the next value
                    pred = xgb_model.predict(X_next.values.reshape(1, -1))[0]
                    
                    # Append the prediction to future predictions list
                    future_pred_xgb.append(pred)

                    # Store the row with prediction for future iterations
                    next_row['sales'] = pred
                    if i == 0:
                        future_rows = next_row
                    else:
                        future_rows = pd.concat([future_rows, next_row])

                # The future predictions are now stored in `future_pred_xgb` and `future_rows`

                # Create prediction intervals (90%)
                lower_bound_xgb = [pred * (1 - margin) for pred in future_pred_xgb]
                upper_bound_xgb = [pred * (1 + margin) for pred in future_pred_xgb]

                # XGBoost predictions in desired format
                xgb_results = pd.DataFrame({
                    'Tanggal': [date.strftime('%Y-%m-%d') for date in future_dates],
                    'Prediksi Penjualan (Rp)': [int(round(pred)) for pred in future_pred_xgb],
                    'Lower Bound (Rp)': [int(round(lb)) for lb in lower_bound_xgb],
                    'Upper Bound (Rp)': [int(round(ub)) for ub in upper_bound_xgb]
                })

                print("\nPrediksi XGBoost 7 Hari Ke Depan:")
                print(xgb_results.head(7))

                result_json=xgb_results.to_dict(orient='records')

                return jsonify({
                    "message": "✅ Prediksi berhasil",
                    "data": result_json
                }), 200
            
            except Exception as e:

                traceback.print_exc()
        

                return jsonify({
                "message": "❌ Terjadi kesalahan saat melakukan prediksi",
                "error": str(e)
                }), 500
            
    elif freq_data.upper() == 'W':
            try:
                # === 1. Decompose ===
                stl = STL(df_result['sales'], period=52)
                res = stl.fit()
                resid = res.resid
                trend = res.trend
                seasonal = res.seasonal
                future_steps_weekly = 12

                # === 2. Scaling ===
                scaler_resid = StandardScaler()
                scaler_other = MinMaxScaler()
                resid_scaled = scaler_resid.fit_transform(resid.values.reshape(-1, 1)).flatten()
                other_scaled = scaler_other.fit_transform(np.vstack([trend, seasonal]).T)

                # === 3. Sequence Building ===
                def create_dual_sequences(resid, other, window_size):
                    X_resid, X_other, y = [], [], []
                    for i in range(len(resid) - window_size):
                        X_resid.append(resid[i:i+window_size])
                        X_other.append(other[i:i+window_size])
                        y.append(resid[i+window_size])
                    return np.expand_dims(np.array(X_resid), 2), np.array(X_other), np.array(y)

                window_size = 16
                X1, X2, y = create_dual_sequences(resid_scaled, other_scaled, window_size)
                X1_test = X1.astype(np.float32)
                X2_test = X2.astype(np.float32)
                resid_input_test = y.astype(np.float32).reshape(-1, 1)

                # === 4. Load Model ===
                model = load_model(model_path_weakly)

                # === 5. Forecasting ===
                future_resid_preds = []
                current_X1 = X1_test[-1].copy()
                current_X2 = X2_test[-1].copy()
                current_resid = resid_input_test[-1].copy()

                for i in range(future_steps_weekly):
                    pred_scaled = model.predict([
                        np.expand_dims(current_X1, 0),
                        np.expand_dims(current_X2, 0),
                        np.expand_dims(current_resid, 0)
                    ], verbose=0).flatten()[0]

                    pred_resid = scaler_resid.inverse_transform([[pred_scaled]])[0, 0]
                    future_resid_preds.append(pred_resid)

                    current_X1 = np.roll(current_X1, -1)
                    current_X1[-1] = pred_scaled

                    current_week = (len(resid) + i) % 52
                    trend_val = trend.values[-1]
                    seasonal_val = seasonal.values[current_week]

                    current_X2 = np.roll(current_X2, -1, axis=0)
                    current_X2[-1] = scaler_other.transform([[trend_val, seasonal_val]])[0]
                    current_resid = np.array([pred_scaled])

                # === 6. Reconstruct Forecast ===
                last_date_weekly = pd.to_datetime(df_result['date']).iloc[-1]
                future_dates_weekly = [last_date_weekly + timedelta(weeks=i+1) for i in range(future_steps_weekly)]
                trend_forecast = [trend.values[-1]] * future_steps_weekly
                seasonal_forecast = [seasonal.values[(len(resid) + i) % 52] for i in range(future_steps_weekly)]

                future_forecast_attlstm = (
                    np.array(future_resid_preds)
                    + np.array(trend_forecast)
                    + np.array(seasonal_forecast)
                )

                margin = 0.10
                lower_bound_attlstm = future_forecast_attlstm * (1 - margin)
                upper_bound_attlstm = future_forecast_attlstm * (1 + margin)

                # === 7. Buat Output ke Dictionary ===
                forecast_dict = {
                    'status': 'success',
                    'forecast': []
                }

                for i in range(future_steps_weekly):
                    forecast_dict['forecast'].append({
                        'tanggal': future_dates_weekly[i].strftime('%Y-%m-%d'),
                        'Prediksi Penjualan (Rp)': int(round(future_forecast_attlstm[i])),
                        'Lower Bound (Rp)': int(round(lower_bound_attlstm[i])),
                        'upper_bound (Rp)': int(round(upper_bound_attlstm[i]))
                    })

                # Format hasil ke DataFrame
                attlstm_results = pd.DataFrame(forecast_dict['forecast'])

                # Print hasil prediksi mingguan
                print("\nPrediksi ATT-LSTM 4 Minggu Ke Depan:")
                print(attlstm_results.head(4))

                result_json = attlstm_results.to_dict(orient='records')

                return jsonify({
                    "message": "✅ Prediksi berhasil",
                    "data": result_json
                }), 200

            except Exception as e:
                traceback_str = traceback.format_exc()
                print(f"❌ Terjadi error:\n{traceback_str}")
                return jsonify({
                    "message": "❌ Terjadi error saat melakukan prediksi.",
                    "error": str(e)
                }), 500
