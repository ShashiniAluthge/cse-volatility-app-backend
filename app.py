from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import shap
import yfinance as yf
import os
import requests
import warnings
warnings.filterwarnings('ignore')

app  = Flask(__name__)
CORS(app)

# -------------------------------------------------------
# GOOGLE DRIVE FILE IDs — paste yours here
# -------------------------------------------------------
DRIVE_FILES = {
    "gradient_boosting_model.pkl": os.getenv("GDRIVE_MODEL_ID"),
    "scaler.pkl"                 : os.getenv("GDRIVE_SCALER_ID"),
    "feature_cols.pkl"           : os.getenv("GDRIVE_FEATURES_ID"),
    "model_metrics.csv"          : os.getenv("GDRIVE_METRICS_ID"),
    "feature_importances.csv"    : os.getenv("GDRIVE_IMPORTANCES_ID"),
}

MODEL_DIR   = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------
# AUTO DOWNLOAD FROM GOOGLE DRIVE
# -------------------------------------------------------
def download_from_drive(file_id, dest_path):
    print(f"  Downloading {os.path.basename(dest_path)}...")
    url      = f"https://drive.google.com/uc?export=download&id={file_id}"
    session  = requests.Session()
    response = session.get(url, stream=True)

    # Handle Google Drive large file warning
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        response = session.get(url, params={'confirm': token}, stream=True)

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f"  ✅ {os.path.basename(dest_path)} downloaded!")

def ensure_files():
    for filename, file_id in DRIVE_FILES.items():
        if filename.endswith('.csv'):
            dest = os.path.join(RESULTS_DIR, filename)
        else:
            dest = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(dest):
            print(f"  File missing: {filename} — downloading from Drive...")
            download_from_drive(file_id, dest)
        else:
            print(f"  ✅ {filename} already exists locally")

print("🔄 Checking model files...")
ensure_files()

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
print("\n📦 Loading model...")
model        = joblib.load(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
explainer    = shap.TreeExplainer(model)
print(f"✅ Model loaded | Features: {len(FEATURE_COLS)}")

# -------------------------------------------------------
# TECHNICAL INDICATOR FUNCTIONS
# -------------------------------------------------------
def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def compute_bollinger(series, period=20):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + 2*std
    lower = sma - 2*std
    return upper, lower, (upper - lower) / (sma + 1e-9)

def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low  - close.shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_obv(close, volume):
    return (volume * np.sign(close.diff()).fillna(0)).cumsum()

def build_features(hist_df):
    df     = hist_df.copy().reset_index(drop=True)
    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    volume = df['Volume']
    open_  = df['Open']

    for w in [5, 10, 20, 50]:
        df[f'MA_{w}']  = close.rolling(w).mean()
        df[f'EMA_{w}'] = close.ewm(span=w, adjust=False).mean()

    df['Daily_Return']     = close.pct_change()
    df['HL_Pct']           = (high - low) / (close + 1e-9)
    df['OC_Pct']           = (close - open_) / (open_ + 1e-9)
    df['Price_MA5_Ratio']  = close / (df['MA_5']  + 1e-9)
    df['Price_MA20_Ratio'] = close / (df['MA_20'] + 1e-9)
    df['Price_MA50_Ratio'] = close / (df['MA_50'] + 1e-9)
    df['MA5_MA20_Cross']   = df['MA_5']  - df['MA_20']
    df['MA10_MA50_Cross']  = df['MA_10'] - df['MA_50']

    df['RSI_7']  = compute_rsi(close, 7)
    df['RSI_14'] = compute_rsi(close, 14)
    df['RSI_21'] = compute_rsi(close, 21)

    for p in [3, 5, 10, 20]:
        df[f'ROC_{p}']      = close.pct_change(p) * 100
        df[f'Momentum_{p}'] = close - close.shift(p)

    macd, sig, hist      = compute_macd(close)
    df['MACD']           = macd
    df['MACD_Signal']    = sig
    df['MACD_Hist']      = hist

    bb_u, bb_l, bb_bw    = compute_bollinger(close, 20)
    df['BB_Upper']        = bb_u
    df['BB_Lower']        = bb_l
    df['BB_Bandwidth']    = bb_bw
    df['BB_Position']     = (close - bb_l) / ((bb_u - bb_l) + 1e-9)

    df['Volatility_5']   = close.pct_change().rolling(5).std()
    df['Volatility_10']  = close.pct_change().rolling(10).std()
    df['Volatility_20']  = close.pct_change().rolling(20).std()
    df['ATR_14']         = compute_atr(high, low, close, 14)
    df['ATR_Ratio']      = df['ATR_14'] / (close + 1e-9)

    df['Volume_MA5']     = volume.rolling(5).mean()
    df['Volume_MA10']    = volume.rolling(10).mean()
    df['Volume_Ratio']   = volume / (df['Volume_MA5'] + 1e-9)
    df['OBV']            = compute_obv(close, volume)

    for lag in [1, 2, 3, 5]:
        df[f'Return_Lag_{lag}'] = close.pct_change().shift(lag)
        df[f'Close_Lag_{lag}']  = close.shift(lag)

    df['Open']   = open_
    df['High']   = high
    df['Low']    = low
    df['Close']  = close
    df['Volume'] = volume

    return df[FEATURE_COLS].iloc[-1]

# -------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'  : 'ok',
        'model'   : 'GradientBoostingClassifier',
        'features': len(FEATURE_COLS)
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        metrics_df = pd.read_csv(
            os.path.join(RESULTS_DIR, 'model_metrics.csv'))
        feats_df   = pd.read_csv(
            os.path.join(RESULTS_DIR, 'feature_importances.csv')).head(10)

        metrics = {}
        for _, row in metrics_df.iterrows():
            metrics[row['Metric']] = {
                'validation': round(float(row['Validation']), 4),
                'test'      : round(float(row['Test']),       4)
            }

        features = [
            {'feature'   : row['Feature'],
             'importance': round(float(row['Importance']), 4)}
            for _, row in feats_df.iterrows()
        ]

        params = model.get_params()
        best_params = {
            'n_estimators'    : params.get('n_estimators'),
            'learning_rate'   : params.get('learning_rate'),
            'max_depth'       : params.get('max_depth'),
            'min_samples_leaf': params.get('min_samples_leaf'),
            'subsample'       : params.get('subsample'),
            'max_features'    : params.get('max_features'),
        }

        return jsonify({
            'metrics'            : metrics,
            'features'           : features,
            'best_params'        : best_params,
            'n_features'         : len(FEATURE_COLS),
            'n_estimators_actual': len(model.estimators_)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<ticker>', methods=['GET'])
def predict_ticker(ticker):
    try:
        print(f"Fetching live data for {ticker}...")
        df = yf.download(
            ticker, period='120d', interval='1d',
            progress=False, auto_adjust=True
        )

        if df is None or len(df) < 60:
            return jsonify({'error': f'Not enough data for {ticker}'}), 400

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        df = df[['Date','Open','High','Low','Close','Volume']].copy()
        for col in ['Open','High','Low','Close','Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        features_row = build_features(df)

        if features_row.isnull().any():
            return jsonify({'error': 'NaN in computed features'}), 400

        X             = scaler.transform([features_row.values])
        pred          = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        confidence    = float(probabilities[pred]) * 100

        sv      = explainer.shap_values(X)
        sv_flat = sv[0] if len(np.array(sv).shape) > 1 else sv

        shap_list = sorted([
            {
                'feature'   : feat,
                'shap_value': round(float(sv_flat[i]), 5),
                'importance': round(float(model.feature_importances_[i]), 5)
            }
            for i, feat in enumerate(FEATURE_COLS)
        ], key=lambda x: abs(x['shap_value']), reverse=True)

        recent = df.tail(30)
        price_history = [
            {
                'date'  : str(row['Date'].date()),
                'close' : round(float(row['Close']), 2),
                'high'  : round(float(row['High']),  2),
                'low'   : round(float(row['Low']),   2),
                'volume': int(row['Volume'])
            }
            for _, row in recent.iterrows()
        ]

        latest = df.iloc[-1]
        return jsonify({
            'ticker'        : ticker,
            'latest_date'   : str(latest['Date'].date()),
            'latest_close'  : round(float(latest['Close']),  2),
            'latest_open'   : round(float(latest['Open']),   2),
            'latest_high'   : round(float(latest['High']),   2),
            'latest_low'    : round(float(latest['Low']),    2),
            'latest_volume' : int(latest['Volume']),
            'data_rows'     : len(df),
            'prediction'    : pred,
            'volatility'    : 'HIGH' if pred == 1 else 'LOW',
            'confidence'    : round(confidence, 2),
            'probabilities' : {
                'LOW' : round(float(probabilities[0]) * 100, 2),
                'HIGH': round(float(probabilities[1]) * 100, 2)
            },
            'top_features'  : shap_list[:10],
            'price_history' : price_history,
            'interpretation': (
                'High market volatility expected. Prices may fluctuate significantly.'
                if pred == 1 else
                'Low market volatility expected. Prices likely to remain stable.'
            )
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)