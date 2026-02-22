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
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

app = Flask(__name__)
CORS(app)


# GOOGLE DRIVE FILE IDs — loaded from environment variables

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

# Global model variables
model        = None
scaler       = None
FEATURE_COLS = None
explainer    = None


# AUTO DOWNLOAD FROM GOOGLE DRIVE

def download_from_drive(file_id, dest_path):
    print(f"  Downloading {os.path.basename(dest_path)}...")

    url     = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()

    # First request
    response = session.get(url, stream=True)

    # Handle Google's virus scan warning for large files
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        response = session.get(
            url,
            params ={'confirm': token},
            stream  = True
        )

    # Write file
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    size_kb = os.path.getsize(dest_path) / 1024
    print(f"{os.path.basename(dest_path)} downloaded! ({size_kb:.1f} KB)")


def ensure_files():
    """Download any missing model files from Google Drive"""
    all_ok = True
    for filename, file_id in DRIVE_FILES.items():

        # Determine destination folder
        dest = os.path.join(
            RESULTS_DIR if filename.endswith('.csv') else MODEL_DIR,
            filename
        )

        if os.path.exists(dest):
            print(f"{filename} already exists")
        else:
            if not file_id:
                print(f"ERROR: No Drive ID set for {filename}")
                print(f"     Set environment variable for this file")
                all_ok = False
                continue
            try:
                print(f"{filename} missing — downloading...")
                download_from_drive(file_id, dest)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                all_ok = False

    return all_ok


def initialize():
    """Load model files — called once on server startup"""
    global model, scaler, FEATURE_COLS, explainer

    print("\n" + "="*50)
    print("  CSE VOLATILITY PREDICTOR — STARTUP")
    print("="*50)

    #  Check/download files from Google Drive
    print("\n Step 1: Checking model files...")
    ok = ensure_files()
    if not ok:
        print("Some files missing — check environment variables!")

    #  Load model into memory
    print("\n Step 2: Loading model into memory...")
    try:
        model        = joblib.load(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"))
        scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
        explainer    = shap.TreeExplainer(model)

        print(f"   GradientBoostingClassifier loaded")
        print(f"   StandardScaler loaded")
        print(f"   Feature columns loaded: {len(FEATURE_COLS)} features")
        print(f"   SHAP TreeExplainer ready")
        print(f"\n Server ready! All systems operational.")
        print("="*50 + "\n")

    except Exception as e:
        print(f"  Failed to load model: {e}")
        raise e



# TECHNICAL INDICATOR FUNCTIONS

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
    upper = sma + 2 * std
    lower = sma - 2 * std
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
    """
    Build all 56 features from raw OHLCV data.
    Must exactly match the feature engineering done in Colab Cell 4.
    """
    df     = hist_df.copy().reset_index(drop=True)
    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    volume = df['Volume']
    open_  = df['Open']

    # Moving Averages
    for w in [5, 10, 20, 50]:
        df[f'MA_{w}']  = close.rolling(w).mean()
        df[f'EMA_{w}'] = close.ewm(span=w, adjust=False).mean()

    # Price Features
    df['Daily_Return']     = close.pct_change()
    df['HL_Pct']           = (high - low) / (close + 1e-9)
    df['OC_Pct']           = (close - open_) / (open_ + 1e-9)
    df['Price_MA5_Ratio']  = close / (df['MA_5']  + 1e-9)
    df['Price_MA20_Ratio'] = close / (df['MA_20'] + 1e-9)
    df['Price_MA50_Ratio'] = close / (df['MA_50'] + 1e-9)
    df['MA5_MA20_Cross']   = df['MA_5']  - df['MA_20']
    df['MA10_MA50_Cross']  = df['MA_10'] - df['MA_50']

    # RSI
    df['RSI_7']  = compute_rsi(close, 7)
    df['RSI_14'] = compute_rsi(close, 14)
    df['RSI_21'] = compute_rsi(close, 21)

    # Momentum & ROC
    for p in [3, 5, 10, 20]:
        df[f'ROC_{p}']      = close.pct_change(p) * 100
        df[f'Momentum_{p}'] = close - close.shift(p)

    # MACD
    macd, sig, hist   = compute_macd(close)
    df['MACD']        = macd
    df['MACD_Signal'] = sig
    df['MACD_Hist']   = hist

    # Bollinger Bands
    bb_u, bb_l, bb_bw  = compute_bollinger(close, 20)
    df['BB_Upper']      = bb_u
    df['BB_Lower']      = bb_l
    df['BB_Bandwidth']  = bb_bw
    df['BB_Position']   = (close - bb_l) / ((bb_u - bb_l) + 1e-9)

    # Volatility
    df['Volatility_5']  = close.pct_change().rolling(5).std()
    df['Volatility_10'] = close.pct_change().rolling(10).std()
    df['Volatility_20'] = close.pct_change().rolling(20).std()
    df['ATR_14']        = compute_atr(high, low, close, 14)
    df['ATR_Ratio']     = df['ATR_14'] / (close + 1e-9)

    # Volume
    df['Volume_MA5']   = volume.rolling(5).mean()
    df['Volume_MA10']  = volume.rolling(10).mean()
    df['Volume_Ratio'] = volume / (df['Volume_MA5'] + 1e-9)
    df['OBV']          = compute_obv(close, volume)

    # Lag Features
    for lag in [1, 2, 3, 5]:
        df[f'Return_Lag_{lag}'] = close.pct_change().shift(lag)
        df[f'Close_Lag_{lag}']  = close.shift(lag)

    # Raw OHLCV
    df['Open']   = open_
    df['High']   = high
    df['Low']    = low
    df['Close']  = close
    df['Volume'] = volume

    # Return last row (most recent data point)
    return df[FEATURE_COLS].iloc[-1]



# API ENDPOINTS


@app.route('/health', methods=['GET'])
def health():
    """Health check — confirms server and model are running"""
    return jsonify({
        'status'         : 'ok',
        'model'          : 'GradientBoostingClassifier',
        'problem'        : 'CSE Stock Volatility Prediction',
        'features'       : len(FEATURE_COLS) if FEATURE_COLS else 0,
        'model_loaded'   : model is not None,
        'scaler_loaded'  : scaler is not None,
        'explainer_ready': explainer is not None
    })


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Returns real model metrics from CSV files saved during training.
    No hardcoded values — everything comes from actual training results.
    """
    try:
        metrics_df = pd.read_csv(
            os.path.join(RESULTS_DIR, 'model_metrics.csv'))
        feats_df   = pd.read_csv(
            os.path.join(RESULTS_DIR, 'feature_importances.csv')).head(10)

        # Build metrics dict from CSV
        metrics = {}
        for _, row in metrics_df.iterrows():
            metrics[row['Metric']] = {
                'validation': round(float(row['Validation']), 4),
                'test'      : round(float(row['Test']),       4)
            }

        # Build feature importance list from CSV
        features = [
            {
                'feature'   : row['Feature'],
                'importance': round(float(row['Importance']), 4)
            }
            for _, row in feats_df.iterrows()
        ]

        # Get actual hyperparameters from trained model
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
    """
    Main prediction endpoint.
    1. Fetches live data from Yahoo Finance
    2. Computes 56 technical indicators
    3. Scales using saved StandardScaler
    4. Predicts HIGH/LOW volatility using trained model
    5. Computes SHAP values for explainability
    6. Returns full result including price history
    """
    try:
        print(f"\n Prediction request: {ticker}")

        # Fetch live stock data 
        print(f"  Fetching live data from Yahoo Finance...")
        df = yf.download(
            ticker,
            period      = '120d',
            interval    = '1d',
            progress    = False,
            auto_adjust = True
        )

        if df is None or len(df) < 60:
            return jsonify({
                'error': f'Not enough data for {ticker}. Got {len(df) if df is not None else 0} rows. Need at least 60.'
            }), 400

        # Flatten multi-level columns (yfinance sometimes returns these)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"  Got {len(df)} rows | Latest: {df['Date'].iloc[-1].date()} | Close: {df['Close'].iloc[-1]:.2f} LKR")

        # Build features 
        print(f" Computing 56 technical indicators...")
        features_row = build_features(df)

        if features_row.isnull().any():
            null_feats = features_row[features_row.isnull()].index.tolist()
            return jsonify({
                'error': f'NaN found in features: {null_feats[:5]}. Try a different ticker.'
            }), 400

        # Scale features 
        X = scaler.transform([features_row.values])

        # Predict
        pred          = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        confidence    = float(probabilities[pred]) * 100

        print(f" Prediction: {'HIGH' if pred==1 else 'LOW'} volatility ({confidence:.1f}% confidence)")

        #  SHAP Explainability 
        print(f" Computing SHAP values...")
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

        # Build price history for chart
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
            # Stock info
            'ticker'        : ticker,
            'latest_date'   : str(latest['Date'].date()),
            'latest_close'  : round(float(latest['Close']),  2),
            'latest_open'   : round(float(latest['Open']),   2),
            'latest_high'   : round(float(latest['High']),   2),
            'latest_low'    : round(float(latest['Low']),    2),
            'latest_volume' : int(latest['Volume']),
            'data_rows'     : len(df),

            # Prediction
            'prediction'    : pred,
            'volatility'    : 'HIGH' if pred == 1 else 'LOW',
            'confidence'    : round(confidence, 2),
            'probabilities' : {
                'LOW' : round(float(probabilities[0]) * 100, 2),
                'HIGH': round(float(probabilities[1]) * 100, 2)
            },

            # Explainability
            'top_features'  : shap_list[:10],

            # Price chart data
            'price_history' : price_history,

            # Human readable explanation
            'interpretation': (
                'High market volatility expected. Prices may fluctuate significantly in the next 5 trading days. Consider risk management strategies.'
                if pred == 1 else
                'Low market volatility expected. Prices likely to remain relatively stable in the next 5 trading days.'
            )
        })

    except Exception as e:
        import traceback
        print(f" Error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500



# STARTUP — Initialize model when server starts

initialize()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)