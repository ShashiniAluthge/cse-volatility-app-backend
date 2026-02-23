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


# ── GOOGLE DRIVE FILE IDs ─────────────────────────────────────────────────────

DRIVE_FILES = {
    "gradient_boosting_model.pkl" : os.getenv("GDRIVE_MODEL_ID"),
    "scaler.pkl"                  : os.getenv("GDRIVE_SCALER_ID"),
    "feature_cols.pkl"            : os.getenv("GDRIVE_FEATURES_ID"),
    "model_metrics.csv"           : os.getenv("GDRIVE_METRICS_ID"),
    "feature_importances.csv"     : os.getenv("GDRIVE_IMPORTANCES_ID"),
    "random_forest_model.pkl"     : os.getenv("GDRIVE_RF_MODEL_ID"),
    "rf_metrics.csv"              : os.getenv("GDRIVE_RF_METRICS_ID"),
    "rf_feature_importances.csv"  : os.getenv("GDRIVE_RF_IMPORTANCES_ID"),
}

MODEL_DIR   = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Global model variables ────────────────────────────────────────────────────

model        = None   # GradientBoostingClassifier
scaler       = None
FEATURE_COLS = None
explainer    = None   # SHAP for GB

rf_model     = None   # RandomForestClassifier
rf_explainer = None   # SHAP for RF


# ── GOOGLE DRIVE DOWNLOAD ─────────────────────────────────────────────────────

def download_from_drive(file_id, dest_path):
    print(f"  Downloading {os.path.basename(dest_path)}...")
    url      = f"https://drive.google.com/uc?export=download&id={file_id}"
    session  = requests.Session()
    response = session.get(url, stream=True)

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

    size_kb = os.path.getsize(dest_path) / 1024
    print(f"  {os.path.basename(dest_path)} downloaded! ({size_kb:.1f} KB)")


def ensure_files():
    """Download any missing model files from Google Drive"""
    all_ok = True
    for filename, file_id in DRIVE_FILES.items():
        dest = os.path.join(
            RESULTS_DIR if filename.endswith('.csv') else MODEL_DIR,
            filename
        )
        if os.path.exists(dest):
            print(f"  {filename} already exists")
        else:
            if not file_id:
                print(f"  ERROR: No Drive ID set for {filename}")
                all_ok = False
                continue
            try:
                download_from_drive(file_id, dest)
            except Exception as e:
                print(f"  Failed to download {filename}: {e}")
                all_ok = False
    return all_ok


def initialize():
    """Load all model files — called once on server startup"""
    global model, scaler, FEATURE_COLS, explainer, rf_model, rf_explainer

    print("\n" + "="*55)
    print("  CSE VOLATILITY PREDICTOR — STARTUP")
    print("="*55)

    print("\nStep 1: Checking model files...")
    ensure_files()

    print("\nStep 2: Loading models into memory...")

    # ── Gradient Boosting ──
    try:
        model        = joblib.load(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"))
        scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
        explainer    = shap.TreeExplainer(model)
        print(f"  GradientBoostingClassifier loaded | Features: {len(FEATURE_COLS)}")
        print(f"  StandardScaler loaded")
        print(f"  SHAP TreeExplainer (GB) ready")
    except Exception as e:
        print(f"  Failed to load GradientBoosting: {e}")
        raise e

    # ── Random Forest ──
    try:
        rf_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
        if os.path.exists(rf_path):
            rf_model     = joblib.load(rf_path)
            rf_explainer = shap.TreeExplainer(rf_model)
            print(f"  RandomForestClassifier loaded")
            print(f"  SHAP TreeExplainer (RF) ready")
        else:
            print(f"  WARNING: Random Forest model not found — RF endpoints will be unavailable")
    except Exception as e:
        print(f"  WARNING: Failed to load RandomForest: {e}")

    print(f"\nServer ready! All systems operational.")
    print("="*55 + "\n")


# ── TECHNICAL INDICATOR FUNCTIONS ─────────────────────────────────────────────

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

    macd, sig, hist   = compute_macd(close)
    df['MACD']        = macd
    df['MACD_Signal'] = sig
    df['MACD_Hist']   = hist

    bb_u, bb_l, bb_bw = compute_bollinger(close, 20)
    df['BB_Upper']     = bb_u
    df['BB_Lower']     = bb_l
    df['BB_Bandwidth'] = bb_bw
    df['BB_Position']  = (close - bb_l) / ((bb_u - bb_l) + 1e-9)

    df['Volatility_5']  = close.pct_change().rolling(5).std()
    df['Volatility_10'] = close.pct_change().rolling(10).std()
    df['Volatility_20'] = close.pct_change().rolling(20).std()
    df['ATR_14']        = compute_atr(high, low, close, 14)
    df['ATR_Ratio']     = df['ATR_14'] / (close + 1e-9)

    df['Volume_MA5']   = volume.rolling(5).mean()
    df['Volume_MA10']  = volume.rolling(10).mean()
    df['Volume_Ratio'] = volume / (df['Volume_MA5'] + 1e-9)
    df['OBV']          = compute_obv(close, volume)

    for lag in [1, 2, 3, 5]:
        df[f'Return_Lag_{lag}'] = close.pct_change().shift(lag)
        df[f'Close_Lag_{lag}']  = close.shift(lag)

    df['Open']   = open_
    df['High']   = high
    df['Low']    = low
    df['Close']  = close
    df['Volume'] = volume

    return df[FEATURE_COLS].iloc[-1]


def fetch_stock_data(ticker):
    """Shared function to fetch and clean Yahoo Finance data"""
    df = yf.download(ticker, period='120d', interval='1d',
                     progress=False, auto_adjust=True)

    if df is None or len(df) < 60:
        return None, f"Not enough data for {ticker}. Got {len(df) if df is not None else 0} rows."

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, None


def run_shap(shap_explainer, X, mdl):
    """Compute SHAP values — handles both GB and RF output shapes"""
    sv = shap_explainer.shap_values(X)
    if isinstance(sv, list):
        # RF: sv is list of 2D arrays [n_samples, n_features] per class
        # Use class-1 (HIGH volatility) and flatten to 1D
        sv_flat = np.array(sv[1]).flatten()
    else:
        # GB: sv is 2D array [n_samples, n_features]
        arr = np.array(sv)
        sv_flat = arr.flatten() if arr.ndim > 1 else arr
    return sv_flat


def build_shap_list(sv_flat, feature_importances):
    """Build sorted SHAP feature list"""
    return sorted([
        {
            'feature'   : feat,
            'shap_value': round(float(sv_flat[i]), 5),
            'importance': round(float(feature_importances[i]), 5)
        }
        for i, feat in enumerate(FEATURE_COLS)
    ], key=lambda x: abs(x['shap_value']), reverse=True)


# Median values for all 56 features (from CSE training dataset)
FEATURE_MEDIANS = {
    'MA_5': 66.0, 'MA_10': 66.0, 'MA_20': 66.0, 'MA_50': 66.0,
    'EMA_5': 66.0, 'EMA_10': 66.0, 'EMA_20': 66.0, 'EMA_50': 66.0,
    'Daily_Return': 0.0, 'HL_Pct': 0.02, 'OC_Pct': 0.0,
    'Price_MA5_Ratio': 1.0, 'Price_MA20_Ratio': 1.0, 'Price_MA50_Ratio': 1.0,
    'MA5_MA20_Cross': 0.0, 'MA10_MA50_Cross': 0.0,
    'RSI_7': 50.0, 'RSI_14': 50.0, 'RSI_21': 50.0,
    'ROC_3': 0.0, 'ROC_5': 0.0, 'ROC_10': 0.0, 'ROC_20': 0.0,
    'Momentum_3': 0.0, 'Momentum_5': 0.0, 'Momentum_10': 0.0, 'Momentum_20': 0.0,
    'MACD': 0.0, 'MACD_Signal': 0.0, 'MACD_Hist': 0.0,
    'BB_Upper': 70.0, 'BB_Lower': 62.0, 'BB_Bandwidth': 0.08, 'BB_Position': 0.5,
    'Volatility_5': 0.012, 'Volatility_10': 0.015, 'Volatility_20': 0.016,
    'ATR_14': 1.3, 'ATR_Ratio': 0.02,
    'Volume_MA5': 280000.0, 'Volume_MA10': 280000.0,
    'Volume_Ratio': 1.0, 'OBV': 0.0,
    'Return_Lag_1': 0.0, 'Return_Lag_2': 0.0,
    'Return_Lag_3': 0.0, 'Return_Lag_5': 0.0,
    'Close_Lag_1': 66.0, 'Close_Lag_2': 66.0,
    'Close_Lag_3': 66.0, 'Close_Lag_5': 66.0,
    'Open': 66.0, 'High': 67.5, 'Low': 64.5,
    'Close': 66.0, 'Volume': 280000.0,
}


def build_manual_feature_row(user_inputs):
    """Build full 56-feature vector from user inputs + dataset medians"""
    return [
        float(user_inputs[feat]) if feat in user_inputs
        else FEATURE_MEDIANS.get(feat, 0.0)
        for feat in FEATURE_COLS
    ]


def make_prediction_response(mdl, shap_expl, X, ticker, df, interpretation_fn):
    """Shared prediction + SHAP response builder for live endpoints"""
    pred          = int(mdl.predict(X)[0])
    probabilities = mdl.predict_proba(X)[0]
    confidence    = float(probabilities[pred]) * 100

    sv_flat   = run_shap(shap_expl, X, mdl)
    shap_list = build_shap_list(sv_flat, mdl.feature_importances_)

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

    return {
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
            'HIGH': round(float(probabilities[1]) * 100, 2),
        },
        'top_features'  : shap_list[:10],
        'price_history' : price_history,
        'interpretation': interpretation_fn(pred),
    }


def make_manual_response(mdl, shap_expl, X, user_inputs):
    """Shared prediction + SHAP response builder for manual endpoints"""
    pred          = int(mdl.predict(X)[0])
    probabilities = mdl.predict_proba(X)[0]
    confidence    = float(probabilities[pred]) * 100

    sv_flat   = run_shap(shap_expl, X, mdl)
    shap_list = build_shap_list(sv_flat, mdl.feature_importances_)

    return {
        'prediction'    : pred,
        'volatility'    : 'HIGH' if pred == 1 else 'LOW',
        'confidence'    : round(confidence, 2),
        'probabilities' : {
            'LOW' : round(float(probabilities[0]) * 100, 2),
            'HIGH': round(float(probabilities[1]) * 100, 2),
        },
        'top_features'  : shap_list[:10],
        'interpretation': (
            'High volatility expected based on your inputs. Indicators suggest significant price fluctuations.'
            if pred == 1 else
            'Low volatility expected based on your inputs. Indicators suggest relatively stable prices.'
        ),
        'inputs_used'   : user_inputs,
    }


# ── API ENDPOINTS ─────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'       : 'ok',
        'gb_loaded'    : model       is not None,
        'rf_loaded'    : rf_model    is not None,
        'scaler_loaded': scaler      is not None,
        'features'     : len(FEATURE_COLS) if FEATURE_COLS else 0,
        'gb_explainer' : explainer    is not None,
        'rf_explainer' : rf_explainer is not None,
    })


# ── GRADIENT BOOSTING ─────────────────────────────────────────────────────────

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        metrics_df = pd.read_csv(os.path.join(RESULTS_DIR, 'model_metrics.csv'))
        feats_df   = pd.read_csv(os.path.join(RESULTS_DIR, 'feature_importances.csv')).head(10)

        metrics = {}
        for _, row in metrics_df.iterrows():
            metrics[row['Metric']] = {
                'validation': round(float(row['Validation']), 4),
                'test'      : round(float(row['Test']),       4)
            }

        features = [
            {'feature': row['Feature'], 'importance': round(float(row['Importance']), 4)}
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
        print(f"\nGB Prediction: {ticker}")
        df, err = fetch_stock_data(ticker)
        if err:
            return jsonify({'error': err}), 400

        features_row = build_features(df)
        if features_row.isnull().any():
            null_feats = features_row[features_row.isnull()].index.tolist()
            return jsonify({'error': f'NaN in features: {null_feats[:5]}'}), 400

        X = scaler.transform([features_row.values])

        return jsonify(make_prediction_response(
            model, explainer, X, ticker, df,
            lambda pred: (
                'High market volatility expected. Prices may fluctuate significantly in the next 5 trading days.'
                if pred == 1 else
                'Low market volatility expected. Prices likely to remain relatively stable in the next 5 trading days.'
            )
        ))
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        user_inputs = request.get_json().get('features', {})
        print(f"\nGB Manual prediction")
        feature_row = build_manual_feature_row(user_inputs)
        X           = scaler.transform([feature_row])
        return jsonify(make_manual_response(model, explainer, X, user_inputs))
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ── RANDOM FOREST ─────────────────────────────────────────────────────────────

@app.route('/rf_metrics', methods=['GET'])
def get_rf_metrics():
    if rf_model is None:
        return jsonify({'error': 'Random Forest model not loaded'}), 503
    try:
        metrics_df = pd.read_csv(os.path.join(RESULTS_DIR, 'rf_metrics.csv'))
        feats_df   = pd.read_csv(os.path.join(RESULTS_DIR, 'rf_feature_importances.csv')).head(10)

        metrics = {}
        for _, row in metrics_df.iterrows():
            metrics[row['Metric']] = {
                'validation': round(float(row['Validation']), 4),
                'test'      : round(float(row['Test']),       4)
            }

        features = [
            {'feature': row['Feature'], 'importance': round(float(row['Importance']), 4)}
            for _, row in feats_df.iterrows()
        ]

        params = rf_model.get_params()
        best_params = {
            'n_estimators'    : params.get('n_estimators'),
            'max_depth'       : params.get('max_depth'),
            'min_samples_leaf': params.get('min_samples_leaf'),
            'max_features'    : params.get('max_features'),
            'random_state'    : params.get('random_state'),
        }

        return jsonify({
            'metrics'    : metrics,
            'features'   : features,
            'best_params': best_params,
            'n_features' : len(FEATURE_COLS),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_rf/<ticker>', methods=['GET'])
def predict_rf(ticker):
    if rf_model is None:
        return jsonify({'error': 'Random Forest model not loaded'}), 503
    try:
        print(f"\nRF Prediction: {ticker}")
        df, err = fetch_stock_data(ticker)
        if err:
            return jsonify({'error': err}), 400

        features_row = build_features(df)
        if features_row.isnull().any():
            null_feats = features_row[features_row.isnull()].index.tolist()
            return jsonify({'error': f'NaN in features: {null_feats[:5]}'}), 400

        X = scaler.transform([features_row.values])

        return jsonify(make_prediction_response(
            rf_model, rf_explainer, X, ticker, df,
            lambda pred: (
                'Random Forest predicts HIGH volatility. Ensemble of decision trees detected volatile patterns.'
                if pred == 1 else
                'Random Forest predicts LOW volatility. Ensemble of decision trees detected stable patterns.'
            )
        ))
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/predict_manual_rf', methods=['POST'])
def predict_manual_rf():
    if rf_model is None:
        return jsonify({'error': 'Random Forest model not loaded'}), 503
    try:
        user_inputs = request.get_json().get('features', {})
        print(f"\nRF Manual prediction")
        feature_row = build_manual_feature_row(user_inputs)
        X           = scaler.transform([feature_row])
        return jsonify(make_manual_response(rf_model, rf_explainer, X, user_inputs))
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ── STARTUP ───────────────────────────────────────────────────────────────────

initialize()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)