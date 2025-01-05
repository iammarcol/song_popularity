import librosa
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

# Load the model and scaler
xgb_model = xgb.Booster()
xgb_model.load_model("model/xgb_popularity_model.json")  # Adjust path if necessary
scaler = joblib.load("scaler.joblib")  # Adjust path if necessary

def extract_features_and_predict(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # Extract features
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    energy = np.mean(librosa.feature.rms(y=y))
    loudness = 20 * np.log10(energy)

    # Chroma and key detection
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_strength = np.sum(chroma, axis=1)
    key_index = np.argmax(key_strength)
    key = key_index
    mode = 1 if key_strength[key_index] > np.mean(key_strength) else 0

    # Duration
    duration_ms = int(librosa.get_duration(y=y, sr=sr) * 1000)

    # Create feature DataFrame
    features = {
        'energy': [energy],
        'key': [key],
        'loudness': [loudness],
        'mode': [mode],
        'tempo': [tempo],
        'duration_ms': [duration_ms]
    }
    feature_df = pd.DataFrame(features)

    # Scale features
    scaled_features = scaler.transform(feature_df)

    # Predict popularity
    dmatrix_features = xgb.DMatrix(scaled_features)
    predictions = xgb_model.predict(dmatrix_features)

    # Add prediction to DataFrame
    feature_df['predicted_popularity'] = predictions

    return feature_df
