import numpy as np
import joblib
from utils.lstm_model import LOOKBACK

def predict_future(model, df, ticker, days=1):
    scaler = joblib.load(f"models/{ticker}_scaler.save")
    data = scaler.transform(df[["Close"]])
    
    last_sequence = data[-LOOKBACK:]
    preds = []
    
    for _ in range(days):
        pred = model.predict(last_sequence.reshape(1, LOOKBACK, 1))
        preds.append(pred[0][0])
        last_sequence = np.append(last_sequence[1:], pred, axis=0)
    
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    return preds.flatten()
