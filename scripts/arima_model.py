# arima_module.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima

from utils.loader import load_ecg5000
from utils.preprocessing import normalize_sequences
from utils.general import plot_signal_with_anomalies

# --------------------- 1. Load & Preprocess ---------------------
X, y = load_ecg5000()
X_norm = normalize_sequences(X)
signal = X_norm[0]               # analyze first sequence
time_idx = np.arange(len(signal))

# --------------------- 2. Stationarity ---------------------
def adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
    return result

adf_result = adf_test(signal)
if adf_result[1] >= 0.05:
    signal = np.diff(signal)
    adf_test(signal)

# --------------------- 3. STL Decomposition ---------------------
stl = STL(pd.Series(signal), period=20)
res = stl.fit()

# Extract components
trend = res.trend
seasonal = res.seasonal
residual = res.resid


plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(time_idx, trend, color='blue')
plt.title("Trend")

plt.subplot(3, 1, 2)
plt.plot(time_idx, seasonal, color='green')
plt.title("Seasonal")

plt.subplot(3, 1, 3)
plt.plot(time_idx, residual, color='red')
plt.title("Residual")

plt.tight_layout()
plt.show()


# --------------------- 4. ACF & PACF ---------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_acf(signal, lags=40, ax=plt.gca())
plt.title("ACF")
plt.subplot(1, 2, 2)
plot_pacf(signal, lags=40, ax=plt.gca(), method='ywm')
plt.title("PACF")
plt.tight_layout()
plt.show()

# --------------------- 5. ARIMA Modeling ---------------------
model = ARIMA(signal, order=(1,0,1))
model_fit = model.fit()
print(model_fit.summary())

# --------------------- 6. Auto ARIMA ---------------------
auto_model = auto_arima(signal, seasonal=False, trace=True,
                        error_action='ignore', suppress_warnings=True,
                        stepwise=True)
print(auto_model.summary())

order = auto_model.order
model = ARIMA(signal, order=order)
model_fit = model.fit()

# --------------------- 7. Forecasting & Anomaly Detection ---------------------
n_forecast = 20
forecast_result = model_fit.get_forecast(steps=n_forecast)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

plt.figure(figsize=(10,5))
plt.plot(time_idx, signal, label="Observed")
plt.plot(np.arange(len(signal), len(signal)+n_forecast), forecast,
         label="Forecast", color='red')
plt.fill_between(np.arange(len(signal), len(signal)+n_forecast),
                 conf_int[:,0], conf_int[:,1], color='pink', alpha=0.3)
plt.legend()
plt.show()

in_sample_preds = model_fit.predict(start=0, end=len(signal)-1)
forecast_errors = signal - in_sample_preds
threshold = 3 * np.std(forecast_errors)
anomalies = np.where(np.abs(forecast_errors) > threshold)[0]

plot_signal_with_anomalies(signal, anomalies, "ECG with ARIMA Anomalies")

