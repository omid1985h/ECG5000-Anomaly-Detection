# ECG5000 – ECG Arrhythmia Detection using Machine Learning and Deep Learning

## 1. Project Overview
This project performs a comprehensive analysis of the ECG5000 dataset, combining **biomedical signal processing, statistical time-series analysis, classical machine learning, and deep learning autoencoders** to detect arrhythmias from single-lead ECG signals.  
The goal is to explore and benchmark different approaches for **anomaly detection and classification** of ECG beats.

---

## 2. Dataset
- **ECG5000 dataset**: 5000 single-lead ECG heartbeats, each represented by 140 time-series points plus a class label.  
- **Data shape**: `(5000, 141)`  
- **Class distribution (binary mapping)**:
  - Class 0 (Normal): 2919 samples  
  - Class 1 (Arrhythmia): 2081 samples  
- **Binary task**: Class 1 = Normal → 0, All others → 1 (Arrhythmia)

---

## 3. Problem Definition
**Task:** Binary classification of ECG heartbeats as normal or arrhythmic.  
**Approaches explored:**
1. Statistical analysis of ECG signals (ADF test, rolling statistics, ARIMA modeling)  
2. Feature extraction (HRV metrics) + classical ML classifiers (Random Forest, SVM, Logistic Regression)  
3. Deep learning autoencoders (CNN and LSTM) for reconstruction-based anomaly detection

---

## 4. Methodology / Pipeline

1. **Data Loading & Preprocessing**
   - Combined train and test sets
   - Binary label mapping
   - Standardization of features/signals

2. **Statistical Time-Series Analysis**
   - Rolling mean and standard deviation  
   - ADF test for stationarity  
   - Autocorrelation (ACF) and Partial Autocorrelation (PACF)  
   - ARIMA modeling and residual analysis

3. **Feature Extraction**
   - Bandpass filtering (0.5–8 Hz)  
   - HRV features: mean heart rate, heart rate variability, peak counts  

4. **Classical ML**
   - Random Forest, SVM, Logistic Regression  
   - Metrics: Precision, Recall, F1

5. **Deep Learning**
   - CNN and LSTM autoencoders for reconstruction-based anomaly detection  

---

## 5. Results

### 5.1 Representative Plots
**Example ECG waveform:**  
![Example ECG waveform](notebooks/figures/example_ecg.png)

**Rolling statistics (mean ± std):**  
- Normal sample  
![Rolling stats normal](notebooks/figures/rolling_normal.png)  
- Abnormal sample  
![Rolling stats abnormal](notebooks/figures/rolling_abnormal.png)

**ARIMA fit & residuals:**  
- Normal  
![ARIMA normal](notebooks/figures/arima_normal.png)  
- Abnormal  
![ARIMA abnormal](notebooks/figures/arima_abnormal.png)

---

### 5.2 Statistical Analysis
- **ADF Test**
  - Normal: ADF = -1.8, p-value = 0.34 → Non-stationary  
  - Abnormal: ADF = -4.4, p-value = 0.002 → Stationary
- **ARIMA Residual Variance**
  - Normal: 0.11  
  - Abnormal: 0.037  

---

### 5.3 Classical Machine Learning

| Model | Precision | Recall | F1-score |
|-------|----------|--------|----------|
| Random Forest | 0.92 | 0.80 | 0.86 |
| SVM           | ~0.90 | ~0.78 | ~0.84 |
| Logistic Regression | 0.89 | 0.06 | 0.11 |

**Insights:**  
- Random Forest performed best overall  
- Logistic Regression achieved high precision but extremely low recall, failing to detect most arrhythmic samples

---

### 5.4 Deep Learning Autoencoders
- **CNN Autoencoder:** accurately reconstructed ECG signals; good for anomaly detection  
- **LSTM Autoencoder:** failed to reconstruct signals; predictions nearly flat (~0)  

**Representative reconstruction plots:**  
- CNN reconstruction  
![CNN reconstruction](notebooks/figures/cnn_recon.png)  
- LSTM reconstruction  
![LSTM reconstruction](notebooks/figures/lstm_recon.png)

---

## 6. Key Takeaways
- Arrhythmic samples are statistically different from normal beats (ADF, ARIMA residuals)  
- HRV-based features + classical ML (Random Forest) provide strong classification performance  
- CNN autoencoder is effective for ECG reconstruction, while LSTM autoencoder failed on this dataset  
- Logistic Regression is unsuitable for this imbalanced binary classification without additional handling  

---


