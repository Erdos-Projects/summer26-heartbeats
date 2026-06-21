import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
from scipy.fft import fft, fftfreq
from sklearn.base import BaseEstimator, TransformerMixin

class HeartRateFilter(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer for cleaning PAMAP2 Heart Rate data.
    Applies a Median filter for impulse noise (sensor glitches) followed by a 
    Butterworth low-pass filter for high-frequency noise removal.
    """
    def __init__(self, kernel_size=5, cutoff=11.0, fs=100.0, order=5):
        self.kernel_size = kernel_size
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        # Ensure input is a 1D numpy array and handle potential NaNs by forward filling
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            X = X.ffill().bfill().values
            
        X_array = np.asarray(X).flatten()
        
        # 1. Apply Median Filter 
        X_med = medfilt(X_array, self.kernel_size)
        
        # 2. Apply Butterworth Low-Pass Filter
        nyq = 0.5 * self.fs
        b, a = butter(self.order, self.cutoff / nyq, btype='low')
        X_clean = filtfilt(b, a, X_med)
        
        return X_clean.reshape(-1, 1)

if __name__ == "__main__":
    # REAL PAMAP2 Data ---
    print("Testing HeartRateFilter module on real PAMAP2 data...")
    
    # 1. Load the dataset
    # UPDATE THIS PATH to wherever you have the dataset stored !
    file_path = 'PAMAP2_Dataset/Protocol/subject105.dat'    
    try:
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path, sep=r'\s+', header=None)
        
        # Column 2 is the Heart Rate column in PAMAP2
        # We drop NaNs here for the test, but the transformer can handle them too
        raw_signal = df.iloc[:, 2].dropna().values 
        
        fs = 100.0 # 100 Hz sampling rate
        t = np.arange(len(raw_signal)) / fs
        
        # 2. Initialize and run transformer
        hr_transformer = HeartRateFilter(kernel_size=5, cutoff=11.0, fs=fs, order=5)
        cleaned_signal = hr_transformer.fit_transform(raw_signal)
        cleaned_signal_1d = cleaned_signal.flatten()
        
        print(f"Success! Transformer processed {len(cleaned_signal)} real data points.")
        print("Generating plots...")

        # 3. Plot the Results
        plt.figure(figsize=(14, 10))

        # --- Plot 1: Time Domain (Real Signal over Time) ---
        plt.subplot(2, 1, 1)
        # Slicing the plot [::10] so matplotlib doesn't crash from plotting 300k+ points
        plt.plot(t[::10], raw_signal[::10], label='Raw PAMAP2 HR Data', alpha=0.5, color='purple')
        plt.plot(t[::10], cleaned_signal_1d[::10], label='Cleaned HR Data', linewidth=2, color='orange')
        plt.title('Time Domain: Heart Rate Trend over Time (Subject 101)')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Amplitude (BPM)')
        plt.legend()
        plt.grid(True)

        # --- Plot 2: Frequency Domain (FFT Validation) ---
        plt.subplot(2, 1, 2)
        N = len(t)
        
        yf_orig = fft(raw_signal)
        yf_filt = fft(cleaned_signal_1d)
        xf = fftfreq(N, 1/fs)[:N//2]
        
        mag_orig = 2.0/N * np.abs(yf_orig[0:N//2])
        mag_filt = 2.0/N * np.abs(yf_filt[0:N//2])
        
        # Slice off the 0 Hz DC Offset
        plt.plot(xf[1:], mag_orig[1:], label='Raw Data FFT', alpha=0.5, color='purple')
        plt.plot(xf[1:], mag_filt[1:], label='Cleaned Data FFT (< 11.0 Hz)', linewidth=2, color='orange')
        
        plt.axvline(x=11.0, color='red', linestyle='--', label='11Hz Cutoff Wall')
        
        plt.title('Frequency Domain: Real Data Validating 11Hz Low-Pass Filter')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 30)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("filtered_data_validation.png", dpi=300)
        #plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Please check your file path and try again.")
