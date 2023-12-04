# Function to design a Butterworth band-pass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply a band-pass filter
def apply_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def extract_statistical_features(segment):
    return {
        'mean': np.mean(segment),
        'std_dev': np.std(segment),
        'variance': np.var(segment),
        'min': np.min(segment),
        'max': np.max(segment),
    }