from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=5, fs=200, order=4):
    b, a = butter(order, cutoff/(fs/2), btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)