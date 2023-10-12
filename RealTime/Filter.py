from scipy.signal import butter, sosfilt

class ButterWorthBandpassFilter:
    def __init__(self,low_cut, high_cut,fs,order=5):
        self.low_cut = low_cut/(0.5*fs)
        self.high_cut = high_cut / (0.5 * fs)
        self.fs = fs
        self.order = order
        self.sos = butter(order, [low_cut, high_cut], btype='band', fs=fs, output='sos')

    def filter(self, data):
        return sosfilt(self.sos, data)
