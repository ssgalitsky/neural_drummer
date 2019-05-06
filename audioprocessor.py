# audio processing and data preparation functions for neural effect emulation
import numpy as np
import librosa

class AudioProcessor(object):
    """Class for processing stereo wav files into training data pairs"""

    def __init__(self, clean_path, fx_path):
        super(AudioProcessor, self).__init__()
        self.clean_path = clean_path
        self.fx_path = fx_path

    def load_data(self):
        # load the original audio data
        self.no_fx, self.sr = librosa.load(self.clean_path, mono=False, sr=None)
        self.fx, self.sr2 = librosa.load(self.fx_path, mono=False, sr=None)
        if self.sr != self.sr2:
            print("SAMPLE RATE MISMATCH: ", self.sr, self.sr2)
        # make it mono
        self.no_fx = self.no_fx.reshape(self.no_fx.shape[1]*2)
        self.fx = self.fx.reshape(self.fx.shape[1]*2)

    def fft(self, fft_len_sec = 0.2, hop_percentage = 0.5):
        # compute n_fft, hop_length
        self.n_fft = int(fft_len_sec*self.sr)
        self.hop_length = int(hop_percentage*self.n_fft)
        # compute abs values of fft
        self.no_fx_fft = np.abs(librosa.core.stft(self.no_fx,n_fft=self.n_fft,
                                                  hop_length=self.hop_length))
        self.fx_fft = np.abs(librosa.core.stft(self.fx,n_fft=self.n_fft,
                                                  hop_length=self.hop_length))
        # stats
        self.n_freq = self.no_fx_fft.shape[0]
        # print(self.no_fx.shape[0])
        self.total_samples = self.no_fx_fft.shape[1]
        print("raw window length ", self.n_fft)
        print("raw FFT shape ", self.no_fx_fft.shape)

    def create_training_pairs(self, history_len_sec = 3.0, cutoff = 1.0):
        # take blocks of history with every output vector
        # apply cutoff on input ffts
        hop_sec = self.hop_length/float(self.sr)
        # print(hop_sec)
        self.history_samples = int(history_len_sec/hop_sec)
        self.n_cutoff = int(cutoff*self.n_freq)
        print("using ", self.history_samples, " fft samples as history")
        print("input cutoff at ", self.n_cutoff, " of ", self.n_freq)

        # make the input samples, creating 'images' of history_samples x no_fx_fft.shape[0] (feature length)
        # coupled with output vectors of no_fx.shape[0] output vecs
        input_list = []
        for i in range(1,self.total_samples+1):
            # add zero padding, leading
            if i < self.history_samples:
                arr = np.zeros((self.history_samples,self.n_cutoff),dtype="float32")
                arr[-i:,:] = self.no_fx_fft[:self.n_cutoff,:i].T
            else:
                arr = self.no_fx_fft[:self.n_cutoff,i-self.history_samples:i].T
            input_list.append(arr)
        self.input_arr = np.expand_dims(np.array(input_list),axis=1)
        print("input array shape: ",self.input_arr.shape)
        print("fx_fft and input_arr now accessible")
