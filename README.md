# Neural Drummer
General audio work using LSTMs
first part: imitate analogue effects
inputs: raw and processed audio
converted to wavelet tranform vectors
1-to-1 mapping with tensorflow LSTM


second part (TODO):

input: DWT of drums, corresponding bpm

architecture: convolutional + lstm?

convolutions might be too heavy for real-time use, also might result in frequency inaccuracy because of pooling but interesting to look at
