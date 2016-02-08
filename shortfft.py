import numpy as np
from scipy.signal import blackman, hanning
#Check different windows

#fs = frequency
#framesz = size of frame in seconds
#hop = size of step in seconds
def stft(x, fs, framesz, hop):
  framesamp = int(framesz*fs)
  hopsamp = int(hop*fs)
  w = hanning(framesamp)
  X = np.array([np.fft.fft(w*x[i:i+framesamp]) 
  for i in range(0, len(x)-framesamp, hopsamp)])
  return X

def istft(X, fs, T, hop):
  x = np.zeros(T*fs)
  framesamp = X.shape[1]
  hopsamp = int(hop*fs)
  for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
    x[i:i+framesamp] += np.real(np.fft.ifft(X[n]))
  return x
