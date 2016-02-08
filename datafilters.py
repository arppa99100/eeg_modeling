import pywt
import numpy as np
from numpy import fft
from thselect import thselect
from shortfft import stft, istft
from scipy.signal import iirfilter, filtfilt

#Butterworth low-high bandpass filter from 0.3 to 30 Hz to remove DC noise
#Check difference between butter vs four pole elliptic
def apply_dc_filter(y, fs, dc_lowcut, dc_highcut, dc_order, dc_type, dc_func_type):
  b, a = iirfilter(dc_order, np.array([dc_lowcut, dc_highcut])/(fs/2), btype=dc_type, ftype=dc_func_type)
  return(filtfilt(b,a,y))

#DWT with Daubechies 2 level 4
def apply_dwt_filter(y, dwt_type, dwt_level, dwt_thresh_func, dwt_thresh_type):
  coeffs = pywt.wavedecn(y, dwt_type, level=dwt_level)
  for i in range(1,dwt_level+1):
    coeffs[i]["d"] = pywt.threshold(coeffs[i]["d"], thselect(coeffs[i]["d"], dwt_thresh_type), dwt_thresh_func)
  return(pywt.waverecn(coeffs, dwt_type))
  
#FFT with multiple windows
def apply_stfft_filter(y, fs, sample_time, fft_window, fft_step, fft_thresh, fft_set_thresh):
  _y = stft(y, fs, fft_window, fft_step)
  norm = 2.0/_y[:,0].size
  for i in range(_y[0,:].size):
    thresh_invalid = norm*np.abs(_y[:,i]) < fft_thresh
    _y[thresh_invalid,i] = fft_set_thresh
  return(istft(_y, fs, sample_time, fft_step))
