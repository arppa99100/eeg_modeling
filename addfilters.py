import numpy as np

from dataio import readdata, readdata2, readlabels, writedata
from datafilters import apply_dc_filter, apply_dwt_filter, apply_stfft_filter

########### Hyperparams #########################################
#DC Filter
enable_dc = True
dc_lowcut = 8.0 #Only get alpha and beta, most related to movement
dc_highcut = 30.0
dc_order = 2
dc_type = "bandpass"
dc_func_type = "butter"

#DWT Filter
enable_dwt = True
dwt_type = "db2" #investigate with ellip and four poles
dwt_level = 4
dwt_thresh_func = "soft"
dwt_thresh_type = "rigrsure"

#STFT Filter
enable_fft = True
fft_window = 0.50
fft_step = 0.25 #50% windows
fft_thresh = 2.0 #Works with new dc_lowcut, investigate better values
fft_set_thresh = 0.0

#Datasets Sizes (%)
test_size = 0.20
valid_size = 0.20

###################################################################

#Get unfiltered data
dataset = readdata("sampledata")
labels = readlabels("sampleinput")

#Constants
fs = 250.0 #Frequency in Hz
sample_time = dataset.shape[1]/fs #Total time for sample

#Apply filters
print("DC Filter:", enable_dc)
print("Discrete Wavelet Transform:", enable_dwt)
print("Fast Fourier Transform:", enable_fft)
print("Applying filters...")
dataset.flags['WRITEABLE'] = True
for i in range(0,dataset.shape[0]):
  for j in range(0,dataset.shape[2]):
    if enable_dc:
      dataset[i,:,j] = apply_dc_filter(dataset[i,:,j], fs, dc_lowcut, dc_highcut, dc_order, dc_type, dc_func_type)
    if enable_dwt:
      dataset[i,:,j] = apply_dwt_filter(dataset[i,:,j], dwt_type, dwt_level, dwt_thresh_func, dwt_thresh_type)
    if enable_fft:
      dataset[i,:,j] = apply_stfft_filter(dataset[i,:,j], fs, sample_time, fft_window, fft_step, fft_thresh, fft_set_thresh)
    #Normalize
    dataset[i,:,j] = dataset[i,:,j]/np.linalg.norm(dataset[i,:])


# Test Set
dataindex = range(0,dataset.shape[0])
test_index = np.random.choice(dataindex, int(dataset.shape[0]*test_size))
test_dataset = dataset[test_index,:,:]
test_labels = labels[test_index]
dataset = np.delete(dataset, test_index, axis = 0)
labels = np.delete(labels, test_index, axis = 0)

# Validation Set
dataindex = range(0,dataset.shape[0])
valid_index = np.random.choice(dataindex, int(dataset.shape[0]*valid_size))
valid_dataset = dataset[valid_index,:,:]
valid_labels = labels[valid_index]

# Train Set
train_dataset = dataset
train_labels = labels

print("Finished applying filters. Data structure:")
print("Training:", train_dataset.shape, train_labels.shape)
print("Validation:", valid_dataset.shape, valid_labels.shape)
print("Testing:", test_dataset.shape, test_labels.shape)
