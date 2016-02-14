import os
import sys
import numpy as np

from dataio import readdata, readlabels, writedata
from datafilters import apply_dc_filter, apply_dwt_filter, apply_stfft_filter

########### Hyperparams #########################################
#DC Filter
enable_dc = True
dc_lowcut = 1.0
dc_highcut = 13.0
dc_order = 2
dc_type = "bandpass"
dc_func_type = "butter"

#DWT Filter
enable_dwt = False
dwt_type = "db2" #investigate with ellip and four poles
dwt_level = 4
dwt_thresh_func = "soft"
dwt_thresh_type = "rigrsure"

#STFT Filter
enable_fft = False
fft_window = 0.50
fft_step = 0.25 #50% windows
fft_thresh = 2.0 #Works with new dc_lowcut, investigate better values
fft_set_thresh = 0.0

#Datasets Sizes (%)
test_size = 0.15
valid_size = 0.15

###################################################################

#Get unfiltered data
path = os.path.abspath(os.path.join(__file__,"../"))
predataset = readdata(path + "/curated/raw-presamples")
dataset = readdata(path + "/curated/raw-samples")
labels = readlabels(path + "/curated/raw-inputs")

#Constants
fs = 250.0 #Frequency in Hz
sample_time = dataset.shape[1]/fs #Total time for sample
presample_time = predataset.shape[1]/fs #Total time for sample

#Apply filters
print("DC Filter:", enable_dc)
print("Discrete Wavelet Transform:", enable_dwt)
print("Fast Fourier Transform:", enable_fft)
print("Applying filters...")
dataset.flags['WRITEABLE'] = True
predataset.flags['WRITEABLE'] = True
for i in range(0,dataset.shape[0]):
  for j in range(0,dataset.shape[2]):
    if enable_dc:
      predataset[i,:,j] = apply_dc_filter(predataset[i,:,j], fs, dc_lowcut, dc_highcut, dc_order, dc_type, dc_func_type)
      dataset[i,:,j] = apply_dc_filter(dataset[i,:,j], fs, dc_lowcut, dc_highcut, dc_order, dc_type, dc_func_type)
    if enable_dwt:
      predataset[i,:,j] = apply_dwt_filter(predataset[i,:,j], dwt_type, dwt_level, dwt_thresh_func, dwt_thresh_type)
      dataset[i,:,j] = apply_dwt_filter(dataset[i,:,j], dwt_type, dwt_level, dwt_thresh_func, dwt_thresh_type)
    if enable_fft:
      predataset[i,:,j] = apply_stfft_filter(predataset[i,:,j], fs, presample_time, fft_window, fft_step, fft_thresh, fft_set_thresh)
      dataset[i,:,j] = apply_stfft_filter(dataset[i,:,j], fs, sample_time, fft_window, fft_step, fft_thresh, fft_set_thresh)
    #Normalize
    dataset[i,:,j] = dataset[i,:,j]/np.linalg.norm(dataset[i,:,j])
    #predataset[i,:,j] = predataset[i,:,j]/np.linalg.norm(predataset[i,:,j])
    #np.mean(predataset[i,:,j])

print("Partitioning Data...")
# Test Set
dataindex = range(0,dataset.shape[0])
test_index = np.random.choice(dataindex, int(dataset.shape[0]*test_size))
test_dataset = dataset[test_index,:,:]
test_labels = labels[test_index]
dataset = np.delete(dataset, test_index, axis = 0)
labels = np.delete(labels, test_index, axis = 0)

# Validation Set
#dataindex = range(0,dataset.shape[0])
#valid_index = np.random.choice(dataindex, int(dataset.shape[0]*valid_size))
#valid_dataset = dataset[valid_index,:,:]
#valid_labels = labels[valid_index]

# Train Set
train_dataset = dataset
train_labels = labels

print("Finished applying filters. Data structure:")
print("Training:", train_dataset.shape, train_labels.shape)
#print("Validation:", valid_dataset.shape, valid_labels.shape)
print("Testing:", test_dataset.shape, test_labels.shape)


#Write training
writedata(path + "/curated/train_dataset", train_dataset)
writedata(path + "/curated/train_labels", train_labels)
#Write validation
#writedata(path + "/curated/valid_dataset", valid_dataset)
#writedata(path + "/curated/valid_labels", valid_labels)
#Write test
writedata(path + "/curated/test_dataset", test_dataset)
writedata(path + "/curated/test_labels", test_labels)
