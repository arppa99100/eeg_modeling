import os
import sys
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

from readbytes import _read8, _read32, _read_chunks
from datafilters import apply_dc_filter, apply_dwt_filter, apply_stfft_filter

# Constants
samplespath = "./samples"
predictingpath = "./predicting"

dt_i8 = np.dtype("<u1")
dt_f32 = np.dtype("<f4")

input_count = 0
num_samples = 0
num_channels = 8
max_samples = 360
max_presamples = 120


# Get Latest Folder
print("Reading data")
samplefolder = sorted(os.listdir(samplespath))[-1]
path = samplespath+"/"+samplefolder
    
# Get Inputs
with open(path+"/input", "rb") as bytestream:
    num_inputs = _read8(bytestream)
    buf = bytestream.read(num_inputs * dt_i8.itemsize)
    labels = np.frombuffer(buf, dtype=dt_i8)

# Get Samples and Presamples
dataset = []
predataset = []
allsamples = sorted(os.listdir(path+"/sample"))
for sample in allsamples:
    with open(path+"/sample/"+sample, "rb") as bytestream:
        #Check magic is 2049
        magic = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(max_samples * cols * dt_f32.itemsize)
        dataset.append(np.frombuffer(buf, dtype=dt_f32))
    with open(path+"/presample/"+sample, "rb") as bytestream:
        num_samples += 1
        #Check magic is 2049
        magic = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(max_presamples * cols * dt_f32.itemsize)
        predataset.append(np.frombuffer(buf, dtype=dt_f32))

# Reshape
dataset = np.array(dataset)
predataset = np.array(predataset)
dataset.shape = (num_samples, max_samples, num_channels)
predataset.shape = (num_samples, max_presamples, num_channels)


#Hyperparams

#DC Filter
enable_dc = True
dc_lowcut = 1.0
dc_highcut = 13.0
dc_order = 2
dc_type = "bandpass"
dc_func_type = "butter"

#DWT Filter
enable_dwt = False
dwt_type = "db2"
dwt_level = 4
dwt_thresh_func = "soft"
dwt_thresh_type = "rigrsure"

#STFT Filter
enable_fft = False
fft_window = 0.50
fft_step = 0.25
fft_thresh = 2.0
fft_set_thresh = 0.0

#Constants
fs = 250.0 #Frequency in Hz
sample_time = dataset.shape[1]/fs #Total time for sample
presample_time = predataset.shape[1]/fs #Total time for sample

#To get alpha and beta waves
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


dataset.shape = (dataset.shape[0], num_channels * max_samples)

# Load latest model
with open(predictingpath + '/eeg.model', 'rb') as readstream:
    clf = pickle.load(readstream)

# Predict
clf_pred = clf.predict(dataset)
print("Pred Labels: ", clf_pred)
print("Actual Labels: ", labels)
print("Accuracy: ", accuracy_score(labels, clf_pred))
