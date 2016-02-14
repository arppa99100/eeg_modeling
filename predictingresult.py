import os
import sys
import json
import pickle

import numpy as np
from readbytes import _read8, _read32
from datafilters import apply_dc_filter
from sklearn.ensemble import RandomForestClassifier

num_channels = 8
max_rows = 470
masterfolder = "predicting"
dt_f32 = np.dtype("<f4")

#Read data from Hardware
path = os.path.abspath(os.path.join(__file__,sys.argv[1]))
testingpath = path + "/predicting/sample"
filename = os.listdir(testingpath)[0]

with open(testingpath+"/"+filename, "rb") as readstream:
  magic = _read32(readstream)
  cols = _read32(readstream)
  rows = _read32(readstream)
  buf = readstream.read(max_rows * cols * dt_f32.itemsize)
  data = np.frombuffer(buf, dtype=dt_f32)
  data.shape = (max_rows, cols)

min_thresh = -60000
max_thresh = -10000

#DC Filter
fs = 250
enable_dc = True
dc_lowcut = 1.0 #Only get alpha and beta, most related to movement
dc_highcut = 13.0
dc_order = 2
dc_type = "bandpass"
dc_func_type = "butter"

data.flags['WRITEABLE'] = True
for i in range(0,data.shape[1]):
  if enable_dc:
    data[:,i] = apply_dc_filter(data[:,i], fs, dc_lowcut, dc_highcut, dc_order, dc_type, dc_func_type)
data.shape = (data.shape[0], num_channels * max_rows)

#Read Model
with open('./predicting/eeg.model', 'rb') as readstream:
  forest = pickle.load(readstream)

print(forest.predict(data))
