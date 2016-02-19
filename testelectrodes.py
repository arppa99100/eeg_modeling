import os
import sys
import json

import numpy as np
from readbytes import _read8, _read32
from datafilters import apply_dc_filter

dt_f32 = np.dtype("<f4")
electrodes = [False, False, False, False, False, False, False, False]

masterfolder = "testing"
samplefolder = "sample"

#Read raw data
path = os.path.abspath(os.path.join(__file__,"../"))
testingpath = path + "/"+ masterfolder + "/" + samplefolder
filename = os.listdir(testingpath)[0]

max_rows = 260

with open(testingpath+"/"+filename, "rb") as readstream:
  magic = _read32(readstream)
  rows = _read32(readstream)
  cols = _read32(readstream)
  buf = readstream.read(max_rows * cols * dt_f32.itemsize)
  data = np.frombuffer(buf, dtype=dt_f32)
  data.shape = (max_rows, cols)

min_thresh = -60000
max_thresh = -10000
#min_thresh = -38000
#max_thresh = -12000
for i in range(0,data.shape[1]):
  if max(data[:,i]) < max_thresh and min(data[:,i]) > min_thresh:
    electrodes[i] = True

#DC Filter
fs = 250
enable_dc = True
dc_lowcut = 8.0 #Only get alpha and beta, most related to movement
dc_highcut = 30.0
dc_order = 2
dc_type = "bandpass"
dc_func_type = "butter"

data.flags['WRITEABLE'] = True
for i in range(0,data.shape[1]):
  if enable_dc:
    data[:,i] = apply_dc_filter(data[:,i], fs, dc_lowcut, dc_highcut, dc_order, dc_type, dc_func_type)

#data = np.delete(data, range(data.shape[0] - 25, data.shape[0]), 0)
#data = np.delete(data, range(0,25), 0)
np.savetxt(path + "/../public/data/test.csv", data, delimiter=",",newline="\n")

with open(path+"/testing/out.log", "w") as writestream:
  writestream.write(json.dumps({"electrodes": electrodes}))
