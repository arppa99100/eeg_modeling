import os
import sys
import json

import numpy as np
from readbytes import _read8, _read32
from datafilters import apply_dc_filter

dt_f32 = np.dtype("<f4")
electrodes = [False, False, False, False, False, False, False, False]

#Gather data from Hardware
#Get data from hardware
#fileloc = "./../lib/start_stream.py"
#dumploc = "../../modeling"
masterfolder = "testing"
samplefolder = "sample"
#sampletime = 2 #sec
#sampleid = 0
#os.system("python3 ./../lib/start_stream.py ../../modeling/ 2 testing sample 0")

#Read data from Hardware
path = os.path.abspath(os.path.join(__file__,sys.argv[1]))
testingpath = path + "/"+ masterfolder + "/" + samplefolder
filename = os.listdir(testingpath)[0]

max_rows = 470

with open(testingpath+"/"+filename, "rb") as readstream:
  magic = _read32(readstream)
  cols = _read32(readstream)
  rows = _read32(readstream)
  buf = readstream.read(max_rows * cols * dt_f32.itemsize)
  data = np.frombuffer(buf, dtype=dt_f32)
  data.shape = (max_rows, cols)


min_thresh = -38000
max_thresh = -12000
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

np.savetxt(path + "/../public/data/test.csv", data, delimiter=",",newline="\n")

with open(path+"/testing/out.log", "w") as writestream:
  writestream.write(json.dumps({"electrodes": electrodes}))
