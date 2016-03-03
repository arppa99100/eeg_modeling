import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import json
import scipy
import numpy as np

from sklearn.preprocessing import normalize

from readbytes import _read8, _read32
from datafilters import apply_dc_filter
from wignerville import wvd, filtered_wvd

filename = sys.argv[1]
values = json.loads(sys.argv[2])

fs = 250
num_channels = 8
max_rows = int(fs*float(values["sampletime"]) - 10)
rto = max_rows/(fs/2)
dt_f32 = np.dtype("<f4")


path = os.path.abspath(os.path.join(__file__,"../"))
visualizepath = path + "/visualizing"

with open(visualizepath+"/sample/"+filename, "rb") as readstream:
  magic = _read32(readstream)
  rows = _read32(readstream)
  cols = _read32(readstream)
  buf = readstream.read(max_rows * cols * dt_f32.itemsize)
  data = np.frombuffer(buf, dtype=dt_f32)
  data.shape = (max_rows, cols)

#DC Filter
dc_lowcut = float(values["panel1"]["lowcut"])
dc_highcut = float(values["panel1"]["highcut"])
dc_order = 2
dc_type = "bandpass"
dc_func_type = "butter"

data_dc = np.copy(data)
data_dc.flags['WRITEABLE'] = True
for i in range(0, data.shape[1]):
    data_dc[:,i] = apply_dc_filter(data_dc[:,i], fs, dc_lowcut, dc_highcut, dc_order, dc_type, dc_func_type)

interval_num = 6
xminsec = float(values["panel1"]["xmin"])
xmaxsec = float(values["panel1"]["xmax"])
xintervalsec = np.around([(xmaxsec - xminsec)/interval_num], decimals=2)[0]
xmin = xminsec*fs
xmax = xmaxsec*fs
xinterval = int((xmax-xmin)/interval_num)

plt.figure(figsize=(5,3))
plt.ylim([int(values["panel1"]["ymin"]), int(values["panel1"]["ymax"])])
plt.ylabel("Micro-electroVolts (uVs)")
plt.xlim([int(xmin),int(xmax)])
plt.xticks(np.arange(xmin, xmax, xinterval), np.arange(xminsec, xmaxsec, xintervalsec))
plt.xlabel("Time (sec)")
plt.plot(data_dc)
plt.savefig(path + "/../public/eegimages/" + values["panel1"]["name"] + "/" + filename + ".png", rasterized=True)


#DC Filter
dc_lowcut = float(values["panel2"]["lowcut"])
dc_highcut = float(values["panel2"]["highcut"])
dc_order = 2
dc_type = "bandpass"
dc_func_type = "butter"

data_dc = np.copy(data[:,int(values["panel2"]["channel"])])
data_dc.flags['WRITEABLE'] = True
data_dc = apply_dc_filter(data_dc, fs, dc_lowcut, dc_highcut, dc_order, dc_type, dc_func_type)

interval_num = 6
xminsec = float(values["panel2"]["xmin"])
xmaxsec = float(values["panel2"]["xmax"])
xintervalsec = np.around([(xmaxsec - xminsec)/interval_num], decimals=2)[0]
xmin = xminsec*fs
xmax = xmaxsec*fs
xinterval = int((xmax-xmin)/interval_num)

cut_values = np.concatenate([np.arange(0, int(xminsec * fs)), np.arange(int(xmaxsec * fs), max_rows)])
data_dc = np.delete(data_dc, cut_values, 0)
data_wv = wvd(data_dc)[0]
if int(values["panel2"]["normalize"]) == 1:
  data_wv = normalize(data_wv, axis=1, norm='l1')

freq_min = float(values["panel2"]["ymin"])
freq_max = float(values["panel2"]["ymax"])
del_values = np.concatenate([np.arange(0, int(freq_min*rto)), np.arange(int(freq_max*rto), max_rows)])

im = data_wv.T
im = np.delete(im, del_values, 0)
im = scipy.absolute(im)

extent = [0, xmaxsec - xminsec, freq_min, freq_max]
interpolation = "lanczos"
plt.figure(figsize=(5,3))
plt.imshow(im, extent=extent, origin='lower', aspect='auto', interpolation=interpolation)
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (sec)")
#if int(values["panel2"]["normalize"]) == 1:
  #plt.clim(float(values["panel2"]["cmin"]), float(values["panel2"]["cmax"]))
plt.colorbar()
plt.savefig(path + "/../public/eegimages/" + values["panel2"]["name"]+ "/" + filename + ".png", rasterized=True)
