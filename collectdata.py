import os;
import sys;
import numpy as np;
from readbytes import _read8, _read32, _read_chunks

samplespath = "./samples"
#Merge data into one file
dt_i8 = np.dtype("<u1")
dt_i32 = np.dtype("<u4")
dt_f32 = np.dtype("<f4")

#Our samples usually have 1150 +- 2, to make sure we are consistent, we remove 20
input_count = 0
num_samples = 0
num_channels = 8
max_samples = 1130

#Prepare sample reading
samples_folders = sorted(os.listdir(samplespath))
if "sampledata" in samples_folders: samples_folders.remove("sampledata")
if "sampledata-nometa" in samples_folders: samples_folders.remove("sampledata-nometa")
if "sampleinput" in samples_folders: samples_folders.remove("sampleinput")
if "sampleinput-nometa" in samples_folders: samples_folders.remove("sampleinput-nometa")
temp_sample_output = open(samplespath+"/../"+"sampledata-nometa", "wb")
temp_input_output = open(samplespath+"/../"+"sampleinput-nometa", "wb")
#Read every sample into file
for samplefolder in samples_folders:
  sample_list = sorted(os.listdir(samplespath+"/"+samplefolder))
  #Read Input
  sample_list.remove("input")
  with open(samplespath+"/"+samplefolder+"/"+"input", "rb") as bytestream:
    #No Magic, fix
    num_inputs = _read8(bytestream)
    buf = bytestream.read(num_inputs*dt_i8.itemsize)
    temp_input_output.write(buf)
    input_count += num_inputs

  #Read Status, fix
  sample_list.remove("status")
  
  #Read Samples
  for sample in sample_list:
    num_samples+=1
    with open(samplespath+"/"+samplefolder+"/"+sample, "rb") as bytestream:
      #Check magic is 2049
      magic = _read32(bytestream)
      cols = _read32(bytestream)
      rows = _read32(bytestream)
      buf = bytestream.read(rows*cols*dt_f32.itemsize)
      temp_sample_output.write(buf)
temp_sample_output.close()
temp_input_output.close()

#Write metadata to samples file
metadata = [2049, num_channels, max_samples, num_samples]
with open(samplespath+"/../"+"sampledata-nometa", "rb") as readstream, open(samplespath+"/../"+"sampledata", "wb") as writestream:
  writestream.write(np.array(metadata, dtype=dt_i32))
  for chunk in _read_chunks(readstream):
    writestream.write(chunk)
os.remove(samplespath+"/../"+"sampledata-nometa")

#Write metadata to input file
metadata = [2049, input_count]
with open(samplespath+"/../"+"sampleinput-nometa", "rb") as readstream, open(samplespath+"/../"+"sampleinput", "wb") as writestream:
  writestream.write(np.array(metadata, dtype=dt_i32))
  writestream.write(readstream.read())
  #for chunk in _read_chunks(readstream):
    #writestream.write(chunk)
os.remove(samplespath+"/../"+"sampleinput-nometa")

#Read samples
#with open(samplespath+"/"+"sampledata", "rb") as bytestream:
  #Check magic is 2049
  #magic = _read32(bytestream)
  #cols = _read32(bytestream)
  #rows = _read32(bytestinput)
  #samples = _read32(bytestream)
  #buf = bytestream.read(rows*cols*samples*dt_f32.itemsize)
  #data = np.frombuffer(buf, dtype=dt_f32)
  #data.shape = (samples,rows,cols)

#Read inputs
#with open(samplespath+"/"+"sampleinput", "rb") as bytestream:
  #Check magic is 2049
  #magic = _read32(bytestream)
  #num_inputs = _read32(bytestream)
  #buf = bytestream.read(num_inputs*dt_i8.itemsize)
  #data = np.frombuffer(buf, dtype=dt_i8)
