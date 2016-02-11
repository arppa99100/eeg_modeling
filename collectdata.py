import os;
import sys;
import numpy as np;
from readbytes import _read8, _read32, _read_chunks

#File Paths
samplespath = "./samples"
curatedpath = "./curated"

#Data Types
dt_i8 = np.dtype("<u1")
dt_i32 = np.dtype("<u4")
dt_f32 = np.dtype("<f4")

#Vars
input_count = 0
num_samples = 0
num_channels = 8
max_samples = 470
max_presamples = 120


#Open Files to use
temp_presample_output = open(curatedpath+"/raw-presamples-nometa", "wb")
temp_sample_output = open(curatedpath+"/raw-samples-nometa", "wb")    
temp_input_output = open(curatedpath+"/raw-inputs-nometa", "wb")

samples_folders = sorted(os.listdir(samplespath))
for samplefolder in samples_folders:
  path = samplespath+"/"+samplefolder
  sample_list = sorted(os.listdir(samplespath+"/"+samplefolder))
            
  #Get Status
            
  #Get Inputs
  with open(path+"/input", "rb") as bytestream:
    num_inputs = _read8(bytestream)
    buf = bytestream.read(num_inputs * dt_i8.itemsize)
    temp_input_output.write(buf)
  input_count += num_inputs

  #Get Samples
  allsamples = sorted(os.listdir(path+"/sample"))
  for sample in allsamples:
    num_samples += 1
    with open(path+"/sample/"+sample, "rb") as bytestream:
      #Check magic is 2049
      magic = _read32(bytestream)
      cols = _read32(bytestream)
      rows = _read32(bytestream)
      buf = bytestream.read(max_samples * cols * dt_f32.itemsize) #homogeneous size
      temp_sample_output.write(buf)
    with open(path+"/presample/"+sample, "rb") as bytestream:
      #Check magic is 2049
      magic = _read32(bytestream)
      cols = _read32(bytestream)
      rows = _read32(bytestream)
      buf = bytestream.read(max_presamples * cols * dt_f32.itemsize) #homogeneous size
      temp_presample_output.write(buf)

temp_presample_output.close()
temp_sample_output.close()
temp_input_output.close()

metadata = [2049, num_samples, max_samples, num_channels]
with open(curatedpath+"/raw-samples-nometa", "rb") as readstream, open(curatedpath+"/raw-samples", "wb") as writestream:
  writestream.write(np.array(metadata, dtype=dt_i32))
  for chunk in _read_chunks(readstream):
    writestream.write(chunk)
os.remove(curatedpath+"/raw-samples-nometa")

metadata = [2049, num_samples, max_presamples, num_channels]
with open(curatedpath+"/raw-presamples-nometa", "rb") as readstream, open(curatedpath+"/raw-presamples", "wb") as writestream:
  writestream.write(np.array(metadata, dtype=dt_i32))
  for chunk in _read_chunks(readstream):
    writestream.write(chunk)
os.remove(curatedpath+"/raw-presamples-nometa")

metadata = [2049, input_count]
with open(curatedpath+"/raw-inputs-nometa", "rb") as readstream, open(curatedpath+"/raw-inputs", "wb") as writestream:
  writestream.write(np.array(metadata, dtype=dt_i32))
  writestream.write(readstream.read())
os.remove(curatedpath+"/raw-inputs-nometa")
