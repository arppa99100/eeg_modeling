import numpy as np
from readbytes import _read8, _read32, _read_chunks

def readdata(pathname):
  dt_f32 = np.dtype("<f4")
  with open(pathname, "rb") as bytestream:
    #Check magic is 2049
    magic = _read32(bytestream)
    samples = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows*cols*samples*dt_f32.itemsize)
    dataset = np.frombuffer(buf, dtype=dt_f32)
    dataset.shape = (samples, rows, cols)
  return(dataset)

def readdata2(pathname):
  dt_f32 = np.dtype("<f4")
  dataset = []
  with open(pathname, "rb") as bytestream:
    #Check magic is 2049
    magic = _read32(bytestream)
    num_tests = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    for i in range(0, num_tests):
      samples = _read32(bytestream)
      buf = bytestream.read(rows*cols*samples*dt_f32.itemsize)
      dataset_temp = np.frombuffer(buf, dtype=dt_f32)
      dataset_temp.shape = (samples, rows, cols)
      dataset.append(dataset_temp)
  return(dataset)

def readlabels(pathname):
  dt_i8 = np.dtype("<u1")
  with open(pathname, "rb") as bytestream:
    #Check magic is 2049
    magic = _read32(bytestream)
    num_inputs = _read32(bytestream)
    buf = bytestream.read(num_inputs*dt_i8.itemsize)
    labels = np.frombuffer(buf, dtype=dt_i8)
    labels.shape = (num_inputs)
  return(labels)

def readlabels2(pathname):
  dt_i8 = np.dtype("<u1")
  labels = []
  with open(pathname, "rb") as bytestream:
    #Check magic is 2049
    magic = _read32(bytestream)
    num_tests = _read32(bytestream)
    for i in range(0, num_tests):
      num_inputs = _read32(bytestream)
      buf = bytestream.read(num_inputs*dt_i8.itemsize)
      labels_temp = np.frombuffer(buf, dtype=dt_i8)
      labels_temp.shape = (num_inputs)
      labels.append(labels_temp)
  return(labels)

def writedata(pathname, data):
  dt_i32 = np.dtype("<u4")
  metadata = [2049]
  for i in range(0, len(data.shape)):
    metadata.append(data.shape[i])
  with open(pathname, "wb") as writestream:
    writestream.write(np.array(metadata, dtype=dt_i32))
    writestream.write(data)
