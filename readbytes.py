import numpy as np

def _read8(bytestream):
  dt = np.dtype(np.uint8).newbyteorder("<")
  return np.frombuffer(bytestream.read(1), dtype=dt)[0]

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder("<")
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def _read_chunks(file_object, chunk_size=1024*1024):
  #Chunk size at 1Mb
  while True:
    data = file_object.read(chunk_size)
    if not data:
      break
    yield data
