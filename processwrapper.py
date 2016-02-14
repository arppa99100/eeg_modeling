import os
import sys
import time
import subprocess

#Temp fix for starting process while collecting data. 
#We should check pids to know when the process finished and run this
time.sleep(2.5)



my_dir = os.path.abspath(os.path.join(__file__,"../"))
#CollecData
print("Collecting Data...")
subprocess.call(["python3", my_dir + "/collectdata.py"])

#AddFilters
print("Adding Filters...")
subprocess.call(["python3", my_dir + "/addfilters.py"])

#CreateModel
print("Creating Model...")
subprocess.call(["python3", my_dir + "/createmodel.py"])

