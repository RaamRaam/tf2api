import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import math
import inspect
import functools
import time

from shutil import copyfile
from google.colab import drive


def mount():
  drive.mount('/content/drive')


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer
@timer
def checkTFversion(func):
  if str(tf.__version__)[:1]!='2':  
    # print('your version of tensorflow is ', tf.__version__)    
    raise AssertionError('\n your version of tensorflow is ' + tf.__version__ + '\n Please execute the following to upgrade to 2.0 \n pip uninstall tensorflow \n pip install -U --pre tensorflow \n')    
    return 'F'
  else:
    @functools.wraps(func)
    def wrapper_checkTFversion(*args, **kwargs):
      value = func(*args, **kwargs)
      return value
    return wrapper_checkTFversion

@timer
def install_tf2():
  if str(tf.__version__)[:1]!='2': 
    os.system('pip uninstall tensorflow')
    # os.system('pip install -U --pre tensorflow-gpu')
    os.system('pip install tensorflow')
    os.system('pip install tf-nightly-2.0-preview')
    os.system('pip install tensorflow-gpu')
    print('TF 2.0 installed')
  else:
    print('TF 2.0 already installed')