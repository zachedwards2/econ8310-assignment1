import unittest
import patsy as pt
from pygam import LinearGAM, f, s, l
import numpy as np
import pandas as pd
import statsmodels as statsmodels
import pygam
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
import prophet
import statsmodels.tsa.holtwinters as hw

# Import your code from parent directory
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from assignment1 import model, modelFit, pred

# Run the checks

def checkNumbers(series):
  for i in series:
    if not isinstance(i, (float, int)):
      return False
  return True


def test_valid_model():
    assert (isinstance(model, statsmodels.regression.linear_model.OLS) or isinstance(model, statsmodels.tsa.arima.model.ARIMA) or isinstance(model, ES) or isinstance(model, pygam.pygam.LinearGAM) or isinstance(model, prophet.forecaster.Prophet), "Make sure that you are using a model\ncovered in class in the variable 'model'.")

