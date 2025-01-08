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

def test_fitted_model():
  if isinstance(modelFit, statsmodels.regression.linear_model.RegressionResultsWrapper) or isinstance(modelFit, statsmodels.tsa.arima.model.ARIMAResultsWrapper) or isinstance(modelFit, hw.results.HoltWintersResultsWrapper):
    assert (True)
  elif hasattr(modelFit, '_is_fitted'):
    if modelFit._is_fitted:
      assert (True)
  elif hasattr(modelFit, 'history'):
      assert (bool(modelFit.history))
  else:
    print("Make sure that you store your fitted model in the variable 'modelFit'.")
    assert (False)

def test_valid_pred():
  assert (len(np.squeeze(pred))==744 and checkNumbers(np.squeeze(pred))), "Make sure your prediction consists of integers\nor floating point numbers, and is a list or array of 744\nfuture predictions!"

def test_accuracy_level_1():
  dataTest = pd.read_csv(currentdir + "/testData.csv")['trips']
  rmse = sum([(np.squeeze(pred)[i]-dataTest[i])**2 for i in range(len(np.squeeze(pred)))])
  rmse = np.sqrt(rmse)*1/744

  assert rmse<220, "Your forecasts have an RMSE above 220"

def test_accuracy_level_2():
  dataTest = pd.read_csv(currentdir + "/testData.csv")['trips']
  rmse = sum([(np.squeeze(pred)[i]-dataTest[i])**2 for i in range(len(np.squeeze(pred)))])
  rmse = np.sqrt(rmse)*1/744

  assert rmse<185, "Your forecasts have an RMSE above 185"

def test_accuracy_level_3():
  dataTest = pd.read_csv(currentdir + "/testData.csv")['trips']
  rmse = sum([(np.squeeze(pred)[i]-dataTest[i])**2 for i in range(len(np.squeeze(pred)))])
  rmse = np.sqrt(rmse)*1/744

  assert rmse<171, "Your forecasts have an RMSE above 171"


