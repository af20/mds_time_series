import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# TIME
from datetime import date, datetime
from dateutil.relativedelta import relativedelta as rd
import time

# DATA
import numpy as np
import pandas as pd

# PLOT
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot

# MODELS
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


class c_arima_result:
  def __init__(self, v_hd, pq):
    # model_type sempre ARMA perchè differenzio negli input_values (DIFFERENCIATE)
    model_type = 'ARMA'

    assert model_type in ['ARMA', 'ARIMA'], 'Error (model type)'
    if model_type == 'ARMA':
      pdq=(pq[0], 0, pq[1])
    else:
      pdq=(pq[0], 1, pq[1])

    self.pq=pq
    self.pdq=pdq
    self.model_type=model_type
    
    self.model = ARIMA(v_hd, order=pdq) #  (0,0,1)   (p, d, q)
    self.results = self.model.fit(method_kwargs={"warn_convergence": False})

    #model.fit(start_params=[0, 0, 0, 1])

    self.residuals = pd.DataFrame(self.results.resid)
    self.aic = round(self.results.aic,2)
    self.bic = round(self.results.bic,2)
    # REDDIT SPIEGAZ SU AIC   BIC  MIGLIORI  https://www.reddit.com/r/AskStatistics/comments/5ydt2c/if_my_aic_and_bic_are_negative_does_that_mean/

  def _plot_ACF(self, label):
    from library import plot_ACF
    plot_ACF(self.residuals, label=label)

  def _plot_PACF(self, label):
    from library import plot_PACF
    plot_PACF(self.residuals, label=label)

  def get_one_step_forecast(self):
    # one-step out-of sample forecast
    # array containing the forecast value, the standard error of the forecast, and the confidence interval information.
    FC = self.results.forecast()
    forecast = round(FC[0],6)
    return forecast
  


  def get_multi_step_forecast(self, v_hd_train, v_hd_test):#start_index, end_index):

    v_hd_forecast = pd.Series(data=[], index=[])
    v_forecasts = []
    for i in range(len(v_hd_test)):
      model = ARIMA(v_hd_train, order=self.pdq) #  (0,0,1)   (p, d, q)
      results = model.fit(method_kwargs={"warn_convergence": False})
      forecast = results.forecast()#.predict(n_periods=1)#start=start_index, end=end_index)

      ser = pd.Series(data=[v_hd_test.iloc[i]], index=[v_hd_test.index[0]])
      v_hd_train = pd.concat([v_hd_train, ser])#forecast])
      
      forecast_value = round(forecast.values[0], 6)
      v_forecasts.append(forecast_value)

      serF = pd.Series(data=[forecast_value], index=[v_hd_test.index[i]])
      v_hd_forecast = pd.concat([v_hd_forecast, serF])#forecast])


      print('   ',i, '   |   ', forecast.index[0], '      ', forecast_value)
    return v_hd_forecast
    # https://machinelearningmastery.com/make-sample-forecasts-arima-python/
    # https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7




  def get_multi_step_forecast_mean(self, v_hd_train, v_hd_test):#start_index, end_index):

    v_hd_forecast = pd.Series(data=[], index=[])

    v_forecasts = []
    for i in range(len(v_hd_test)):
      my_mean = round(np.mean(v_hd_train), 6)
      ser = pd.Series(data=[v_hd_test.iloc[i]], index=[v_hd_test.index[i]])
      v_hd_train = pd.concat([v_hd_train, ser])#forecast])
      v_forecasts.append(my_mean)

      serF = pd.Series(data=[my_mean], index=[v_hd_test.index[i]])
      v_hd_forecast = pd.concat([v_hd_forecast, serF])#forecast])

      #print('   ',i, '   |   ', my_mean)
    return v_hd_forecast
    # https://machinelearningmastery.com/make-sample-forecasts-arima-python/
    # https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7
