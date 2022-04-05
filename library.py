import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# INPUT VALUES
from input_values import *

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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ENV
from dotenv import load_dotenv
load_dotenv()
FRED_API_KEY = os.environ["FRED_API_KEY"]
FRED_API_KEY = '25a15d2a4ae98406fe78567ba4344b76' if FRED_API_KEY == None else FRED_API_KEY



def lib_get_fred_history(symbol, frequency, series_name=None, download_even_if_present=None):
  from os.path import exists
  from fredapi import Fred

  assert frequency in ['d', 'w', 'm', 'q'], "Error, frequency must be in 'd'(daily) | 'w'(weekly) | 'm'(monthly) | 'q'(quarterly)"

  series_name = series_name if type(series_name) == str else ''
  download_even_if_present = True if download_even_if_present == True else False

  path_to_file = 'data/' + frequency + '_' + symbol + '.csv'
  file_exists = exists(path_to_file)
  if file_exists == True and download_even_if_present == False:
    return

  fred = Fred(api_key=FRED_API_KEY)
  try:
    df = fred.get_series(symbol, frequency=frequency)#, aggregation_method='eop')
    if df.empty:
      print('    ', symbol,"is empty in FRED!")
      return []
  except:
    print('    ', symbol," doesnt exist in FRED!")
    return []

  df.dropna(axis=0, inplace=True)
  df = df.reset_index()
  df.to_csv(path_to_file, header=['time', 'value'], index=False)
  print('    created new file ==>', path_to_file)




def download_fred_data(v_d_series):
  for x in v_d_series:
    lib_get_fred_history( symbol=x['symbol'],
                          frequency=x['frequency'],
                          series_name=x['series_name']
                        )








def get_initial_data(d_SERIE, DIFFERENCIATE):
  path_file_name = 'data/' + d_SERIE['frequency'] + '_' + d_SERIE['symbol'] + '.csv'
  df = pd.read_csv(path_file_name)#, index_col='time')
  df['time'] = pd.to_datetime(df['time']).dt.date
  df['time_dt'] = pd.to_datetime(df['time']).dt.date
  df.sort_values(by=['time'])
  df = df.set_index('time')
  df.index = pd.DatetimeIndex(df.index).to_period('M')

  
  if DIFFERENCIATE == True:
    # Take the log difference to make data stationary
    df['value'] = np.log(df['value'])
    df['value'] = df['value'].diff()
    df = df.drop(df.index[0])

  return df





def get_train_test_data(df):
  IDX = int(df.shape[0] * TRAIN_PERCENTAGE/100)
  df_train = df.iloc[:IDX, :]
  df_test = df.iloc[IDX:, :]
  # print('     df_train.shape[0]', df_train.shape[0], '     df_test.shape[0]', df_test.shape[0])  
  return df_train, df_test





def plot_basic_chart(df, series_name):

  fig,ax = plt.subplots(figsize=(12,7))

  years = matplotlib.dates.YearLocator(base = 5)
  years_fmt = matplotlib.dates.DateFormatter('%Y')

  try:
    ax.plot(df['time'], df['value'], color ='green')
  except:
    ax.plot(df['time_dt'], df['value'], color ='green')
  ax.xaxis.set_major_locator(years)
  ax.xaxis.set_major_formatter(years_fmt)

  plt.title(series_name)
  plt.xlabel("Year")
  #plt.ylabel(series_name)
  plt.xticks(rotation=0)
  plt.legend()
  plt.show()


def plot_resiuals(v_hd):
  plt.plot(v_hd)
  plt.show()



def plot_ACF(v_hd, n_lags=None, label=''):
  from statsmodels.graphics import tsaplots
  #plot autocorrelation function
  #fig = tsaplots.plot_acf(v_hd, lags=50)
  n_lags = n_lags if type(n_lags) == int else n_lags_ACF_PACF_to_plot
  fig = tsaplots.plot_acf(v_hd, lags=n_lags, color='g', title='ACF ' + str(label)) # title='Autocorrelation function '
  plt.show()


def plot_PACF(v_hd, n_lags=None, label=''):
  from statsmodels.graphics import tsaplots
  n_lags = n_lags if type(n_lags) == int else n_lags_ACF_PACF_to_plot
  fig = tsaplots.plot_pacf(v_hd, lags=n_lags, color='g', title='PACF ' + str(label)) # title='Partial Autocorrelation function '
  plt.show()



def OLD_plot_ACF(v_hd, n_lags=None):
  # https://www.statology.org/autocorrelation-python/
  # https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_acf.html
  n_lags = n_lags if type(n_lags) == int else n_lags_ACF_PACF_to_plot
  plot_acf(v_hd, n_lags) # es. n_lags=50
  pyplot.show()



def OLD_plot_PACF(v_hd, n_lags=None):
  n_lags = n_lags if type(n_lags) == int else n_lags_ACF_PACF_to_plot
  fig = plot_pacf(v_hd, n_lags)
  plt.show()

  #pyplot.show()




def get_if_series_is_stationary(v_hd, print_result=None):
  #def dickey_fuller(v_hd, print_result=None):
  '''
    Since the p-value is not less than .05, we fail to reject the null hypothesis.
    This means the time series is non-stationary. In other words, it has some time-dependent structure and does not have constant variance over time.
  '''
  from statsmodels.tsa.stattools import adfuller
  #perform augmented Dickey-Fuller test
  DFU = adfuller(v_hd)
  t_stat = round(DFU[0],4)
  p_value = round(DFU[1],4)
  print('        t_stat:', t_stat, '    p_value', p_value)

  if p_value < 0.05:
    if print_result:
      print('   We Accept H0, the series is Stationary  |  p_value is < 0.05 (', p_value,')')
    return True
  else:
    if print_result:
      print('   We Reject H0, the series is Non-Stationary  |  p_value is > 0.05 (', p_value,')')
    return False




def get_v_pq_tuples(p_max, q_max):
  from itertools import product

  v_p = range(0, p_max+1, 1)
  v_q = range(0, q_max+1, 1)
  
  # Create a list with all possible combination of parameters
  parameters = product(v_p, v_q)
  parameters_list = list(parameters)
  parameters_list = [x for x in parameters_list if x != (0,0)]
  return parameters_list


def find_best_model(v_hd, p_max, q_max):
  from classes import c_arima_result

  v_pq = get_v_pq_tuples(p_max, q_max)

  v_arima_results = []
  for i,pq in enumerate(v_pq):
    print('      ', i+1, '/', len(v_pq), '  |   ', pq)
    v_arima_results.append(c_arima_result(v_hd, pq))

  R = v_arima_results
  df_results = pd.DataFrame({  'pq': [x.pq for x in R],   'aic': [x.aic for x in R],    'bic': [x.bic for x in R] })
  print('AIC\n',df_results.sort_values(by=['aic', 'pq']))
  print('BIC\n',df_results.sort_values(by=['bic', 'pq']))



def do_ARMA_and_residuals_analysis(v_hd, pq: tuple, is_arma=None):
  from statsmodels.tsa.arima.model import ARIMA

  if is_arma == False:
    pdq = (pq[0], 0, pq[1])
  else:
    pdq = (pq[0], 1, pq[1])

  model = ARIMA(v_hd, order=pdq) #  (0,0,1)   (p, d, q)
  results = model.fit()
  # print('model fit summary ==>', results.summary())

  #print('results.aic', results.aic, '      results.bic', results.bic)

  #plt.plot(results.fittedvalues, color='red'); plt.plot(v_hd, color='blue'); plt.show()

  residuals = pd.DataFrame(results.resid)
  #plt.plot(residuals, color='red'); plt.plot(v_hd, color='blue'); plt.show()
  #OLD_plot_ACF(residuals, label='(Residuals)')
  #a=888/0
  plot_ACF(residuals, label='(Residuals)')
  plot_PACF(residuals, label='(Residuals)')

  
def model_forecast(model_fit):
  # one-step out-of sample forecast
  forecast = model_fit.forecast()[0]




def lib_get_if_model_is_significantly_better(df_test, v_hd_forecast_model, v_hd_forecast_mean):
  ''' Test Diebold-Mariano'''
  from lib_dm_test import dm_test
  DM_results = dm_test(df_test.value.tolist(), v_hd_forecast_model.tolist(), v_hd_forecast_mean.tolist())
  DM_t_stat, DM_p_value = round(DM_results[0],4), round(DM_results[1],4)
  msg = 'Diebold-Mariano Test   |   t_stat:', DM_t_stat, '   p_value:', DM_p_value
  is_better = True if DM_p_value < 0.05 else False
  if is_better == True:
    print(msg + ' | ==> Model A is significantly better than B.')
  else:
    print(msg + ' | ==> Model A is NOT significantly better than B.')
  return is_better
