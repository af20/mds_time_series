import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datetime import date, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


from input_values import *
from classes import *
from library import *


# Download data from FRED
download_fred_data(v_d_series)

# Get the series to use and adjust its Name
d_SERIE = v_d_series[INDEX_SERIES_TO_USE]
series_name_oth = d_SERIE['series_name']
if DIFFERENCIATE == True:
  d_SERIE['series_name'] = d_SERIE['series_name'] + ' (Return)'
series_name = d_SERIE['series_name']


# Get DataFrame
df_oth = get_initial_data(d_SERIE, not DIFFERENCIATE)
df = get_initial_data(d_SERIE, DIFFERENCIATE)

''' Plot charts (Series, ACF, PACF) '''
# plot_basic_chart(df_oth, series_name_oth)
# plot_basic_chart(df, series_name)
# plot_ACF(v_hd, label=' - ' + series_name)
# plot_PACF(v_hd, label=' - ' + series_name)
df.drop(columns=['time_dt'], inplace=True)

''' Train e Test'''
df_train, df_test = get_train_test_data(df)
v_hd = df['value']
v_hd_train = df_train['value']
v_hd_test = df_test['value']


'''Stationarity Test (Dickey Fuller test)'''
#is_series_stationary = get_if_series_is_stationary(v_hd, print_result=True) #my_model = 'ARMA' if is_series_stationary == True else 'ARIMA'
#print('   is_series_stationary:', is_series_stationary)

''' Find Best Model'''
find_best_model(v_hd, p_max, q_max)


''' Choose Best Model'''
pq=(0,1)
print('   The Best Model has pq =', pq)
Best_Model = c_arima_result(v_hd_train, pq)


''' Plot ACF, PACF'''
# title
my_model_final = 'ARIMA' if DIFFERENCIATE == True else 'ARMA'
str_pq = str((pq[0], 1, pq[1])) if DIFFERENCIATE == True else str(pq)
label = ' | ' + my_model_final + str_pq + ' Residuals'

# plot
Best_Model._plot_ACF(label)
Best_Model._plot_PACF(label)
forecast = Best_Model.get_one_step_forecast()
print('forecast', forecast)


#print('PRE v_hd_train', v_hd_train)
LEN_1 = len(v_hd_train)

v_hd_forecast_model = Best_Model.get_multi_step_forecast(v_hd_train, v_hd_test)
v_hd_forecast_mean = Best_Model.get_multi_step_forecast_mean(v_hd_train, v_hd_test)
#print(df)
LEN_2 = len(v_hd_train)


''' PLOT Forecast'''
df_test_1 = df_test+1
wealth_origin = df_test_1.cumprod()

# Model
v_hd_forecast_model_1 = v_hd_forecast_model +1
wealth_model = v_hd_forecast_model_1.cumprod()

# Mean
v_hd_forecast_mean_1 = v_hd_forecast_mean +1
wealth_mean = v_hd_forecast_mean_1.cumprod()


''' PLOT dati percentuali'''
fig, ax = plt.subplots()
ax.title.set_text('Data Forecast (returns)')
ax.plot(df_test.values, label='Data')
#ax.plot([])
ax.plot(v_hd_forecast_model.values, label='Model')
ax.plot(v_hd_forecast_mean.values, label='Mean')
ax.legend()
plt.show()

''' PLOT dati originali'''
fig, ax = plt.subplots()
ax.title.set_text('Data Forecast')
ax.plot(wealth_origin.values, label='Data')
ax.plot(wealth_model.values, label='Model')
ax.plot(wealth_mean.values, label='Mean')
ax.legend()
plt.show()



''' ERRORS'''
df_temp = pd.DataFrame({'original': df_test.value.tolist(), 'model': v_hd_forecast_model.values, 'mean': v_hd_forecast_mean.values}, index=df_test.index)
v_error_model = df_temp['model'] - df_temp['original']
v_error_mean = df_temp['mean'] - df_temp['original']
v_error_model.values
df_errors = pd.DataFrame({'model': v_error_model.tolist(), 'mean': v_error_mean.tolist()}, index=df_test.index)

fig, ax = plt.subplots()
ax.title.set_text('Forecast Errors')
ax.plot(df_errors['model'].tolist(), label='Errors - Model')
ax.plot(df_errors['mean'].tolist(), label='Errors - Mean')
ax.legend()
plt.show()


''' MSE'''
from sklearn.metrics import mean_squared_error
from math import sqrt
rMSE_model = round(sqrt(mean_squared_error(df_test, v_hd_forecast_model)),4)
rMSE_mean = round(sqrt(mean_squared_error(df_test, v_hd_forecast_mean)),4)
print('rMSE_model', rMSE_model, '    rMSE_mean', rMSE_mean)
# Riporta i valori da variazioni percentuali in originali
# Rappresentare GRAFICAM previsioni e errori di previsione
# Calcolare MSE

''' Test Diebold-Mariano'''
is_better = lib_get_if_model_is_significantly_better(df_test, v_hd_forecast_model, v_hd_forecast_mean)


