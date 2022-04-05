import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Which data you want to Download from FRED
v_d_series = [
  {'symbol': 'WILL5000INDFC', 'frequency': 'm', 'series_name': 'Wilshire 5000 Total Market Full Cap Index'},
]
INDEX_SERIES_TO_USE=0 # Index of the Dataset in 'v_d_series' to be used
DIFFERENCIATE=True    # True to compute the 1st difference of the Dataset
TRAIN_PERCENTAGE = 70

# Charts
n_lags_ACF_PACF_to_plot = 10 # How many lags to be printed in the ACF, PACF charts

# Arima Models - maximum order of 'p' and 'q'
p_max=4
q_max=4


