#!pip install pygam
import pandas as pd
from pygam import LinearGAM, s
import plotly.graph_objects as go


train = pd.read_csv(
    "https://raw.githubusercontent.com/dustywhite7/econ8310-assignment1/main/assignment_data_train.csv"
)


train_2018 = train[train['year'] == 2018].copy()
train_2018['datetime'] = pd.to_datetime(train_2018[['year','month','day','hour']])
train_2018['day_of_week'] = train_2018['datetime'].dt.dayofweek
train_2018['day_of_year'] = train_2018['datetime'].dt.dayofyear

X_train = train_2018[['hour','day_of_week','day_of_year']].values
y_train = train_2018['trips'].values


model = LinearGAM(
    s(0, n_splines=24, spline_order=3) +      # hour of day
    s(1, n_splines=7, spline_order=3) +       # day of week
    s(2, n_splines=30, spline_order=3)        # day of year (trend)
)
