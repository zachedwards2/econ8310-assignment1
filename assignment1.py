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
    s(2, n_splines=30, spline_order=3)        # day of year
)

modelFit = model.fit(X_train, y_train)

pred_dates = pd.date_range(start='2019-01-01 00:00:00', end='2019-01-31 23:00:00', freq='h')
pred_df = pd.DataFrame()
pred_df['datetime'] = pred_dates
pred_df['hour'] = pred_df['datetime'].dt.hour
pred_df['day_of_week'] = pred_df['datetime'].dt.dayofweek
pred_df['day_of_year'] = pred_df['datetime'].dt.dayofyear

X_pred = pred_df[['hour','day_of_week','day_of_year']].values
pred = modelFit.predict(X_pred)
fig = go.Figure()


fig.add_trace(go.Scatter(
    x=train_2018['datetime'],
    y=y_train,
    mode='lines',
    name='Actual - 2018',
    line=dict(color='red')
))

fig.add_trace(go.Scatter(
    x=pred_df['datetime'],
    y=pred,
    mode='lines',
    name='Predicted - Jan 2019',
    line=dict(color='blue')
))

fig.update_layout(
    title='Hourly NYC Taxi Trips: 2018 Actual vs 2019 Predicted January',
    xaxis_title='Datetime',
    yaxis_title='Number of Trips',
    xaxis=dict(rangeslider=dict(visible=True)),
    template='plotly_white'
)

fig.show()

