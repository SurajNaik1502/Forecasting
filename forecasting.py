import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import OneHotEncoder
from darts.utils.timeseries_generation import datetime_attribute_timeseries

# Load dataset
df = pd.read_csv('/Users/rajs/Downloads/ledger_history_1.csv')

# Preprocessing
df.rename(columns={'column7655': 'date', 'column8142': 'ledger', 'dr_cr': 'value'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])

# Sort the dataset by date
df.sort_values(by='date', inplace=True)

# Fill missing values in 'dr_cr' and other columns
df['value'].fillna(0, inplace=True)
df.fillna(0, inplace=True)  # Fill missing values in other columns

# One-hot encoding for ledger column and other categorical columns
encoder = OneHotEncoder(sparse=False)
ledger_encoded = encoder.fit_transform(df[['ledger']])
ledger_encoded_df = pd.DataFrame(ledger_encoded, columns=encoder.get_feature_names_out(['ledger']))
df = pd.concat([df, ledger_encoded_df], axis=1)

# Normalize the 'dr_cr' values using Darts Scaler
scaler = Scaler()
ts_series = TimeSeries.from_dataframe(df, time_col='date', value_cols='value')
ts_series_scaled = scaler.fit_transform(ts_series)

# Covariates: Adding other columns as covariates
covariates = ['column7656', 'column6076', 'column51784']  # List other relevant columns
for col in covariates:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)  # Ensure numeric covariates

# Normalize covariates
covariates_ts = TimeSeries.from_dataframe(df, time_col='date', value_cols=covariates)
covariates_ts_scaled = scaler.fit_transform(covariates_ts)

# Additional date-time covariates (year, month, day, day of week)
year_series = datetime_attribute_timeseries(df['date'], attribute="year")
month_series = datetime_attribute_timeseries(df['date'], attribute="month")
day_series = datetime_attribute_timeseries(df['date'], attribute="day")
weekday_series = datetime_attribute_timeseries(df['date'], attribute="weekday")

# Stack all covariates together
covariates_final = covariates_ts_scaled.stack(year_series).stack(month_series).stack(day_series).stack(weekday_series)

# Create and train the RNN model (multivariate with covariates)
model = RNNModel(
    input_chunk_length=12,
    output_chunk_length=1,
    model="LSTM",
    n_epochs=100,
    random_state=42,
    likelihood=None
)

# Train model using both the target series and the covariates
model.fit(series=ts_series_scaled, past_covariates=covariates_final)

# Predict the next 30 days for 'dr_cr' considering covariates
forecast = model.predict(n=30, series=ts_series_scaled, past_covariates=covariates_final)

# Inverse scale the forecast to get the actual values
forecast = scaler.inverse_transform(forecast)

# Display forecasted values
print(forecast)

