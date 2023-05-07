import pandas as pd
from datetime import datetime
import numpy as np

def date_to_num(date_obj):
    datetime_obj = datetime(date_obj.year, date_obj.month, date_obj.day)
    return datetime_obj.timestamp()

def num_to_date(timestamp):
    datetime_obj = datetime.fromtimestamp(timestamp)
    return datetime_obj.date()

def preprocess(df):
    # drop all the columns except id and pickup_datetime
    df = df[['id', 'pickup_datetime']]
    
    # convert the pickup_datetime column to date only format
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime']).dt.date
    
    # sort the dataframe by pickup_datetime
    df.sort_values(by=['pickup_datetime'], inplace=True)
    
    # group the DataFrame by the 'pickup_datetime' column and get the size of each group
    date_counts = df.groupby('pickup_datetime').size()
    
    # convert the resulting Series to a dictionary
    date_counts_dict = date_counts.to_dict()
    
    # create a new dataframe from the dictionary
    df_poisson = pd.DataFrame.from_dict(date_counts_dict, orient='index', columns=['count'])
    df_poisson.reset_index(inplace=True)
    df_poisson.columns = ['date', 'count']
    
    # convert the date column to a timestamp
    df_poisson['timestamp'] = df_poisson['date'].apply(date_to_num)
    
    return df_poisson

def evaluate_error(y_true, y_pred, metric='mse'):
    """
    Calculate the Mean Squared Error (MSE) or Mean Absolute Error (MAE) of the model.

    Parameters:
    y_true (numpy.array): The true values of the target variable.
    y_pred (numpy.array): The predicted values of the target variable.
    metric (str): The error metric to compute. Either 'mse' or 'mae'. Default is 'mse'.

    Returns:
    float: The error value.
    """
    if metric == 'mse':
        error = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        error = np.mean(np.abs(y_true - y_pred))
    else:
        raise ValueError("Invalid metric. Use 'mse' or 'mae'.")

    return error

def r_squared(y_true, y_pred):
    """
    Calculate the R-squared (R2) value of the model.

    Parameters:
    y_true (numpy.array): The true values of the target variable.
    y_pred (numpy.array): The predicted values of the target variable.

    Returns:
    float: The R-squared value.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)  # Sum of squared residuals
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    r2 = 1 - (ss_res / ss_tot)  # R-squared value

    return r2
