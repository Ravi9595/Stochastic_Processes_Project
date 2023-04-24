import pandas as pd
import numpy as np

def model(data, states, transition_matrix, initial_probabilities, n_days):
    # Predict future states
    current_state = np.random.choice(states, p=initial_probabilities)
    predicted_states = []  # Initialize as an empty list
    for i in range(n_days):  # Change the loop range to (n_days)
        current_index = states.index(current_state)
        next_state = np.random.choice(states, p=transition_matrix[current_index])
        predicted_states.append(next_state)
        current_state = next_state

    # Convert predicted states to stock price returns
    predicted_returns = []
    for i in range(1, len(predicted_states)):
        current_state = states.index(predicted_states[i-1])
        next_state = states.index(predicted_states[i])
        diff = data['Price'].iloc[-1] * (transition_matrix[current_state, next_state] - 1)
        predicted_returns.append(diff)

    # Compute predicted stock prices
    predicted_prices = [data['Price'].iloc[-1]]
    for i in range(len(predicted_returns)):
        predicted_prices.append(predicted_prices[-1] + predicted_returns[i])

    return predicted_prices

def get_state(return_value):
    if return_value > 0.01:
        return 'Up'
    elif return_value < -0.01:
        return 'Down'
    else:
        return 'Neutral'

def initialize(data, states):
    # Estimate transition matrix and initial state probabilities
    num_states = len(states)
    transition_counts = np.zeros((num_states, num_states))
    initial_counts = np.zeros(num_states)
    current_state = states.index(get_state(data['Return'].iloc[0]))
    initial_counts[current_state] += 1

    for i in range(len(data)-1):
        next_state = states.index(get_state(data['Return'].iloc[i+1]))
        transition_counts[current_state, next_state] += 1
        current_state = next_state
        initial_counts[current_state] += 1

    transition_matrix = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
    initial_probabilities = initial_counts / np.sum(initial_counts)
    return transition_matrix, initial_probabilities

def split_data(input_df, test_size=5):
    # Split data into train and test sets
    train_df = input_df.iloc[:-test_size]
    test_df = input_df.iloc[-test_size:]
    return train_df, test_df


def mean_absolute_error(y_true, y_pred):
    error = np.abs(y_true - y_pred)
    mae = np.mean(error)
    return mae
