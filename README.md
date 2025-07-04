# Pixels-to-Profit

This repository contains codes for the study Pixels to Profit.

Table of Contents
Installation
Data Preparation
Normalization and Feature Extraction
Spatio-Temporal Data Generation
CVaR Calculation
Distance Matrix and Bootstrapping
Testing and Evaluation
Contributing


### Installation
Before running the code, please ensure you have installed the required packages. You can install them using pip:

```py
pip install numpy pandas matplotlib yfinance pandas_datareader yahoofinancials ta
```
Data Preparation
Download historical stock data and save them as csv files (.csv).
Place the csv files in the same directory as the script.

The script reads data for the following stocks:

Apple (AAPL) <br>
AMD (AMD) <br>
Amazon (AMZN) <br>
IBM (IBM) <br>
Oracle (ORCL) <br>
Microsoft (MSFT) <br>
Intel (INTC) <br>
Activision Blizzard (ATVI) <br>
NVIDIA (NVDA) <br>

### Normalization and Feature Extraction
The normalize_data function normalizes stock data columns (Close, Volume, gap). The do_all function calculates features and normalizes them for each stock.

```py
def prepare_for_stacking(name_df):
    selected_columns = name_df[['gap', 'Close', 'Volume']]
    arr = selected_columns.values
    arr = arr.reshape(-1, 9, 3)
    return arr
```
Then, save all the generated images inside a folder. 

### CVaR Calculation
Calculate the portfolio returns and compute CVaR.

```py
def historicalVaR(returns, alpha=5):
    return np.percentile(returns, alpha)

def historicalCVaR(returns, alpha=5):
    belowVaR = returns <= historicalVaR(returns, alpha=alpha)
    return returns[belowVaR].mean()
```

### Distance Matrix and Bootstrapping
Compute distance matrices and perform bootstrap analysis to detect stock market crashes.

```py
def bootstrap(matrix, event_list, iterations=1000):
    mid = matrix.shape[0] // 2
    training_data = matrix[:mid, :]
    testing_data = matrix[mid:, :]
    
    event_index = [num for num in index_list if num < mid]
    non_event_index = [num for num in range(0, mid) if num not in index_list]
    
    event_event_dist = []
    event_non_event_dist = []
    
    for _ in range(iterations):
        event_pairs = random.sample(event_index, k=len(event_index))
        event_pair_scores = [np.sum(np.abs(training_data[i, :] - training_data[j, :])) for i, j in zip(event_pairs[::2], event_pairs[1::2])]
        event_event_dist.append(np.mean(event_pair_scores))
        
        event_non_event_pairs = [(random.choice(event_index), random.choice(non_event_index)) for _ in range(len(event_index))]
        event_non_event_pair_scores = [np.sum(np.abs(training_data[i, :] - training_data[j, :])) for i, j in event_non_event_pairs]
        event_non_event_dist.append(np.mean(event_non_event_pair_scores))
    
    return event_event_dist, event_non_event_dist
```
### Testing and Evaluation
Classify new observations and evaluate the performance of the classifier.

```py
def classify(new_observation, event_event_mean, event_event_variance, event_nonevent_mean, event_nonevent_variance):
    prob_event_event = calculate_prob(new_observation, event_event_mean, event_event_variance)
    prob_event_nonevent = calculate_prob(new_observation, event_nonevent_mean, event_nonevent_variance)
    return 1 if prob_event_event > prob_event_nonevent else 0

def test(testing_data, mid, event_index, nonevent_index, buffer, event_event_mean, event_event_variance, event_nonevent_mean, event_nonevent_variance):
    y_true = [1 if i in event_index else 0 for i in range(testing_data.shape[0])]
    avg_distance = [np.sum(np.abs(testing_data[i] - buffer)) / len(buffer) for i in range(testing_data.shape[0])]
    y_pred = [classify(dist, event_event_mean, event_event_variance, event_nonevent_mean, event_nonevent_variance) for dist in avg_distance]
    accuracy, precision, recall, f1 = evaluate(y_true, y_pred)
    return accuracy, precision, recall, f1
```
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
