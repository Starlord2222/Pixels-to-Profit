# Pixels-to-Profit

This repository contains code to normalize stock market data, generate spatio-temporal data representations, compute the Conditional Value at Risk (CVaR), and perform bootstrap analysis for stock market crash detection.

Table of Contents
Installation
Data Preparation
Normalization and Feature Extraction
Spatio-Temporal Data Generation
CVaR Calculation
Distance Matrix and Bootstrapping
Testing and Evaluation
Contributing
License
Installation
Before running the code, ensure you have the required packages installed. You can install them using pip:

.. code-block:: console
pip install numpy pandas matplotlib yfinance pandas_datareader yahoofinancials ta

Data Preparation
Download historical stock data and save them as Excel files (.xlsx).
Place the Excel files in the same directory as the script.
The script reads data for the following stocks:

Apple (AAPL)
AMD (AMD)
Amazon (AMZN)
IBM (IBM)
Oracle (ORCL)
Microsoft (MSFT)
Intel (INTC)
Activision Blizzard (ATVI)
NVIDIA (NVDA)

Normalization and Feature Extraction
The normalize_data function normalizes stock data columns (Close, Volume, gap). The do_all function calculates features and normalizes them for each stock.

python
Copy code
def normalize_data(df):
    min_val = df.min()
    max_val = df.max()
    return (df - min_val) / (max_val - min_val)

def do_all(df):
    df['gap'] = df['Open '] - df['Close'].shift(1)
    df = df.dropna()
    df['Close'] = normalize_data(df['Close'])
    df['Volume'] = normalize_data(df['Volume'])
    df['gap'] = normalize_data(df['gap'])
    return df
Spatio-Temporal Data Generation
Convert the normalized data into a 9x9x3 spatio-temporal representation and save them as images.

python
Copy code
def prepare_for_stacking(name_df):
    selected_columns = name_df[['gap', 'Close', 'Volume']]
    arr = selected_columns.values
    arr = arr.reshape(-1, 9, 3)
    return arr

output_folder = "path/to/output/folder"
for index, array in enumerate(final_array):
    fig = plt.figure()
    plt.imshow(array, interpolation="nearest", cmap="viridis")
    plt.axis('off')
    fig.savefig(Path(output_folder, f"9cross9_close_{index}.jpg"), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
CVaR Calculation
Calculate the portfolio returns and compute CVaR.

python
Copy code
def historicalVaR(returns, alpha=5):
    return np.percentile(returns, alpha)

def historicalCVaR(returns, alpha=5):
    belowVaR = returns <= historicalVaR(returns, alpha=alpha)
    return returns[belowVaR].mean()
Distance Matrix and Bootstrapping
Compute distance matrices and perform bootstrap analysis to detect stock market crashes.

python
Copy code
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
Testing and Evaluation
Classify new observations and evaluate the performance of the classifier.

python
Copy code
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
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
