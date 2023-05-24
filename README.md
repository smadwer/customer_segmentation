# Customer Segmentation using K-Means Clustering
This code performs customer segmentation using the K-Means clustering algorithm. It uses the pandas, sklearn, and matplotlib libraries in Python.

## Prerequisites
 - Python 3.0 or higher
 - pandas
 - scikit-learn (sklearn)
 - matplotlib


## Installation
Clone the repository or download the code files.

Install the required libraries using pip:


## Usage
Ensure that the dataset file "Mall_Customers.csv" is in the same directory as the code file.

Import the necessary libraries:

```python
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
```

Read the dataset file:

```python
dataset = pd.read_csv('Mall_Customers.csv')
```

Select the features to be used for clustering:

```python
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
```

Create an instance of the K-means clustering algorithm:
    
```python
kmeans = KMeans(n_clusters=5)
```

Fit the model to the data:

```python
df['Cluster'] = kmeans.fit_predict(X)
```

Visualize the clusters:

```python
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.scatter(df.loc[df['Cluster'] == i, 'Annual Income (k$)'], df.loc[df['Cluster'] == i, 'Spending Score (1-100)'], label=f'Cluster {i}')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

Run the code and observe the plotted clusters of customers based on their annual income and spending score.
