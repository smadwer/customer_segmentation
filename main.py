import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create DataFrame
df = pd.read_csv('Mall_Customers.csv')

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Define KMeans
kmeans = KMeans(n_clusters=5)

# Fit and predict
df['Cluster'] = kmeans.fit_predict(X)

# Plot
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.scatter(df.loc[df['Cluster'] == i, 'Annual Income (k$)'], 
                df.loc[df['Cluster'] == i, 'Spending Score (1-100)'],
                label=f'Cluster {i}')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()





