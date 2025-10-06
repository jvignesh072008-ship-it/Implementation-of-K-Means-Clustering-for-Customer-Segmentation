# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Data Acquisition: Import customer data from 'Mall_Customer.csv' containing demographic and behavioral attributes.

2. Feature Selection: Extract numerical features (Age, Annual Income, Spending Score) for clustering analysis.

3. Data Normalization: Standardize selected features to ensure equal weighting in the distance calculations.

4. Optimal Cluster Determination: Apply Elbow Method by plotting WCSS against k (1-10) to identify the optimal number of clusters.

5. Model Training: Execute K-Means algorithm with the optimal k value and random centroids initialization.

6. Customer Assignment: Assign each customer to the nearest centroid's cluster based on Euclidean distance.

7. Visualization & Analysis: Plot the resulting segments and analyze cluster characteristics to derive marketing insights.
 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: VIGNESH J 
RegisterNumber: 25014705
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')
print("First 5 rows:")
print(data.head())

# Select features for clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Print cluster sizes
print("\nCluster sizes:")
print(data['Cluster'].value_counts())

# Visualize clusters (Income vs Spending Score)
plt.figure(figsize=(10, 6))
for i in range(5):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(
        cluster_data['Annual Income (k$)'],
        cluster_data['Spending Score (1-100)'],
        label=f'Cluster {i}'
    )

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

## Output:
![K Means Clustering for Customer Segmentation](sam.png)
<img width="661" height="155" alt="Screenshot 2025-10-06 202538" src="https://github.com/user-attachments/assets/c21bdeb6-d878-4d7d-b480-a6e49fb37e16" />
<img width="961" height="587" alt="Screenshot 2025-10-06 202550" src="https://github.com/user-attachments/assets/c54ba428-3a39-43cf-9e2f-2c680cbcdfd0" />
<img width="363" height="184" alt="Screenshot 2025-10-06 202610" src="https://github.com/user-attachments/assets/960a900d-16be-4401-85a8-e64bb86c6629" />
<img width="1237" height="689" alt="Screenshot 2025-10-06 202623" src="https://github.com/user-attachments/assets/682f9b98-817a-4124-9578-84d1fcb71723" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
