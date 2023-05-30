# K means Clustering
K means can find clusters of data based on specified numbers of centroids. <br><br>
By starting with "random" initialized points, and through multiple steps, minimizing the average distance between each "centroid" and closest data point, cluster, of data can be found


# Find Closest Centroids
```python
    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###
    
    for i in range(X.shape[0]):
        
        distance = [] # Distance between X[i] and centroids[j]
        
        for j in range(centroids.shape[0]):
            # Compute distance
            norm = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm)
            
        idx[i] = np.argmin(distance)
        
     ### END CODE HERE ###
    
    return idx
```

# Find New Centroids
```python
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    ### START CODE HERE ###
    
    for i in range(K):
        target_points = X[idx == i]
        
        centroids[i] = np.mean(target_points, axis = 0)
    ### END CODE HERE ## 
    
    return centroids
```

# Perform K means
```python
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx
```
