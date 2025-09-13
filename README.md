# detection_clustering
## Background and purpose
The purpose of these clustering models is to identify shared behaviors of high purity.
These behaviors may need to satisfy additional criteria (e.g. size of single clusters, cluster coverage, aggregated conditions, etc.), depending on the specific use case.

## Process flow
To meet cluster purity requirements, we apply different feature preprocessing and feature selection methods iteratively until we obtain the shortest possible feature set of significant variables.
### First preprocessing
We start by removing duplicated variables, features of very low or very high variance, and highly intercorrelated features.
### First clustering
We then apply scaling on the numerical, and one hot encoding on the categorical features, merge the outputs, and apply PCA with a fixed n_components (=6). We then feed this data into a KMeans clustering model with n_clusters=8, label the original data, and examine each feature individually for significant distribution differences across clusters, dropping similarly distributed features across all clusters.
### Second preprocessing
The remaining features are applied with a Standard Scaler and OneHotEncoder. We then apply tSNE manifold compression to the output data to identify hidden complex patterns in the data. This data is then used for the second round of clustering.
### Second clustering
In the second round of clustering, our goal is to ensure that cluster coverage and size conditions are met, while keeping the clusters pure and aligned with any other user provided conditions. This requires noise removal, thus using DBSCAN. 
To facilitate this, we dynamically iterate over a set of epsilons as specified by the user, and examine the coverage of all clusters meeting the provided size and conditional criteria. The setting with the largest coverage is chosen as the final clustering model.
