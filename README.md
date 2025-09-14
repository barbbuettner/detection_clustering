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
### Notes on design choices
#### t-SNE for clustering:
Although PCA is used earlier in the process for dimensionality reduction and potential inverse transformations, we found that t-SNE produces significantly better cluster separation for our specific business context. Therefore, t-SNE is applied before the second round of clustering to capture non-linear relationships and hidden patterns that PCA alone might miss.
#### Coverage as eps selection criterion:
When choosing the optimal eps for DBSCAN, we prioritize maximum coverage of valid clusters. This means that the selected eps maximizes the number of data points assigned to clusters that satisfy both size and user-defined validity conditions. This approach ensures that clusters are not only pure but also include as much relevant data as possible, which is crucial in practical applications where coverage is important.
Alternative metrics (e.g., silhouette score) could be applied, but in our business context, coverage proved to be the most effective measure.

## Usage / Instructions
```
clustering = DetectionClustering()
clustering.fit_predict(X)
```
This will call the clustering class and execute all aforementioned process flows in the backend. The labels to data X can then be called by ```clustering.labels``` attribute. 
### Hyperparameters
```min_cluster_size``` : denotes the smallest allowed size for the clusters. Will be used in DBSCAN directly, prevents noise from forming clusters.

```max_cluster_size``` : denotes the largest allowed size for the clusters. Will be examined after the clusters are built, to weed out clusters that are too large to be meaningful. 

```constant_threshold``` : denotes the rate of variability that is required. A constant_threhsold of 0.95 means that the largest contributor must contribute to at least 5% of the data (otherwise its variability is too high), and no more than 95% of the data (otherwise its variability will be too high).

```min_eps, max_eps, n_bins_eps``` : These parameters control the DBSCAN grid search. Ultimately, we choose the eps within the range defined by these three parameters that yields the highest coverage.

```cluster_feature_map, cluster_agg_map, cluster_exclusion_condition``` : these three parameters define the exact criteria to render a cluster invalid based on user input. cluster_feature_map is required to derive a new feature based on an input feature on a row-level basis (for example: cluster_feature_map['low_value'] = lambda df: (df['value'] < 10).astype(int) ). cluster_agg_map is required to define the cluster level aggregations (for example: 'low_value' : np.mean, 'country' : np.nunique). We will obtain aggregated value by cluster through this application. Naming convention is {colname}_{func} (i.e. low_value_mean, country_nunique). cluster_exclusion_conditon specifies the exact boolean conditions to render a cluster invalid, such as '(low_value_mean > 0.2) & (country_nunique > 5)', will exclude clusters where both conditions apply, '(low_value_mean > 0.2) | (country_nunique > 5)', will exclude clusters where at least one condition applies.
