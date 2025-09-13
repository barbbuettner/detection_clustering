import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans

from scipy.stats import chi2_contingency
  
class DetectionClustering:
  def __init__(self, min_cluster_size:int, max_cluster_size:int, constant_threshold:float, min_eps:float, max_eps:float, n_bins_eps:int, cluster_exclusion_condition=False, cluster_exclusion_agg_map={}):
    self.min_cluster_size = min_cluster_size
    self.max_cluster_size = max_cluster_size
    self.constant_threshold = constant_threshold
    self.min_eps = min_eps
    self.max_eps = max_eps
    self.n_bins_eps = n_bins_eps
    self.cluster_exclusion_condition = cluster_exclusion_condition
    self.cluster_exclusion_agg_map = cluster_exclusion_agg_map

  def fit_predict(self, data: pd.DataFrame):
    if hasattr(self, 'dbscan'):
      print("already fitted")
      return self
    data_tsne = self.prepare_clustering(data)
    labels = self.perform_dynamic_clustering(data_tsne)
    return labels
  
  def prepare_clustering(self, data: pd.DataFrame):
    
    if data.empty:
      return data
    
    data = self._remove_duplicated_columns(data)
    data = self._remove_near_constants(data)
    data = self._remove_near_uniques(data)
    data = self._remove_correlated(data)
    
    self.num_cols, self.cat_cols = self._feature_importance_analysis_unsupervised(data)
    
    #numerical handling
    if len(self.num_cols) > 0:
      scaler = StandardScaler()
      x_num = scaler.fit_transform(data[self.num_cols].fillna(0))
      self.scaler = scaler
    else:
      x_num = None
    
    #categorical handling
    if len(self.cat_cols) > 0:
      ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
      x_cat = ohe.fit_transform(data[self.cat_cols].fillna(''))
      self.ohe = ohe
    else:
      x_cat = None
      
    if isinstance(x_num, np.ndarray):
      if isinstance(x_cat, np.ndarray):
        x = np.concatenate([x_num, x_cat], axis = 1)
      else:
        x = x_num
    else:
      if isinstance(x_cat, np.ndarray):
        x = x_cat
      else:
        print("Error during preprocessing. No features remaining for clustering")
        return
    tsne = TSNE()
    x_tsne = tsne.fit_transform(x)
    self.tsne = tsne
    
    data_tsne = pd.DataFrame(x_tsne)
    data_tsne.columns = ['tsne_1', 'tsne_2']
    data_tsne.index = data.index
    return data_tsne
  
  def perform_dynamic_clustering(self, data: pd.DataFrame):
    
    if data.empty:
      return np.array([])
    
    epses = np.linspace(self.min_eps, self.max_eps, self.n_bins_eps)
    coverages = []

    
    print(f"Starting DBSCAN with min_cluster_size={self.min_cluster_size}, max_cluster_size={self.max_cluster_size}. Trying to find highest coverage.")
    
    for eps in epses:
      dbscan = DBSCAN(eps=eps, min_samples = self.min_cluster_size)
      labels = dbscan.fit_predict(data)
      unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
      coverage = sum(counts[counts <= self.max_cluster_size])
      coverages.append(coverage)
    best_eps = epses[np.argmax(coverages)]
    print(f"Best eps found: {best_eps}, with coverage: {max(coverages)}")
    best_dbscan = DBSCAN(eps=best_eps, min_samples = self.min_cluster_size)
    labels = best_dbscan.fit_predict(data)
    self.dbscan = best_dbscan
    unique_labels = np.unique(labels)
    print(f"Found {len(unique_labels[unique_labels != -1])} clusters before validation")
    
    return labels
  
  
  def filter_valid_clusters(self, data: pd.DataFrame, group_name: str) -> pd.DataFrame:
    print(f"\nStarting cluster validation with {len(data)} records")

    if 'cluster' not in data.columns:
      print("Need to define a cluster variable. Returning unchanged data.")
      return data

    data_agg = data.groupby('cluster').agg(self.cluster_exclusion_agg_map).reset_index()

    print(f"Calculated metrics for {len(data_agg)} clusters")
    data_agg.columns = ['cluster'] + list(self.cluster_exclusion_agg_map.keys())

    exclusion_clusters = data_agg[self.cluster_exclusion_condition]['cluster'].tolist()

    valid_clusters = [cluster for cluster in list(set(data['cluster'])) if cluster != -1 and cluster not in exclusion_clusters]

    print(f"Found {len(valid_clusters)} valid clusters after validation")


    if len(valid_clusters) == 0:
      print("No clusters meeting the criteria found.")
      return None

    data = data[data['cluster'].isin(valid_clusters)]
    data = data.merge(data_agg, on='cluster', how='left')
    print(f"Final output contains {len(data)} records in {len(valid_clusters)} clusters")
    return data
    
    
  def _remove_duplicated_columns(self, data: pd.DataFrame):
    data_T = data.T
    data_T.drop_duplicates(inplace = True)
    return data_T.T
  
  def _remove_near_constants(self, data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[:, data.apply(lambda x: (x.value_counts() / len(data)).max() < self.constant_threshold)]
  
  def _remove_near_uniques(self, data: pd.DataFrame):
    num_cols = []
    cat_cols = []
    for col in data.columns:
      try:
        data[col] = data[col].astype(float)
        num_cols.append(col)
      except:
        cat_cols.append(col)
    columns_to_keep = pd.Series(True, index = data.columns)
    for col in cat_cols:
      if (data[col].value_counts() / len(data)).max() <= 1 - self.constant_threshold:
        columns_to_keep[col] = False
    return data.loc[:, columns_to_keep]
  
  def _remove_correlated(self, data: pd.DataFrame) -> pd.DataFrame:
    num_cols = data.select_dtypes(include=['integer', 'float']).columns
    data_num = data[num_cols]
    cat_cols = data.select_dtypes(exclude=['integer', 'float']).columns
    data_cat = data[cat_cols]
    
    #numerical correlation
    data_corr_num = data_num.corr().abs()
    upper = data_corr_num.where(np.triu(np.ones(data_corr_num.shape), k=1).astype(bool))
    to_drop_num = [col for col in upper.columns if any(upper[col] > 0.8)]
    data_num = data_num.drop(to_drop_num, axis=1)
    
    #categorical correlation
    to_drop_cat = []
    for i in range(len(cat_cols)):
      for j in range(i+1, len(cat_cols)):
        col1, col2 = cat_cols[i], cat_cols[j]
        cardinality_col1 = data[col1].nunique()
        cardinality_col2 = data[col2].nunique()
        estimated_cells = cardinality_col1 * cardinality_col2
        # ignore if size of estimated cells is too large to avoid breakdown
        if estimated_cells < 100000:
          try:
            _, p_value, _, _ = chi2_contingency(pd.crosstab(data[col1], data[col2]))
            if p_value >= 0.05:
              to_drop_cat.append(col2)
          except:
            pass
        else:
          if cardinality_col1 > 1000:
            to_drop_cat.append(col1)
          if cardinality_col2 > 1000:
            to_drop_cat.append(col2)
    to_drop_cat = list(set(to_drop_cat))
    data_cat = data_cat.drop(to_drop_cat, axis = 1)
    
    return pd.concat([data_num, data_cat], axis = 1)
  
  def _feature_importance_analysis_unsupervised(self, data: pd.DataFrame):
    num_cols = data.select_dtypes(include=['integer', 'float']).columns
    data_num = data[num_cols]
    cat_cols = data.select_dtypes(exclude=['integer', 'float']).columns
    data_cat = data[cat_cols]
    
    data_num = data_num.fillna(-1)
    data_num = data_num.astype(float)
    data_cat = data_cat.fillna('')
    data_cat = data_cat.astype(str)
    
    #categorical handling
    ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False, max_categories=int(data.shape[0]*0.05))
    data_cat_ohe = ohe.fit_transform(data_cat)
    data_cat_ohe = pd.DataFrame(data_cat_ohe)
    data_cat_ohe.columns = ohe.get_feature_names_out(cat_cols)
    data_cat_ohe.index = data.index
    
    #numerical handling
    scaler = StandardScaler()
    data_num_scaled = scaler.fit_transform(data_num)
    data_num_scaled = pd.DataFrame(data_num_scaled)
    data_num_scaled.columns = [f"{nc}_scaled" for nc in num_cols]
    data_num_scaled.index = data.index
    
    #merged handling
    pca = PCA(n_components=6)
    data_preprocessed = pd.concat([data_cat_ohe, data_num_scaled], axis = 1)
    data_preprocessed = pca.fit_transform(data_preprocessed)
    data_preprocessed = pd.DataFrame(data_preprocessed)
    data_preprocessed.columns = [f"pca_component_{i+1}" for i in range(6)]
    data_preprocessed.index = data.index
    
    #kmeans for analysis
    km = KMeans()
    km.fit(data_preprocessed)
    #looking for numerical and categorical features where distribution within a cluster is vastly different to overall distribution
    relevant_features = []
    
    for lbl in list(set(km.labels_)):
      mask = km.labels_ == lbl
      data_num_lbl = data_num.loc[mask]
      scaler_lbl = StandardScaler()
      scaler_lbl.fit(data_num_lbl)
      relevant_features += [col for col, lbl_mean, orig_mean, orig_var in zip(num_cols, scaler_lbl.mean_, scaler.mean_, scaler.var_) if np.abs(lbl_mean - orig_mean) > 2 * orig_var]
      
      data_cat_lbl = data_cat.loc[mask]

      for col in cat_cols:
        # Create a contingency table
        contingency_table = pd.crosstab(data_cat[col], columns='count')
        contingency_table_lbl = pd.crosstab(data_cat_lbl[col], columns='count')

        # Ensure both tables have the same categories
        all_categories = set(contingency_table.index) | set(contingency_table_lbl.index)
        for cat in all_categories:
          if cat not in contingency_table.index:
            contingency_table.loc[cat] = 0
          if cat not in contingency_table_lbl.index:
            contingency_table_lbl.loc[cat] = 0

        contingency_table = contingency_table.sort_index()
        contingency_table_lbl = contingency_table_lbl.sort_index()

        # Perform chi-square test
        _, p_value, _, _ = chi2_contingency([contingency_table, contingency_table_lbl])

        # If p-value is less than a threshold (e.g., 0.05), consider the feature relevant
        if p_value < 0.05:
          relevant_features.append(col)

    relevant_features = list(set(relevant_features))
    relevant_num = [col for col in num_cols if col in relevant_features]
    relevant_cat = [col for col in cat_cols if col in relevant_features]
    
    return relevant_num, relevant_cat
