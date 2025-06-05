import pandas as pd
from sklearn.cluster import KMeans

# 1. Load
df = pd.read_csv('bk_merged_unique_terms.csv')
chips = df[['CIEL_L','CIEL_a','CIEL_b']].drop_duplicates().reset_index(drop=True)
chips['chip_id'] = chips.index

# 2. Merge to get (chip_id, language, term)
df_merged = df.merge(chips, on=['CIEL_L','CIEL_a','CIEL_b'], how='left')

# 3. k-Means on CIELAB (choose K = 12, for example)
K = 12
kmeans = KMeans(n_clusters=K, random_state=42).fit(chips[['CIEL_L','CIEL_a','CIEL_b']])
chips['cluster'] = kmeans.labels_

# 4. For each cluster, pick the majority term
cluster_labels = {}
for k in range(K):
    cluster_chips = chips.loc[chips['cluster']==k, 'chip_id'].values
    sub_df = df_merged[df_merged['chip_id'].isin(cluster_chips)]
    best_term = sub_df['term'].value_counts().idxmax()
    cluster_labels[k] = best_term

# 5. Assign to each chip its “mixed” label
chips['mixed_term'] = chips['cluster'].map(cluster_labels)

# 6. Export
chips[['CIEL_L','CIEL_a','CIEL_b','mixed_term']].to_csv('mixed_color_kmeans.csv', index=False)
