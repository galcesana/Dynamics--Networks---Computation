import pandas as pd
import numpy as np
from discrete_ib import DiscreteIB  # Ensure discrete_ib.py is in the same folder

def main():
    # 1. Load data
    df = pd.read_csv('bk_merged_unique_terms.csv')

    # 2. Identify unique chips (meanings)
    chips = df[['CIEL_L', 'CIEL_a', 'CIEL_b']].drop_duplicates().reset_index(drop=True)
    chips['chip_id'] = chips.index
    N = len(chips)

    # 3. Merge so each row has a chip_id
    df_merged = df.merge(chips, on=['CIEL_L', 'CIEL_a', 'CIEL_b'], how='left')

    # 4. Build union vocabulary W
    vocab = sorted(df_merged['term'].unique())
    W = {term: idx for idx, term in enumerate(vocab)}
    V = len(vocab)

    # 5. Count how many languages call each chip m by term w
    counts = np.zeros((N, V), dtype=int)
    for _, row in df_merged.iterrows():
        m_id = int(row['chip_id'])
        w_id = W[row['term']]
        counts[m_id, w_id] += 1

    # total_languages[m] = number of distinct languages naming chip m
    total_languages = (
        df_merged.groupby('chip_id')['language']
                 .nunique()
                 .reindex(range(N), fill_value=0)
                 .values
    )

    # 6. Compute p_empirical(w | m)
    p_emp = counts.astype(float) / np.maximum(total_languages.reshape(-1, 1), 1)

    # 7. Uniform prior over chips
    p_m = np.ones(N) / N

    # 8. Set IB hyperparameters: number of clusters (K) and beta
    K_target = 10     # desired number of clusters
    beta_value = 10.0  # trade-off parameter

    print(f"Running IB with K_target = {K_target}, beta = {beta_value}")

    # 9. Instantiate DiscreteIB
    ib = DiscreteIB(p_m=p_m, p_w_given_m=p_emp, n_clusters=K_target, beta=beta_value)

    # 10. Randomly initialize q_k_given_m rather than uniform
    ib.q_k_given_m = np.random.dirichlet(np.ones(K_target), size=N)

    # 11. Run IB training
    ib.train(max_iters=500, tol=1e-6, verbose=True)

    # 12. Extract q(k | m)
    q_k_given_m = ib.q_k_given_m

    # 13. Hard-assign each chip to one cluster
    cluster_of_chip = np.argmax(q_k_given_m, axis=1)

    # 14. Print how many chips per cluster
    counts_per_cluster = pd.Series(cluster_of_chip).value_counts().sort_index()
    print("Cluster sizes (hard argmax):")
    for k in range(K_target):
        print(f"  Cluster {k}: {counts_per_cluster.get(k, 0)} chips")

    # 15. Label clusters by most popular term among their chips
    cluster_labels = []
    for k in range(K_target):
        member_chip_ids = np.where(cluster_of_chip == k)[0]
        if len(member_chip_ids) == 0:
            # If IB ended with an empty cluster, choose the rarest term overall
            rarest_term = min(vocab, key=lambda w: counts[:, W[w]].sum())
            cluster_labels.append(rarest_term)
            continue

        sub_df = df_merged[df_merged['chip_id'].isin(member_chip_ids)]
        best_term = sub_df['term'].value_counts().idxmax()
        cluster_labels.append(best_term)

    # 16. Create final mapping: chip_id -> mixed term
    final_mapping = {m_id: cluster_labels[cluster_of_chip[m_id]] for m_id in range(N)}

    # 17. Attach mixed term to chips and save CSV
    chips['mixed_term'] = chips['chip_id'].map(final_mapping)
    output_csv = 'mixed_color_naming.csv'
    chips.to_csv(output_csv, index=False)
    print(f"Mixed-color-naming CSV saved to: {output_csv}")
    print(chips.head(10))

if __name__ == "__main__":
    main()