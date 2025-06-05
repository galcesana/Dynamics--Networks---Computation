# plot_information_plane.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from discrete_ib import DiscreteIB

# --- 1. Load and preprocess the 20-language color data --------------------------------------

# (a) Read the merged CSV of color‐term assignments:
df = pd.read_csv('bk_merged_unique_terms.csv')

# (b) Identify each unique chip (meaning) by its CIELAB coordinates:
chips = (
    df[['CIEL_L', 'CIEL_a', 'CIEL_b']]
    .drop_duplicates()
    .reset_index(drop=True)
)
chips['chip_id'] = chips.index
N = len(chips)

# (c) Merge back so each row of df has a chip_id:
df_merged = df.merge(chips, on=['CIEL_L', 'CIEL_a', 'CIEL_b'], how='left')

# (d) Build the union vocabulary of all terms across the 20 languages:
vocab = sorted(df_merged['term'].unique())
W = {term: idx for idx, term in enumerate(vocab)}
V = len(vocab)

# (e) Count how many languages call each chip m by each term w:
counts = np.zeros((N, V), dtype=int)
for _, row in df_merged.iterrows():
    m_id = int(row['chip_id'])
    w_id = W[row['term']]
    counts[m_id, w_id] += 1

# (f) Record how many distinct languages name each chip:
total_languages = (
    df_merged
    .groupby('chip_id')['language']
    .nunique()
    .reindex(range(N), fill_value=0)
    .values
)

# (g) Compute the empirical distribution p_empirical(w | m):
#     If a chip m appears in L_m languages, and term w appears c_{m,w} times,
#     then p_emp(w|m) = c_{m,w} / L_m.
p_emp = counts.astype(float) / np.maximum(total_languages.reshape(-1, 1), 1)

# (h) Uniform prior over chips: p(m) = 1/N
p_m = np.ones(N) / N


# --- 2. A helper to compute mutual information I(M;W) given q(w|m) --------------------------

def compute_I_MW(p_m, q_w_given_m):
    """
    Compute I(M;W) = sum_{m,w} p(m) q(w|m) log[ p(m) q(w|m) / ( p(m) p(w) ) ].
    Here:
      - p_m is shape (N,)
      - q_w_given_m is shape (N, K)
      - p(m,w) = p(m)*q(w|m)
      - p(w) = sum_m p(m,w)
    Returns a scalar (in nats). We can convert to bits by dividing by ln(2).
    """
    # Joint p(m,w):
    p_m_w = p_m[:, None] * q_w_given_m  # shape (N, K)

    # Marginal p(w):
    p_w = p_m_w.sum(axis=0)              # shape (K,)

    # Wherever p(m,w) > 0, accumulate p(m,w) * log [ p(m,w) / (p(m)*p(w)) ]:
    nonzero = p_m_w > 0
    log_term = np.zeros_like(p_m_w)
    # p(m)*p(w) = p_m[:,None] * p_w[None,:]
    denom = p_m[:, None] * p_w[None, :]
    log_term[nonzero] = np.log(p_m_w[nonzero] / denom[nonzero])
    I_nat = np.sum(p_m_w[nonzero] * log_term[nonzero])

    # Convert from nats to bits:
    I_bits = I_nat / np.log(2.0)
    return I_bits


# --- 3. Sweep over beta to approximate the IB frontier --------------------------------------

# We'll choose a set of betas, run IB with K = |vocab| (so IB can split fully),
# then compute I(M;W) for each run. In our discrete setting, U=M => I(W;U)=I(M;W).
betas = np.linspace(0.1, 10, 25)  # e.g. 25 points between 0.1 and 10
K_large = V                      # use K = |vocab| to approximate a “continuous” frontier

complexities = []
accuracies  = []

for beta in betas:
    # Instantiate and train Discrete IB:
    ib = DiscreteIB(p_m=p_m, p_w_given_m=p_emp, n_clusters=K_large, beta=beta)
    # Random initialization:
    ib.q_k_given_m = np.random.dirichlet(np.ones(K_large), size=N)
    ib.train(max_iters=300, tol=1e-6, verbose=False)

    # Extract q(k|m):
    q = ib.q_k_given_m  # shape (N, K_large)

    # Compute Complexity I(M;W):
    I_MW = compute_I_MW(p_m, q)

    # In our discrete proxy, Accuracy = I(W;M) = I(M;W):
    I_WM = I_MW

    complexities.append(I_MW)
    accuracies.append(I_WM)

# Convert lists to arrays:
complexities = np.array(complexities)
accuracies  = np.array(accuracies)


# --- 4. Plot the resulting Information Plane -------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(complexities, accuracies, '-k', linewidth=2, label='Discrete IB Frontier')
plt.scatter(complexities, accuracies, color='tab:blue', s=50)

plt.xlabel('Complexity $I(M;W)$ (bits)', fontsize=14)
plt.ylabel('Accuracy $I(W;U)$ (bits)', fontsize=14)
plt.title('Information Plane: Mixed‐Language Color Naming', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()
