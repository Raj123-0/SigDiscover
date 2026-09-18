# Algorithms

## Non-Negative Matrix Factorization (NMF)

Given a mutation count matrix $M$ (shape $n \times c$), we want to find non-negative matrices $S$ (signature profiles, $k \times c$) and $A$ (activities, $n \times k$) such that $M \approx AS$, minimizing the Frobenius norm $||M - AS||_F$.

We use multiplicative update rules (Lee & Seung, 2001):
$$ A \leftarrow A \odot \frac{MS^T}{ASS^T + \epsilon} $$
$$ S \leftarrow S \odot \frac{A^TM}{A^TAS + \epsilon} $$

## Rank Selection

To select the optimal number of signatures $k$, we run NMF multiple times for each $k$ and compute:
1. **Stability**: Mean of the maximum cosine similarities between matched signatures across NMF replicates.
2. **Reconstruction Error**: $1 - \text{cosine\_similarity}(M, AS)$ averaged across samples.

We select the $k$ that maximizes the trade-off score: $\text{stability} - \text{reconstruction\_error}$, provided stability is at least $0.80$.

## COSMIC Assignment

Discovered signatures are matched to COSMIC reference signatures using cosine similarity. A match is assigned if the similarity is $\ge 0.8$; otherwise, it is labeled "Novel".
