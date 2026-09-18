# API Reference

## sigdiscover.extraction.nmf

```python
def nmf_mutational_signatures(M, n_signatures, n_iterations, tolerance, seed, init_method)
```
Extract mutational signatures using Non-Negative Matrix Factorization.

## sigdiscover.matrices.sbs

```python
def build_sbs96_matrix(mutations, genome)
```
Build a 96-channel SBS mutation count matrix.

## sigdiscover.assignment.similarity

```python
def assign_to_cosmic(discovered_signatures, cosmic_signatures, cosmic_names, threshold)
```
Assign discovered signatures to COSMIC signatures using cosine similarity.
