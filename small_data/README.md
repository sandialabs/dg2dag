# Pairwise Examples

This directory contains tests and experiments related to pairwise data examples using different initializations and methods.

## Tests

### `two_node_tests.ipynb`

In this notebook, we explore how different initializations of the Directed Acyclic Graph (DAG) matrix affect accuracy.

- **Equal Weights Initialization:**
  - Initializing the DAG matrix with equal weights for both directions (`[[0, 1], [1, 0]]`) results in an accuracy ranging from **48% to 54%**, depending on the random number generator (RNG) seed.
  
- **Correct/("Biased") Weights Initialization:**
  - Using the "correct" weights (`[[0, 1], [0.1, 0]]` for `direction=1` and reversed non-zero entries for `direction=-1`) achieves **100% accuracy**.
  - This high accuracy is maintained even when the difference in magnitudes is less pronounced. For example, initializing with `[[0, 1], [0.9, 0]]` still yields **100% accuracy**.
  
- **CHD Edge Weights Initialization:**
  - Initializing the DAG matrix with Computational Hypergraph Discovery (CHD) edge weights results in an accuracy of **51.58%**, which is lower than the accuracy obtained when basing the direction solely on edge weights.

### CHD Module

- For more details, see [this issue](https://github.com/TheoBourdais/ComputationalHypergraphDiscovery/issues/6).
- **CHD Data for Scalar-Pair Examples:**
  - The `.npy` files in the [`chd_data`](./chd_data/) directory contain arrays of shape `(N, 3)`, where:
    - **First Column:** Corresponds to the dataset ID from the [`data`](./data/) directory.
    - **Second and Third Columns:** Represent the edge weights for the directions "Column_1" → "Column_2" and "Column_2" → "Column_1", respectively.
  - Note that not all datasets were loaded because some are not scalar-to-scalar mappings and require algorithm revisions.
