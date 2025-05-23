# Sparse Encoder Model

This is a SparseEncoder model with the following components:

1. Base Transformer: bert-base-uncased
2. Pooling: Mean pooling
3. CSRSparsity: 768 -> 3072 dimensions, k=256

## Usage

```python
from sentence_transformers import SparseEncoder

model = SparseEncoder("sparse_encoder_model")
embeddings = model.encode(["Your text here"])
```
