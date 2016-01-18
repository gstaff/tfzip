# tfzip
An experimental compression pipeline for TensorFlow.

# Goal
This project aims to implement a TensorFlow version of the Deep Compression pipeline outlined by [Han et al.](http://arxiv.org/pdf/1510.00149v3.pdf)

# Current Progress
### This project is still in the experimental stage, it is not currently ready for production use
### Pruning
- Initial results:
  - Running `compression_test.py` can generate uncompressed_model and compressed_model protobuf files from a simple MNIST model for comparison.
  - Note that there is no difference in the protobuf file sizes until you apply `gzip` or some other compression tool.
  - The provided protobuf files were generated using a LeNet-300-100 model trained for MNIST digit classification.
  
|              | Parameters | Parameter Compression | Protobuf Size | Protobuf Size (gzipped) | Protobuf Compression | Accuracy |
|--------------|------------|-----------------------|---------------|-------------------------|----------------------|----------|
| Uncompressed | ~267k      |                       | 3.2MB         | ~2.8MB                  |                      | 98.19%   |
| Compressed   | ~27k       | **~10x**              | 3.2MB         | ~490kB                  | **~6x**              | 97.31%   |
|              |            |                       |               |                         |                      |          |
- Next steps:
  - Improve accuracy preservation and increase compression ratio.
    - [Learning both Weights and Connections for Efficient Neural Networks](http://arxiv.org/pdf/1506.02626v3.pdf) suggests ~12x parameter compression without loss of accuracy can be obtained for this architecture.
  - Refactor compression logic so that it can be more easily applied to other networks.
  
### Trained Quantization
- Not started.

### Huffman Coding
- Not started.
