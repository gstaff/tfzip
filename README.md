# tfzip
An experimental compression pipeline for TensorFlow.

# Goal
This project aims to implement a TensorFlow version of the Deep Compression pipeline outlined by [Han et al.](http://arxiv.org/pdf/1510.00149v3.pdf).

# Current Progress
### This project is still in the experimental stage, it is not currently ready for production use
### Pruning
- Running compression_test.py can generate uncompressed_model and compressed_model protobuf files from a simple MNIST model for comparison.
- Note that there is no difference in the protobuf file sizes until you apply gzip or some other compression tool.
- The provided zip files applied pruning of roughly ~50% of the weights in the ~267k parameter LeNet-300-100.
- Without compression both files are ~3.2 MB.
- Compression applied to the trained model yields ~2.8 MB.
- Compression applied to the pruned model yields ~2.3 MB, an appreciable improvement.
- Next steps are to implement iterative pruning and retraining to preserve accuracy; [Learning both Weights and Connections for Efficient Neural Networks](http://arxiv.org/pdf/1506.02626v3.pdf) will be a useful reference.

### Trained Quantization
- Not started.

### Huffman Coding
- Not started.
