# SequenceMemory
Sequence memory with a neural network

The network consists of one-hot vectors where transition is defined as a weight matrix between vectors.  Input and output are associated with the latent network with I/O weight matrices.  Since the latent space is one-hot vectors, it can memorize patterns as many as the number of the neurons (vector dimension).  The neurons have decaying activation and the least active one may be 'recycled.'

The idea is similar to the competitive queuing model.

The images used for testing the main program can be found at [another repository](https://github.com/rondelion/AEPredictor)'s Sample folder, which are of the MNIST dataset.