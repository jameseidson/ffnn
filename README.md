# ffnn
ffnn is a small and simple feed forward neural network library written in C.
It uses backpropogation and gradient descent to train user-defined networks based on user-defined data, and even features a filesystem for saving and loading training progress.

## Usage
```
git clone https://github.com/jameseidson/ffnn.git
```
Then `#include "/ffnn/ffnn.h"` in your program.

### Initialization
Networks are created from scratch using `Net_T *FFNN_init(size_t netSize, size_t *topology)`.
```
size_t topology[NUM_LYR] = { NUM_IN, NUM_HIDDEN, NUM_OUT };
Net_T *myNet = FFNN_init(NUM_LYR, topology);
```
This creates a network with `NUM_LYR` layers whose structure is defined by the topology array. 
In this case, the network has `NUM_IN` input neurons, 1 hidden layer with `NUM_HIDDEN` hidden neurons, and `NUM_OUT` output neurons.
An arbitrary number of layers can be specified, each of which can contain an arbitrary number of neurons.

Networks are freed using `void FFNN_free(Net_T *net)`.

### Saving and Loading
Networks can be written to and initialized from a binary file.
```
FILE *wfp = fopen("MyNet1.ffnn", "wb");
void FFNN_save(myNet1, *wfp);

FILE *rfp = fopen("MyNet1.ffnn", "rb");
Net_T *myNet2 = FFNN_load(rfp);
```
This creates `myNet2` based on the parameters specified in `MyNet1.ffnn`.
Note that weights and biases are preserved across saves/loads, so pre-trained networks can be distributed in this manner.

### Training
Training data is read and configured using the type `TrainSet_T` which consists of the following data members:

- `double lrnRate`

  Specifies the scale factor for gradient descent. Values >= 0.1 and <= 0.3 typically perform well, although the choice here is somewhat arbitrary.

- `size_t numEpoch`

  The maximum number of epochs to train, where an epoch is one full iteration through all provided training data.

- `size_t numElm`

  The total number of training examples in the set.

- `double **in`

  A 2D array where the first dimension corresponds to a given training example and the second corresponds to the activation of each input-layer neuron for that example.

- `double **expOut`

  Same format as `**in`, except the second dimension corresponds to the *expected* activations of the *output*-layer neurons.

Once the `TrainSet_T` has been initialized, the network is trained using `void FFNN_train(Net_T *net, TrainSet_T *tSet, FILE *nfp)`. 
That training progress will be saved to the file `*nfp` after every epoch to ensure nothing is lost if the program is terminated. 
Note that this feature can be disabled by passing `NULL` as `*nfp`.
```
FILE *rfp = fopen("MyNet.ffnn", "rb");
Net_T *myNet = FFNN_load(rfp);

void FFNN_train(myNet, tSet, *rfp);
```
This example loads `myNet` from `MyNet.ffnn`, trains it, and writes its training progress back to the original file.

Note that the user is responsible for mallocing and thus freeing the `TrainSet_T` and all associated memory.

## MNIST Reader 
Included with the library is an example implementation based on the [MNIST database of handwritten images](http://yann.lecun.com/exdb/mnist/). 

`mnistreader` provides the necessary functions to read the data from the binary MNIST files, put them into a format readable by the network, then train the network based on that data. 

The provided network savefile, `mnist.ffnn`, contains a network with 300 hidden layers that achieved 11.5% error while training.
By default, `main.c` loads this file and continues training from the user-specified MNIST files.
