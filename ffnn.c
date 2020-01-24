#include "ffnn.h"

struct Weight {
    double val;
    Neuron_T *next;
};

struct Neuron {
    size_t numWeights;
    Weight_T *weights;
};

struct Layer {
    size_t layerSize;
    Neuron_T *neurons;
};

struct Net {
    size_t netSize;
    size_t *layerSizes;
    Layer_T *layers;
};

Network_T *FF_defineTopology(size_t netSize, size_t *topology) {
    /*TODO*/
}

void FF_free(Network_T *net) {
    /*TODO*/
}
