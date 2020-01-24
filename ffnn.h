#ifndef FFNN_H
#define FFNN_H

#include <stdio.h>

typedef struct Weight Weight_T;
typedef struct Neuron Neuron_T;
typedef struct Layer Layer_T;
typedef struct Network Net_T;

Net_T *FF_defineTopology(size_t netSize, size_t *topology);
void FF_free(Network_T *net);

#endif
