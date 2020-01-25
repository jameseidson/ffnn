#ifndef FFNN_H
#define FFNN_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Weight Weight_T;
typedef struct Neuron Neuron_T;
typedef struct Layer Layer_T;
typedef struct Net Net_T;

Net_T *FF_initNet(size_t netSize, size_t *topology);

void FF_printNet(Net_T *net);

void FF_freeNet(Net_T *net);

#endif
