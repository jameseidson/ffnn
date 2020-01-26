#ifndef FFNN_H
#define FFNN_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

typedef struct Weight Weight_T;
typedef struct Neuron Neuron_T;
typedef struct Layer Layer_T;
typedef struct Net Net_T;

typedef struct Data {
    double *in;
    double *expout;
} Data_T;


Net_T *FFNN_init(size_t netSize, size_t *topology);
void FFNN_free(Net_T *net);

double *FFNN_feedForward(Net_T *net, double *inputs);

#endif
