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
  size_t size;
  double **input;
  double **expOut;
} Data_T;

Net_T *FFNN_init(size_t netSize, size_t *topology);
void FFNN_feedForward(Net_T *net, double *in, double *out);
void FFNN_train(Net_T *net, double **in, double **expOut, size_t numElem, size_t epoch);
void FFNN_free(Net_T *net);

void FFNN_print(Net_T *net);

#endif
