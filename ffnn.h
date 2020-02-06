#ifndef FFNN_H
#define FFNN_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <assert.h>

typedef struct Net Net_T;

typedef struct TrainSet {
  size_t numEpoch;
  size_t numElm;
  double learnRate;
  double **in;
  double **expOut;
} TrainSet_T;

Net_T *FFNN_init(size_t netSize, size_t *topology);
void FFNN_feedForward(Net_T *net, double *in, double *out);
void FFNN_train(Net_T *net, TrainSet_T *tSet);
void FFNN_print(Net_T *net);
void FFNN_free(Net_T *net);

#endif
