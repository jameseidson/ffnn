#include "ffnn.h"

typedef struct Layer Layer_T;
typedef struct Neuron Neuron_T;
typedef struct Weight Weight_T;

struct Net {
  size_t numLyr;
  Layer_T *layers;
};

struct Layer {
  size_t numNrn;
  Neuron_T *neurons;
};

struct Neuron {
  double activ;
  double curErr;
  size_t numWgt;
  Weight_T *weights;
};

struct Weight {
  double val;
  Neuron_T *next;
};

double FFNN_backprop(Net_T *, const double *, const double *, double);
void FFNN_gradDescent(Net_T *, double);
double FFNN_sig(double);
double FFNN_dsig(double);

const uint32_t MAGICNUM = 0;

Net_T *FFNN_init(size_t netSize, size_t *topology) {
  /* validate input */
  assert(netSize != 0);
  for (size_t i = 0; i < netSize; i++) {
    assert(topology[i] != 0);
  }
  srand(time(NULL));
  /* allocate and initialize a new network */
  Net_T *net = malloc(sizeof(Net_T));
  assert(net != NULL);
  net->numLyr = netSize;
  net->layers = malloc(net->numLyr * sizeof(Layer_T));
  assert(net->layers != NULL);

  /* allocate and initialize layers */
  for (size_t i = 0; i < netSize; i++) {
    Layer_T *curLyr = &net->layers[i];
    curLyr->numNrn = topology[i];
    /* add extra neuron in layer to be bias */
    if (i != 0 && i != netSize - 1) {
      curLyr->numNrn++;
    }

    curLyr->neurons = malloc(curLyr->numNrn * sizeof(Neuron_T));
    assert(curLyr->neurons != NULL);
  }

  /* allocate and initialize weights */
  for (size_t i = 0; i < netSize; i++) {
    for (size_t j = 0; j < net->layers[i].numNrn; j++) {
      Neuron_T *curNrn = &net->layers[i].neurons[j];
      curNrn->activ = 1;
      curNrn->curErr = 0.0;
      if (i != netSize - 1) {
        curNrn->numWgt = topology[i + 1];
        curNrn->weights = malloc(curNrn->numWgt * sizeof(Weight_T));
        assert(curNrn->weights != NULL);

        /* point kth weight in curr layer to kth neuron in next layer */
        for (size_t k = 0; k < curNrn->numWgt; k++) {
          curNrn->weights[k].next = &net->layers[i + 1].neurons[k];
          curNrn->weights[k].val = (((double)rand() / RAND_MAX) - 0.5) * 2;
        }
      } else {
        /* dont want to alloc weights in last layer */
        curNrn->numWgt = 0;
        curNrn->weights = NULL;
      }
    }
  }

  return net;
}

void FFNN_save(Net_T *net, char *netFile) {
  FILE *nfp = fopen(netFile, "wb");
  assert(nfp != NULL);

  fwrite(&MAGICNUM, sizeof(MAGICNUM), 1, nfp);

  /* write num layers and layer sizes */
  fwrite(&net->numLyr, sizeof(net->numLyr), 1, nfp);
  for (size_t i = 0; i < net->numLyr; i++) {
    size_t curLyrSize = net->layers[i].numNrn;
    if (i != 0 && i != net->numLyr - 1) {
      curLyrSize--;
    }
    fwrite(&curLyrSize, sizeof(curLyrSize), 1, nfp);
  }

  /* TODO */

  fclose(nfp);
}

Net_T *FFNN_load(char* netFile) {
  FILE *nfp = fopen(netFile, "rb");
  assert(nfp != NULL);

  uint32_t inMagicNum = 0;
  fread(&inMagicNum, sizeof(inMagicNum), 1, nfp);
  assert(inMagicNum == MAGICNUM);

  /* read num layers and layer sizes */
  size_t numLyr = 0;
  fread(&numLyr, sizeof(numLyr), 1, nfp);

  size_t *topology = malloc(sizeof(size_t) * numLyr);
  fread(topology, sizeof(size_t), numLyr, nfp);

  Net_T *net = FFNN_init(numLyr, topology);
  free(topology);

  /* TODO */

  fclose(nfp);

  return net;
}

void FFNN_feedForward(Net_T *net, double *in, double *out) {
  /* feed in input data */
  for (size_t i = 0; i < net->layers[0].numNrn; i++) {
    net->layers[0].neurons[i].activ = in[i];
  }

  for (size_t i = 1; i < net->numLyr; i++) {
    size_t numNrn;
    if (i == net->numLyr - 1) {
      numNrn = net->layers[i].numNrn;
    } else {
      /* don't include bias */
      numNrn = net->layers[i].numNrn - 1;
    }
    for (size_t j = 0; j < numNrn; j++) {
      Neuron_T *curNrn = &net->layers[i].neurons[j];
      double sum = 0;
      for (size_t k = 0; k < net->layers[i - 1].numNrn; k++) {
        Neuron_T *prevLyrNrn = &net->layers[i - 1].neurons[k];
        sum += prevLyrNrn->activ * prevLyrNrn->weights[j].val;
      }
      curNrn->activ = FFNN_sig(sum);
    }
  }

  /* copy over outputs */
  Layer_T *outLyr = &net->layers[net->numLyr - 1];
  for (size_t i = 0; i < outLyr->numNrn; i++) {
    out[i] = outLyr->neurons[i].activ;
  }
}

void FFNN_train(Net_T *net, TrainSet_T *tSet) {
  assert(tSet->numElm != 0 && tSet->lrnRate != 0);
  size_t numEpoch = tSet->numEpoch;
  size_t numElm = tSet->numElm;
  size_t numOut = net->layers[net->numLyr - 1].numNrn;
  for (size_t i = 0; i < numEpoch; i++) {
    double cost = 0.0;
    for (size_t j = 0; j < numElm; j++) {
      double *out = malloc(numOut * sizeof(double));
      assert(out != NULL);
      FFNN_feedForward(net, tSet->in[j], out);
      cost += FFNN_backprop(net, out, tSet->expOut[j], tSet->lrnRate);

      free(out);
    }
    printf("Avg cost of epoch %lu/%lu: %f\n", i, numEpoch - 1, cost/numElm);
  }
  double *testOut = malloc(numOut * sizeof(double));

  FFNN_feedForward(net, tSet->in[0], testOut);
  printf("Training complete! Testing output...\n");
  printf("Expected:\n");
  for (size_t i = 0; i < numOut; i++) {
    printf("  %lu) %f\n", i, tSet->expOut[0][i]);
  }
  printf("Actual:\n");
  for (size_t i = 0; i < numOut; i++) {
    printf("  %lu) %f\n", i, testOut[i]);
  }

  free(testOut);
}

void FFNN_print(Net_T *net) {
  printf("%zu layer network:\n", net->numLyr);
  for (size_t i = 0; i < net->numLyr; i++) {
    printf("  layer %zu:\n", i);
    for (size_t j = 0; j < net->layers[i].numNrn; j++) {
    Neuron_T *curNrn = &net->layers[i].neurons[j];
    printf("    neuron %zu (activ: %.2f) weights:\n      ", j, curNrn->activ);
      for (size_t k = 0; k < curNrn->numWgt; k++) {
        printf("[%.2f] ", curNrn->weights[k].val);
      }
    printf("\n");
    }
  }
}

void FFNN_free(Net_T *net) {
  for (size_t i = 0; i < net->numLyr; i++) {
    Layer_T *curLyr = &net->layers[i];
    for (size_t j = 0; j < curLyr->numNrn; j++) {
      if (i != net->numLyr - 1) {
        free(curLyr->neurons[j].weights);
      }
    }
    free(curLyr->neurons);
  }
  free(net->layers);
  free(net);
}

double FFNN_backprop(Net_T *net, const double *out, const double *expOut,
                     double lrnRate) {
  /* calculate error of output layer */
  Layer_T *lastLyr = &net->layers[net->numLyr - 1];
  for (size_t i = 0; i < lastLyr->numNrn; i++) {
    Neuron_T *curNrn = &lastLyr->neurons[i];
    curNrn->curErr = expOut[i] - out[i];
  }

  /* calculate errors of other layers*/
  for (size_t i = net->numLyr - 2; i > 0; i--) {
    for (size_t j = 0; j < net->layers[i].numNrn; j++) {
      Neuron_T *curNrn = &net->layers[i].neurons[j];
      double err = 0;
      for (size_t k = 0; k < curNrn->numWgt; k++) {
        Weight_T *wgtToNxt = &curNrn->weights[k];
        err += wgtToNxt->val * wgtToNxt->next->curErr;
      }
      curNrn->curErr = err;
    }
  }

  FFNN_gradDescent(net, lrnRate);

  /* sum error of output layer */
  double totalErr = 0.0;
  for (size_t i = 0; i < lastLyr->numNrn; i++) {
    totalErr += 0.5 * pow(lastLyr->neurons[i].curErr, 2);
  }

  return totalErr;
}

void FFNN_gradDescent(Net_T *net, double lrnRate) {
  /* adjust weights */
  for (size_t i = 0; i < net->numLyr - 1; i++) {
    for (size_t j = 0; j < net->layers[i].numNrn; j++) {
      Neuron_T *curNrn = &net->layers[i].neurons[j];
      for (size_t k = 0; k < curNrn->numWgt; k++) {
        Neuron_T *nxtNrn = curNrn->weights[k].next;
        curNrn->weights[k].val += lrnRate * nxtNrn->curErr * curNrn->activ
                                  * FFNN_dsig(nxtNrn->activ);
      }
    }
  }
}

double FFNN_sig(double x) {
  return 1.0/(1.0 + exp(-x));
}

double FFNN_dsig(double x) {
  return FFNN_sig(x) * (1 - FFNN_sig(x));
}
