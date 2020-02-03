#include "ffnn.h"

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

double FFNN_backprop(Net_T *net, double *out, double *expOut);
double FFNN_sig(double x);
double FFNN_dsig(double x);

void FFNN_print(Net_T *net);

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
    if (i != netSize - 1) {
      curLyr->numNrn++;
    }

    curLyr->neurons = malloc(curLyr->numNrn * sizeof(Neuron_T));
    assert(curLyr->neurons != NULL);
  }

  /* allocate and initialize weights */
  for (size_t i = 0; i < netSize - 1; i++) {
    for (size_t j = 0; j < net->layers[i].numNrn; j++) {
      Neuron_T *curNrn = &net->layers[i].neurons[j];
      curNrn->activ = 1.0;
      curNrn->curErr = 0.0;
      curNrn->numWgt = topology[i + 1];
      curNrn->weights = malloc(curNrn->numWgt * sizeof(Weight_T));
      assert(curNrn->weights != NULL);
      /* point kth weight in curr layer to kth neuron in next layer */
      for (size_t k = 0; k < curNrn->numWgt; k++) {
        curNrn->weights[k].next = &net->layers[i + 1].neurons[k];
        curNrn->weights[k].val = (double)rand() / RAND_MAX;
      }
    }
  }

  return net;
}

/* undefined behavior if in our out is not the right size */
void FFNN_feedForward(Net_T *net, double *in, double *out) {
  /* feed in input data */
  for (size_t i = 0; i < net->layers[0].numNrn - 1; i++) {
    net->layers[0].neurons[i].activ = in[i];
  }

  /* looping weights before neurons allows us to apply sigmoid each iter */
  for (size_t i = 0; i < net->numLyr - 1; i++) {
    for (size_t j = 0; j < net->layers[i].neurons[0].numWgt; j++) {
      for (size_t k = 0; k < net->layers[i].numNrn; k++) {
        Weight_T *curWgt = &net->layers[i].neurons[k].weights[j];
        /* influence activation of next layer */
        curWgt->next->activ += curWgt->val * net->layers[i].neurons[k].activ;
      }
      Neuron_T *nxtLyrNrn = &net->layers[i + 1].neurons[j];
      nxtLyrNrn->activ = FFNN_sig(nxtLyrNrn->activ);
    }
  }

  /* copy over outputs */
  Layer_T *outLyr = &net->layers[net->numLyr - 1];
  for (size_t i = 0; i < outLyr->numNrn; i++) {
    out[i] = outLyr->neurons[i].activ;
  }
}

void FFNN_train(Net_T *net, double **in, double **expOut, size_t numElem,
                size_t epoch) {
  for (size_t i = 0; i < epoch; i++) {
    double cost = 0;
    for (size_t j = 0; j < numElem; j++) {
      double *out = malloc(net->layers[net->numLyr - 1].numNrn * sizeof(double));
      FFNN_feedForward(net, in[j], out);
      cost += FFNN_backprop(net, out, expOut[j]);
      free(out);
    }
    printf("Avg cost of epoch %zu/%zu: %.3f\n", i, epoch, cost/numElem);
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


double FFNN_backprop(Net_T *net, double *out, double *expOut) {
  /* calculate error of output layer */
  Layer_T *lastLyr = &net->layers[net->numLyr - 1];
  for (size_t i = 0; i < lastLyr->numNrn; i++) {
    lastLyr->neurons[i].curErr = pow(expOut[i] - out[i], 2);
  }

  /* calculate error of each neuron */
  for (size_t i = net->numLyr - 2; i > 0; i--) {
    for (size_t j = 0; j < net->layers[i].numNrn; j++) {
      Neuron_T *curNrn = &net->layers[i].neurons[j];
      for (size_t k = 0; k < curNrn->numWgt; k++) {

        Neuron_T *nxtNrn = curNrn->weights[k].next;
        Weight_T *wgtToNxt = &curNrn->weights[k];

        curNrn->curErr = wgtToNxt->val * wgtToNxt->next->curErr;
        curNrn->weights[k].val += FFNN_dsig(curNrn->activ) 
                                * nxtNrn->activ * nxtNrn->curErr * -1;
      }
    }
  }

  /* sum error of output layer */
  double totalErr = 0;
  for (size_t i = 0; i < lastLyr->numNrn; i++) {
    totalErr += lastLyr->neurons[i].curErr;
  }

  return totalErr;
}

double FFNN_sig(double x) {
  return 1/(1 + exp(-x));
}

double FFNN_dsig(double x) {
  return FFNN_sig(x) * (1 - FFNN_sig(x));
}

void FFNN_print(Net_T *net) {
  printf("%zu layer network:\n", net->numLyr);
  for (size_t i = 0; i < net->numLyr; i++) {
    printf("  layer %zu\n", i);
    for (size_t j = 0; j < net->layers[i].numNrn; j++) {
    printf("    neuron %zu weights\n      ", j);
      for (size_t k = 0; k < net->layers[i].neurons[j].numWgt; k++) {
        printf("[%.2f] ", net->layers[i].neurons[j].weights[k].val);
      }
    printf("\n");
    }
  }
}
