#include "ffnn.h"

/* structures */
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
  size_t numWgt;
  Weight_T *weights;
};

struct Weight {
  double val;
  Neuron_T *next;
};

struct Data {
  size_t size;
  double **input;
  double **expOut;
};

/* forward declarations */
double FFNN_sig(double x);
void FFNN_print(Net_T *net);
/* returns cost */
double FFNN_backprop(double *out, double *expOut);

/* definitions */
Net_T *FFNN_init(size_t netSize, size_t *topology) {
  /* validate input */
  assert(netSize != 0);
  for(size_t i = 0; i < netSize; i++) {
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
  for(size_t i = 0; i < netSize; i++) {
    Layer_T *curLyr = &net->layers[i];
    curLyr->numNrn = topology[i];
    /* add extra neuron in layer to be bias */
    if(i != netSize - 1) {
      curLyr->numNrn++;
    }

    curLyr->neurons = malloc(curLyr->numNrn * sizeof(Neuron_T));
    assert(curLyr->neurons != NULL);
  }

  /* allocate and initialize weights */
  for(size_t i = 0; i < netSize - 1; i++) {
    for(size_t j = 0; j < net->layers[i].numNrn; j++) {
      Neuron_T *curNrn = &net->layers[i].neurons[j];
      curNrn->activ = 1;
      curNrn->numWgt = topology[i + 1];
      curNrn->weights = malloc(curNrn->numWgt * sizeof(Weight_T));
      assert(curNrn->weights != NULL);
      /* point kth weight in curr layer to kth neuron in next layer */
      for(size_t k = 0; k < curNrn->numWgt; k++) {
        curNrn->weights[k].next = &net->layers[i + 1].neurons[k];
        curNrn->weights[k].val = (double)rand() / RAND_MAX;
      }
    }
  }

  return net;
}

/* undefined behavior if inputs is not the right size */
double *FFNN_feedForward(Net_T *net, double *inputs) {
  /* feed in input data */
  for(size_t i = 0; i < net->layers[0].numNrn - 1; i++) {
    net->layers[0].neurons[i].activ = inputs[i];
  }
  /* looping weights before neurons allows us to apply sigmoid each iter */
  for(size_t i = 0; i < net->numLyr - 1; i++) {
    for(size_t j = 0; j < net->layers[i].neurons[0].numWgt; j++) {
      for(size_t k = 0; k < net->layers[i].numNrn; k++) {
        Weight_T *curWgt = &net->layers[i].neurons[k].weights[j];
        /* influence activation of next layer */
        curWgt->next->activ += curWgt->val
                    * net->layers[i].neurons[k].activ;
      }
      Neuron_T *nxtLyrNrn = &net->layers[i + 1].neurons[j];
      nxtLyrNrn->activ = FFNN_sig(nxtLyrNrn->activ);
    }
  }

  /* copy and return outputs */
  Layer_T *outLyr = &net->layers[net->numLyr - 1];

  assert(outLyr->neurons != NULL);
  double *outActiv = malloc(outLyr->numNrn * sizeof(double));
  for(size_t i = 0; i < outLyr->numNrn; i++) {
    outActiv[i] = outLyr->neurons[i].activ;
  }

  return outActiv;
}

void FFNN_train(Net_T *net, Data_T *data, size_t epoch) {
  for(size_t i = 0; i < epoch; i++) {
    double cost = 0;
    for(size_t j = 0; j < data->size; j++) {
      double *out = FFNN_feedForward(net, data->input[j]);
      cost += FFNN_backprop(out, data->expOut[j]);
    }
    printf("Average cost of epoch %zu: %f\n", i, cost/data->size);
  }
}

void FFNN_free(Net_T *net) {
  for(size_t i = 0; i < net->numLyr; i++) {
    for(size_t j = 0; j < net->layers[i].numNrn; j++) {
      free(net->layers[i].neurons[j].weights);
    }
    free(net->layers[i].neurons);
  }
  free(net->layers);
  free(net);
}

double FFNN_backprop(double *out, double *expOut) {
  (void)out;
  (void)expOut;

  /* TODO */

  return 0;
}

double FFNN_sig(double x) {
  return 1/(1 + exp(-x));
}

void FFNN_print(Net_T *net) {
  printf("%zu layer network:\n", net->numLyr);
  for(size_t i = 0; i < net->numLyr; i++) {
    printf("  layer %zu\n", i);
    for(size_t j = 0; j < net->layers[i].numNrn; j++) {
    printf("    neuron %zu weights\n      ", j);
      for(size_t k = 0; k < net->layers[i].neurons[j].numWgt; k++) {
        printf("[%.2f] ", net->layers[i].neurons[j].weights[k].val);
      }
    printf("\n");
    }
  }
}
