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
    size_t numWgt;
    Weight_T *weights;
};

struct Weight {
    double val;
    Neuron_T *next;
};

Net_T *FF_initNet(size_t netSize, size_t *topology) {
    srand(time(NULL));
    /* allocate and initialize a new network */
    Net_T *net = malloc(sizeof(Net_T));
    net->numLyr = netSize;
    net->layers = malloc(net->numLyr * sizeof(Layer_T));

    /* allocate and initialize layers */
    for(size_t i = 0; i < netSize; i++) {
        Layer_T *lyr_i = &net->layers[i];
        lyr_i->numNrn = topology[i];
        /* add extra neuron in layer to be bias */
        if(i != netSize - 1) {
            lyr_i->numNrn++;
        }

        lyr_i->neurons = malloc(lyr_i->numNrn * sizeof(Neuron_T));
    }

    /* allocate and initialize weights */
    for(size_t i = 0; i < netSize - 1; i++) {
        for(size_t j = 0; j < net->layers[i].numNrn; j++) {
            Neuron_T *nrn_i = &net->layers[i].neurons[j];
            nrn_i->numWgt = topology[i + 1];
            nrn_i->weights = malloc(nrn_i->numWgt * sizeof(Weight_T));
            /* point kth weight in curr layer to kth neuron in next layer */
            for(size_t k = 0; k < nrn_i->numWgt; k++) {
                nrn_i->weights[k].next = &net->layers[i + 1].neurons[k];
                nrn_i->weights[k].val = (double)rand() / RAND_MAX;
            }
        }
    }

    return net;
}

void FF_printNet(Net_T *net) {
    printf("%zu layer network:\n", net->numLyr);
    for(size_t i = 0; i < net->numLyr; i++) {
        printf("    layer %zu\n", i);
        for(size_t j = 0; j < net->layers[i].numNrn; j++) {
        printf("        neuron %zu weights\n            ", j);
            for(size_t k = 0; k < net->layers[i].neurons[j].numWgt; k++) {
                printf("[%.2f] ", net->layers[i].neurons[j].weights[k].val);
            }
        printf("\n");
        }
    }
}

void FF_freeNet(Net_T *net) {
    for(size_t i = 0; i < net->numLyr; i++) {
        for(size_t j = 0; j < net->layers[i].numNrn; j++) {
            free(net->layers[i].neurons[j].weights);
        }
        free(net->layers[i].neurons);
    }
    free(net->layers);
    free(net);
}
