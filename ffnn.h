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
  double lrnRate; /* scale factor for grad descent- 0.1 - 0.3 is typical */
  size_t numEpoch; /* the maximum number of epochs to train */
  size_t numElm; /* the number of elements in each epoch */
  double **in; /* 2d array of input activations to train on */
  double **expOut; /* 2d array of expected output activations */
} TrainSet_T;

/* init 
 * Params: (size_t) Size of network.
 *         (size_t *) Array of layer sizes.
 * Return: (Net_T *) A new network meeting the provided specifications.
 * Errors: (Checked) netSize <= 0.
 *         (Checked) Any element within topology <= 0.
 *         (Checked) topology == NULL.
 * Initializes a new network based on the provided topology. The size of
 *   topology corresponds to the number of layers in the network, while the
 *   values stored in each index corespond to the number of neurons in that
 *   layer. Note that the counts in topology should not include bias neurons.
 */
Net_T *FFNN_init(size_t netSize, size_t *topology);

/* feedforward
 * Params: (Net_T *) Network to feed forward on.
 *         (double *) Array of input neuron activations.
           (double *) Pass-by-reference parameter that is assgned to the 
                      network's output layer activations after feeding forward.
 * Return: (void)
 * Errors: (Checked) net == NULL || in == NULL || out == NULL.
 *         (Unchecked) Number of neurons in *in != number of input neurons.
 *         (Unchecked) Number of neurons in *out != number of output neurons.
 * Feeds forward the provided input activations to obtain a confidence value in
 *   each output neuron. These confidence values should become more
 *   representative of the provided data set as the network is trained more.
 */
void FFNN_feedForward(Net_T *net, double *in, double *out);


/* train 
 * Params: (Net_T *) Network to train.
 *         (TrainSet_T *) Set of data to train on.
 *         (FILE *) file to save training progess to after each epoch. Can be 
 *                  set to NULL to disable autosaving.
 * Return: (void)
 * Errors: (Checked) net == NULL || tSet == NULL.
 *         (Checked) tSet->numElm == 0 || tSet->lrnRate == 0.
 * Trains the network using backpropogation and gradient descent. Note that 
 *   TrainSet_T is defined above.
 */
void FFNN_train(Net_T *net, TrainSet_T *tSet, FILE *nfp);

/* save
 * Params: (Net_T *) Network to save.
 *         (FILE *) File to save the network to.
 * Return: (void)
 * Errors: (Checked) net == NULL || nfp == NULL.
 * Saves the network structure and training state to nfp in a binary format.
 */
void FFNN_save(Net_T *net, FILE *nfp);

/* load 
 * params: (file *) file to load the network from.
 * return: (net_t *) a new network meeting the specifications defined in nfp.
 * errors: (checked) nfp == null.
 * loads the network stored in nfp. be careful to open nfp in read mode or the
 *   data saved there will be overwritten before being loaded.
 */
Net_T *FFNN_load(FILE *nfp);

/* free 
 * params: (net *) Network to free.
 * return: (void)
 * errors: (checked) net == null.
 * Frees all memory stored in the network. Note that TrainSet_T's are malloced
 *   by the user and thus it is their responsibility to free them.
 */
void FFNN_free(Net_T *net);

/* print 
 * params: (net *) Network to print.
 * return: (void)
 * errors: (checked) net == null.
 * Prints the network structure, activations, and weights to stdout. Intended
 *   mostly for debugging purposes.
 */
void FFNN_print(Net_T *net);

#endif
