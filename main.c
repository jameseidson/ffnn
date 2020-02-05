#include "ffnn.h"
#include "mnistread.h"

#include <assert.h>
#include <stdio.h>

int main(int argc, char **argv) {
  if(argc != 3) {
    fprintf(stderr, "Usage: %s [image file] [label file]\n", argv[0]);
    return 1;
  }

  Image_T *imgs = NULL;

  ImageData_T imgDat = MNIST_read(&imgs, argv[1], argv[2]);
  TrainSet_T *tSet = MNIST_prep(&imgDat, imgs, 1000);
  
  size_t topology[] = { imgDat.numRow * imgDat.numCol, 1, 10 };

  Net_T *network = FFNN_init(3, topology);
  FFNN_train(network, tSet);
  FFNN_free(network);

  MNIST_free(tSet, &imgDat, imgs);

  return 0;
}
