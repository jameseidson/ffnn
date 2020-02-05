#ifndef MNISTREAD_H
#define MNISTREAD_H

#include "ffnn.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

typedef struct Image {
  uint8_t *pixels;
  uint8_t label;
} Image_T;

typedef struct ImageData { 
  uint32_t numImg;
  uint32_t numRow;
  uint32_t numCol;
} ImageData_T;

ImageData_T MNIST_read(Image_T **imgs, char *imgFile, char *lblFile);
TrainSet_T *MNIST_prep(ImageData_T *imgDat, Image_T *imgs, size_t numEpoch);
void MNIST_printPrep(TrainSet_T *tSet, ImageData_T *imgDat);
void MNIST_free(TrainSet_T *tSet, ImageData_T *imgDat, Image_T *imgs);

#endif
