#include "mnistread.h"

void flip_32(uint32_t *);

ImageData_T MNIST_read(Image_T **imgs, char *imgFile, char *lblFile) {
  /* open files */
  FILE *ifp = fopen(imgFile, "rb");
  FILE *lfp = fopen(lblFile, "rb");
  assert(ifp != NULL && lfp != NULL);

  /* ignore magic number */
  fseek(ifp, 4, SEEK_SET);
  fseek(lfp, 4, SEEK_SET);

  /* read header */
  uint32_t hdr[3];
  for(size_t i = 0; i < 3; i++) {
    fread(&(hdr[i]), sizeof(uint32_t), 1, ifp);
    flip_32(&(hdr[i]));
  }
  ImageData_T data;
  data.numImg = hdr[0];
  data.numRow = hdr[1];
  data.numCol = hdr[2];
  size_t imgSize = data.numRow * data.numCol * sizeof(uint8_t);

  uint32_t numLbl;
  fread(&(numLbl), sizeof(uint32_t), 1, lfp);
  flip_32(&(numLbl));
  size_t lblSize = sizeof(uint8_t);
  assert(numLbl == data.numImg);
  /* allocate pixels and labels for each image */
  *imgs = malloc(data.numImg * sizeof(Image_T));
  assert(imgs != NULL);

  for(size_t i = 0; i < data.numImg; i++) {
    Image_T *imgPtr = *imgs;
    Image_T *curImg = &imgPtr[i];

    curImg->pixels = malloc(imgSize);
    assert(curImg != NULL);

    fread(curImg->pixels, imgSize, 1, ifp);
    fread(&curImg->label, lblSize, 1, lfp);
  }

  fclose(ifp);
  fclose(lfp);

  return data;
}

TrainSet_T *MNIST_prep(ImageData_T *imgDat, Image_T *imgs, size_t numEpoch,
                       size_t numToRead) {
  size_t numImg = imgDat->numImg;
  size_t numPxl = imgDat->numRow * imgDat->numCol;

  TrainSet_T *tSet = malloc(sizeof(TrainSet_T));
  assert(tSet != NULL);
  tSet->numElm = (numToRead == 0) ? numImg : numToRead;
  tSet->numEpoch = numEpoch;
  tSet->lrnRate = 0.2;

  tSet->in = malloc(numImg * sizeof(double *));
  tSet->expOut = malloc(numImg * sizeof(double *));
  assert(tSet->in != NULL && tSet->expOut != NULL);

  /* move everything into an array of doubles */
  for (size_t i = 0; i < numImg; i++) {
    tSet->in[i] = malloc(numPxl * sizeof(double));

    for (size_t j = 0; j < numPxl; j++) {
      tSet->in[i][j] = (double)imgs[i].pixels[j] / 255.0;
    }

    /* match labels to expected output layer format */
    tSet->expOut[i] = calloc(10, sizeof(double));
    assert(tSet->expOut[i] != NULL);

    tSet->expOut[i][imgs[i].label] = 1.0; 
  }

  return tSet;
}

void MNIST_printPrep(TrainSet_T *tSet, ImageData_T *imgDat) {
  size_t numImg = imgDat->numImg;
  size_t numPxl = imgDat->numRow * imgDat->numCol;

  for (size_t i = 0; i < numImg; i++) {
    printf("Input:\n");
    for (size_t j = 0; j < numPxl; j++) {
      if (tSet->in[i][j] >= 0.95) {
        printf("*");
      } else {
        printf(" ");
      }
      if (j % 28 == 0) {
        printf("\n");
      }
    }
    printf("\nExpected Outputs:\n");
    for (size_t j = 0; j < 10; j++) {
      printf("%lu: %f\n", j, tSet->expOut[i][j]);
    }
    printf("\n");
  }
}

void MNIST_free(TrainSet_T *tSet, ImageData_T *imgDat, Image_T *imgs) {
  for (size_t i = 0; i < imgDat->numImg; i++) {
    free(imgs[i].pixels);
    free(tSet->in[i]);
    free(tSet->expOut[i]);
  }
  free(tSet->in);
  free(tSet->expOut);
  free(tSet);
  free(imgs);
}

void flip_32(uint32_t *in) {
  uint8_t ptr1, ptr2;
  uint8_t *inPtr = (uint8_t *)in;

  /* pointer indexing magic */
  ptr1 = inPtr[0];
  ptr2 = inPtr[1];
  inPtr[0] = inPtr[3];
  inPtr[1] = inPtr[2];
  inPtr[2] = ptr2;
  inPtr[3] = ptr1;
}
