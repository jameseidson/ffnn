#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef struct Image {
  uint8_t *pixels;
  uint8_t label;
} Image_T;

typedef struct ImageData { uint32_t numImg;
  uint32_t numRow;
  uint32_t numCol;
} ImageData_T;


void flip_32(uint32_t *in) {
  uint8_t i, j;
  uint8_t *t = (uint8_t *)in;

  i = t[0];
  j = t[1];
  t[0] = t[3];
  t[1] = t[2];
  t[2] = j;
  t[3] = i;
}

ImageData_T readImg(Image_T *imgs, char *imgFile, char *lblFile) {
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

  /* allocate images- their pixels and labels */
  imgs = malloc(data.numImg * sizeof(Image_T));
  for(size_t i = 0; i < data.numImg; i++) {
    imgs[i].pixels = malloc(imgSize);
    fread(imgs[i].pixels, imgSize, 1, ifp);
    fread(&imgs[i].label, lblSize, 1, lfp);
  }

  return data;
}

int main(int argc, char **argv) {
  if(argc != 3) {
    printf("Usage: %s [image file] [label file]\n", argv[0]);
    return 1;
  }
  Image_T *imgs = NULL;
  readImg(imgs, argv[1], argv[2]);

  return 0;
}
