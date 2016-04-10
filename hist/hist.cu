#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bitmap.h"

__global__ void get_hist(unsigned char *image, int size, int *hist)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= size) return;
    int value = image[tid];
    atomicAdd(&hist[value], 1);    
}

int main(int argc, char *argv[])
{
    BITMAPINFOHEADER bitmapInfoHeader;
    BITMAPFILEHEADER bitmapFileHeader;
    unsigned char *image;
    image = LoadBitmapFile(argv[1],&bitmapInfoHeader, &bitmapFileHeader);   

    unsigned char *d_image;
    cudaMalloc((void**)&d_image, bitmapInfoHeader.biSizeImage);
    cudaMemcpy(d_image, image, bitmapInfoHeader.biSizeImage, cudaMemcpyHostToDevice);

    int hist[256];
    for(int i = 0; i < 256; i++)
        hist[i] = 0;
    int *d_hist;
    cudaMalloc((void**)&d_hist, 256 * sizeof(int));
    cudaMemset(d_hist, 0, 256 * sizeof(int));
    
    dim3 block(1024);
    dim3 grid((bitmapInfoHeader.biSizeImage + 1023) / 1024);

    printf("begin\n");
    get_hist<<<grid, block>>>(d_image, bitmapInfoHeader.biSizeImage, d_hist);
    printf("finish\n");
    cudaMemcpy(hist, d_hist, 256*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 256; i++)
        printf("%d\t%d\n", i, hist[i]);
    cudaFree(d_image);
    cudaFree(d_hist);

    return 0;
}
