/*
 * Naive Parallel version using CUDA.
 * The performance gain is about 250 times.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bitmap.h"

//define blocksize
#define BLOCKSIZE 1024
#define PIX_KEY_WIDTH 32
#define PIX_KEY_HEIGHT BLOCKSIZE/PIX_KEY_WIDTH

__device__ int col_calculator(int, int);
__device__ int row_calculator(int, int);
__device__ int de_key_generator(int, int, int);

void swap (int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

void randomize ( int arr[], int n )
{
	//randomly switch add[i] with another element with random index
    srand(time(NULL));
    for (int i = n -1; i > 0; i--)
    {
        int j = rand() % (i+1);
        swap(&arr[i], &arr[j]);
    }
}


void substitution_key_generator(int *sub_key)
{
    time_t t;
    srand((unsigned) time(&t));
    for(int i = 0; i < BLOCKSIZE; i++){
        sub_key[i] = rand() % 256;
    }
}

void encryption_permutation_key_generator(int *per_key, int size)
{
    for(int i = 0; i < size; i++){
        per_key[i] = i+1;
    }
}

//Step 1: Pixel substitution kernel
__global__ void substitution(unsigned char *inputImage, int *sub_key,int imagewidth)
{
    int block_x = threadIdx.x % PIX_KEY_WIDTH;	// col position in block
    int block_y = threadIdx.x / PIX_KEY_WIDTH;	// row position in block
    
    int block_idx_x = blockIdx.x % (imagewidth / PIX_KEY_WIDTH);   //block position in the image
    int block_idx_y = blockIdx.x / (imagewidth / PIX_KEY_WIDTH);   
    
    int idx_pixel = (block_idx_y * PIX_KEY_HEIGHT + block_y) * imagewidth + (block_idx_x * PIX_KEY_WIDTH + block_x);
    
    for(int i = 0; i < 3; i++)
    {
        inputImage[3 * idx_pixel + i] = inputImage[3 * idx_pixel + i] ^ sub_key[threadIdx.x];
    }
}

__device__ int col_calculator(int key, int col)
{
    int N = col;
    return key % N;
}

__device__ int row_calculator(int key, int col)
{   
    int N = col;
    return key / N + 1;
}

__device__ int de_key_generator(int col, int row, int width)
{   
    return row * width + col + 1; 
}

//Step 2: Pixel permutation within one block kernel
__global__ void pixel_permutation(unsigned char *inputImage, unsigned char *outputImage,
		int *en_key, int *de_key,int imagewidth)
{
    int block_x = threadIdx.x % PIX_KEY_WIDTH;	// col position in block
    int block_y = threadIdx.x / PIX_KEY_WIDTH;	// row position in block
    
    int block_idx_x = blockIdx.x % (imagewidth / PIX_KEY_WIDTH);   //block position in the image
    int block_idx_y = blockIdx.x / (imagewidth / PIX_KEY_WIDTH);   
    
    int idx_pixel = (block_idx_y * PIX_KEY_HEIGHT + block_y) * imagewidth + (block_idx_x * PIX_KEY_WIDTH + block_x);

    int key = en_key[block_y * PIX_KEY_WIDTH + block_x];
    int new_col = col_calculator(key, PIX_KEY_WIDTH);
    int new_row = row_calculator(key, PIX_KEY_WIDTH);
    if(new_col == 0)
    {
        new_col = PIX_KEY_WIDTH;
        new_row -= 1;
    }

    de_key[(new_row - 1) * PIX_KEY_WIDTH + (new_col - 1)]
           = de_key_generator(block_x, block_y, PIX_KEY_WIDTH);
    for(int i = 0; i < 3; i++)
    {
    	 outputImage[((block_idx_y * PIX_KEY_HEIGHT + new_row - 1) * imagewidth + block_idx_x * PIX_KEY_WIDTH + new_col - 1) * 3 + i]
    	             = inputImage[idx_pixel * 3 + i];
    }  
}

//Step 3: Block permutation kernel
__global__ void block_permutation(unsigned char *inputImage, unsigned char *outputImage,
		int *en_key, int *de_key,int imagewidth,int imageheight)
{
    int block_x = threadIdx.x % PIX_KEY_WIDTH;	// col position in block
    int block_y = threadIdx.x / PIX_KEY_WIDTH;	// row position in block
    
    int block_idx_x = blockIdx.x % (imagewidth / PIX_KEY_WIDTH);   //block position in the image
    int block_idx_y = blockIdx.x / (imagewidth / PIX_KEY_WIDTH);   
    
    int idx_pixel = (block_idx_y * PIX_KEY_HEIGHT + block_y) * imagewidth + (block_idx_x * PIX_KEY_WIDTH + block_x);

    int key = en_key[block_idx_y * (imagewidth/PIX_KEY_WIDTH) + block_idx_x];
    int new_col_block = col_calculator(key, imagewidth / PIX_KEY_WIDTH);
    int new_row_block = row_calculator(key, imagewidth / PIX_KEY_WIDTH);
    if(new_col_block == 0)
    {
        new_col_block = imagewidth / PIX_KEY_WIDTH;
        new_row_block -= 1;
    }

    de_key[(new_row_block - 1) * (imagewidth / PIX_KEY_WIDTH) + (new_col_block - 1)]
           = de_key_generator(block_idx_x, block_idx_y, imagewidth / PIX_KEY_WIDTH);
    for(int i = 0; i < 3; i++)
    {
        outputImage[(((new_row_block-1) * PIX_KEY_HEIGHT) * imagewidth + (new_col_block-1) * PIX_KEY_WIDTH +(block_y * imagewidth+ block_x)) * 3 + i]
                     = inputImage[idx_pixel * 3 + i];
    }
}

int main(int argc, char *argv[])
{
    INFOHEADER bmpInfoHeader;
    FILEHEADER bmpFileHeader;
    unsigned char *input_image;
    unsigned char *pixel_permutation_image;
    unsigned char *block_permutation_iamge;

    INFOHEADER bmpInfoHeader1;
    FILEHEADER bmpFileHeader1;
    unsigned char *output_image;
    unsigned char *out_block_permutation_iamge;
    unsigned char *out_pixel_permutation_image;

    unsigned char *d_input_image;
    unsigned char *d_pixel_permutation_image;
    unsigned char *d_block_permutation_iamge;
    unsigned char *d_output_image;
    unsigned char *d_out_block_permutation_iamge;
    unsigned char *d_out_pixel_permutation_image;
   
    //Loading input image and allocate enough memory for processing
    input_image = LoadImage(argv[1],&bmpInfoHeader, &bmpFileHeader);
    pixel_permutation_image = (unsigned char *) malloc(bmpInfoHeader.imagesize * sizeof(char));
    memset(pixel_permutation_image, 0, bmpInfoHeader.imagesize);
    block_permutation_iamge = (unsigned char *) malloc(bmpInfoHeader.imagesize * sizeof(char));
    memset(block_permutation_iamge, 0, bmpInfoHeader.imagesize);

    printf("Image size: %d\n", bmpInfoHeader.imagesize);
    printf("Image widthh: %d\n", bmpInfoHeader.width);
    printf("Image height: %d\n", bmpInfoHeader.height);
    int imagesize = bmpInfoHeader.imagesize/3;
    int imagewidth = bmpInfoHeader.width;
    int imageheight = bmpInfoHeader.height;

    
    int substitution_key[BLOCKSIZE];
    substitution_key_generator(substitution_key);
    int *d_substitution_key;

    int pix_perm_key[BLOCKSIZE];
    int *d_pix_per_key;
    int *d_pix_per_key_de;
    encryption_permutation_key_generator(pix_perm_key, BLOCKSIZE);
    randomize(pix_perm_key, BLOCKSIZE);
    
    int block_perm_key[imagesize/BLOCKSIZE];
    int *d_block_per_key;
    int *d_block_per_key_de;
    encryption_permutation_key_generator(block_perm_key,imagesize/BLOCKSIZE);
    randomize(block_perm_key, imagesize/BLOCKSIZE);    
    
    //CUDA malloc and CUDA memcpy
    cudaMalloc((void**)&d_input_image, imagesize*3);
    cudaMalloc((void**)&d_pixel_permutation_image, imagesize*3);
    cudaMalloc((void**)&d_block_permutation_iamge, imagesize*3);

    cudaMalloc((void**)&d_substitution_key, BLOCKSIZE * sizeof(int));
    cudaMalloc((void**)&d_pix_per_key, BLOCKSIZE * sizeof(int));
    cudaMalloc((void**)&d_pix_per_key_de, BLOCKSIZE * sizeof(int));
    cudaMalloc((void**)&d_block_per_key, imagesize/BLOCKSIZE* sizeof(int));
    cudaMalloc((void**)&d_block_per_key_de, imagesize/BLOCKSIZE * sizeof(int));

    cudaMemcpy(d_input_image, input_image, imagesize*3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pixel_permutation_image, pixel_permutation_image, imagesize*3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_permutation_iamge, block_permutation_iamge, imagesize*3, cudaMemcpyHostToDevice);

    cudaMemcpy(d_substitution_key, substitution_key, BLOCKSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pix_per_key, pix_perm_key, BLOCKSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_pix_per_key_de, 0, BLOCKSIZE * sizeof(int));
    cudaMemcpy(d_block_per_key, block_perm_key, imagesize/BLOCKSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_block_per_key_de, 0,imagesize/BLOCKSIZE * sizeof(int));

    //define the dimension of CUDA grid and block
    dim3 gridDim(imagesize/BLOCKSIZE , 1, 1);
    dim3 blockDim(BLOCKSIZE, 1, 1);

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t end;
    cudaEventCreate(&end);
    float en_substitution_time, en_pixel_permutation_time, en_block_permutation_time;
    float de_substitution_time, de_pixel_permutation_time, de_block_permutation_time;

    //Image Encryption using three CUDA kernels
    cudaEventRecord(start, 0);
    substitution<<<gridDim, blockDim>>>(d_input_image, d_substitution_key, imagewidth);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&en_substitution_time, start, end);
    printf("Encryption substitution Time: %f ms\n", en_substitution_time);

    cudaEventRecord(start, 0);
    pixel_permutation<<<gridDim, blockDim>>>(d_input_image, d_pixel_permutation_image,
    		d_pix_per_key, d_pix_per_key_de, imagewidth);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&en_pixel_permutation_time, start, end);
    printf("Encryption pixel permutation Time: %f ms\n", en_pixel_permutation_time);

    cudaEventRecord(start, 0);
    block_permutation<<<gridDim, blockDim>>>(d_pixel_permutation_image, d_block_permutation_iamge,
    		d_block_per_key, d_block_per_key_de,imagewidth,imageheight);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&en_block_permutation_time, start, end);
    printf("Encryption block permutation Time: %f ms\n", en_block_permutation_time);
    printf("Encryption time: %f ms\n", en_substitution_time + en_pixel_permutation_time + en_block_permutation_time);

    cudaMemcpy(block_permutation_iamge, d_block_permutation_iamge, imagesize*3, cudaMemcpyDeviceToHost);
    SaveImage(argv[2], block_permutation_iamge, &bmpFileHeader, &bmpInfoHeader);


    output_image = LoadImage(argv[2],&bmpInfoHeader1, &bmpFileHeader1);
    out_block_permutation_iamge = (unsigned char *) malloc(bmpInfoHeader.imagesize * sizeof(char));
    memset(out_block_permutation_iamge, 0, imagesize);
    out_pixel_permutation_image = (unsigned char *) malloc(bmpInfoHeader.imagesize * sizeof(char));
    memset(out_pixel_permutation_image, 0, imagesize);

    cudaMalloc((void**)&d_output_image, imagesize*3);
    cudaMalloc((void**)&d_out_block_permutation_iamge, imagesize*3);
    cudaMalloc((void**)&d_out_pixel_permutation_image, imagesize*3);

    cudaMemcpy(d_output_image, output_image, imagesize*3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_block_permutation_iamge, out_block_permutation_iamge, imagesize*3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_pixel_permutation_image, out_pixel_permutation_image, imagesize*3, cudaMemcpyHostToDevice);

    //Image Decryption using three CUDA kernels
    cudaEventRecord(start, 0);
    block_permutation<<<gridDim, blockDim>>>(d_output_image, d_out_block_permutation_iamge,
    		d_block_per_key_de, d_block_per_key,imagewidth,imageheight);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&de_block_permutation_time, start, end);
    printf("Decryption block permutation Time: %f ms\n", de_block_permutation_time);

    cudaEventRecord(start, 0);
    pixel_permutation<<<gridDim, blockDim>>>(d_out_block_permutation_iamge, d_out_pixel_permutation_image,
    		d_pix_per_key_de, d_pix_per_key,imagewidth);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&de_pixel_permutation_time, start, end);
    printf("Decryption pixel permutation Time: %f ms\n", de_pixel_permutation_time);

    cudaEventRecord(start, 0);
    substitution<<<gridDim, blockDim>>>(d_out_pixel_permutation_image, d_substitution_key,imagewidth);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&de_substitution_time, start, end);
    printf("Decryption substitution Time: %f ms\n", de_substitution_time);
    printf("Decryption time: %f ms\n", de_substitution_time + de_pixel_permutation_time + de_block_permutation_time);

    cudaMemcpy(out_pixel_permutation_image, d_out_pixel_permutation_image, imagesize*3, cudaMemcpyDeviceToHost);
    SaveImage(argv[3], out_pixel_permutation_image, &bmpFileHeader1, &bmpInfoHeader1);

    //Memory free
    cudaFree(d_input_image);
    cudaFree(d_pixel_permutation_image);
    cudaFree(d_block_permutation_iamge);
    cudaFree(d_output_image);
    cudaFree(d_out_block_permutation_iamge);
    cudaFree(d_out_pixel_permutation_image);

    cudaFree(d_substitution_key);
    cudaFree(d_pix_per_key);
    cudaFree(d_pix_per_key_de);
    cudaFree(d_block_per_key);
    cudaFree(d_block_per_key_de);
    
    free(input_image);
    free(pixel_permutation_image);
    free(block_permutation_iamge);
    free(output_image);
    free(out_block_permutation_iamge);
    free(out_pixel_permutation_image);

    return 0;
}
