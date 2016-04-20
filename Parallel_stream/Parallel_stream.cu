/*
 * Parallelization with CUDA streams.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bitmap.h"

//define blocksize
#define BLOCKSIZE 1024
#define PIX_KEY_WIDTH 32
#define PIX_KEY_HEIGHT BLOCKSIZE/PIX_KEY_WIDTH
#define IMAGESIZE 512*512

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
    int block_x = threadIdx.x % PIX_KEY_WIDTH;
    int block_y = threadIdx.x / PIX_KEY_WIDTH;
    
    int block_idx_x = blockIdx.x % (imagewidth / PIX_KEY_WIDTH);  
    int block_idx_y = blockIdx.x / (imagewidth / PIX_KEY_WIDTH);   
    
    int idx_pixel = (block_idx_y * PIX_KEY_HEIGHT + block_y) * imagewidth + (block_idx_x * PIX_KEY_WIDTH + block_x);

    __shared__ unsigned char s_data[BLOCKSIZE * 3];
    __shared__ int s_key[BLOCKSIZE];

    for(int i = 0; i < 3; i++)
    {
    	s_data[threadIdx.x * 3 + i] = inputImage[idx_pixel * 3 + i];
    }

    s_key[threadIdx.x] = sub_key[threadIdx.x];
    __syncthreads();

    for(int i = 0; i < 3; i++)
    {
    	s_data[threadIdx.x * 3 + i] = s_data[threadIdx.x * 3 + i] ^ s_key[threadIdx.x];
		inputImage[idx_pixel * 3 + i] = s_data[threadIdx.x * 3+i];
    }
}

__device__ int col_calculator(int key, int col)
{
    int N = col;
    return key % N;
}

__device__ int row_calculator(int key, int row)
{   
    int N = row;
    return key / N + 1;
}

__device__ int de_key_generator(int col, int row, int width)
{   
    return row * width + col + 1; 
}

//Step 2: Pixel permutation within one block kernel
__global__ void pixel_permutation(unsigned char *inputImage, unsigned char *outputImage, int *en_key, int *de_key,int imagewidth)
{
    int block_x = threadIdx.x % PIX_KEY_WIDTH;
    int block_y = threadIdx.x / PIX_KEY_WIDTH;
    
    int block_idx_x = blockIdx.x % (imagewidth/PIX_KEY_WIDTH);   
    int block_idx_y = blockIdx.x / (imagewidth/PIX_KEY_WIDTH);   
    
    int idx_pixel = (block_idx_y * PIX_KEY_HEIGHT + block_y) * imagewidth + (block_idx_x * PIX_KEY_WIDTH + block_x);

    __shared__ unsigned char s_data[BLOCKSIZE*3];
    __shared__ unsigned char s_data_out[BLOCKSIZE*3];
    __shared__ int s_data_en_key[BLOCKSIZE];
    __shared__ int s_data_de_key[BLOCKSIZE];

    for(int i = 0; i < 3; i++)
    {
    	s_data[threadIdx.x * 3 + i] = inputImage[idx_pixel * 3 + i];
    }
    s_data_en_key[threadIdx.x] = en_key[threadIdx.x];
    __syncthreads();
  
    int key = s_data_en_key[threadIdx.x];
    int new_col = col_calculator(key, PIX_KEY_WIDTH);
    int new_row = row_calculator(key, PIX_KEY_WIDTH);
    if(new_col == 0)
    {
        new_col = PIX_KEY_WIDTH;
        new_row -= 1;
    }
    s_data_de_key[(new_row - 1) * PIX_KEY_WIDTH + (new_col - 1)] = de_key_generator(block_x, block_y, PIX_KEY_WIDTH);

    for(int i = 0; i < 3; i++)
    {
    	s_data_out[((new_row - 1) * PIX_KEY_WIDTH + (new_col - 1)) * 3 + i] = s_data[threadIdx.x * 3 + i];
    }
    __syncthreads();

    for(int i = 0; i < 3; i++)
    {
    	outputImage[idx_pixel * 3 + i] = s_data_out[threadIdx.x * 3 + i];
    }
    de_key[threadIdx.x] = s_data_de_key[threadIdx.x];
}

//Step 3: Block permutation kernel
__global__ void block_permutation(unsigned char *inputImage, unsigned char *outputImage, int *en_key, int *de_key,int imagewidth)
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

    de_key[(new_row_block - 1) * (imagewidth / PIX_KEY_WIDTH) + (new_col_block - 1)] = de_key_generator(block_idx_x, block_idx_y, imagewidth / PIX_KEY_WIDTH);
    for(int i = 0; i < 3; i++)
    {
        outputImage[(((new_row_block-1) * PIX_KEY_HEIGHT) * imagewidth + (new_col_block-1) * PIX_KEY_WIDTH +(block_y * imagewidth+ block_x)) * 3 + i] = inputImage[idx_pixel * 3 + i];
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
    int imagewidth = bmpInfoHeader.width;
    
    //Define two CUDA streams to divide the image into two parts with equal size
    int num_streams = 2;
    cudaStream_t streams[num_streams];

    //the size of each part
    int offset = (IMAGESIZE*3) / num_streams;
    for(int i = 0; i < num_streams; i++)
    {
    	cudaStreamCreate(&streams[i]);
    }

    int substitution_key[BLOCKSIZE];
    substitution_key_generator(substitution_key);
    int *d_substitution_key;

    int pix_perm_key[BLOCKSIZE];
    int *d_pix_per_key;
    int *d_pix_per_key_de;
    encryption_permutation_key_generator(pix_perm_key, BLOCKSIZE);
    randomize(pix_perm_key, BLOCKSIZE);

    int block_perm_key[IMAGESIZE/BLOCKSIZE];
    int *d_block_per_key;
    int *d_block_per_key_de;
    encryption_permutation_key_generator(block_perm_key,IMAGESIZE/BLOCKSIZE);
    randomize(block_perm_key, IMAGESIZE/BLOCKSIZE);
   
    //CUDA malloc and CUDA memcpy
    cudaMalloc((void**)&d_input_image, IMAGESIZE*3);
    cudaMalloc((void**)&d_pixel_permutation_image, IMAGESIZE*3);
    cudaMalloc((void**)&d_block_permutation_iamge, IMAGESIZE*3);

    cudaMalloc((void**)&d_substitution_key, BLOCKSIZE * sizeof(int));
    cudaMalloc((void**)&d_pix_per_key, BLOCKSIZE * sizeof(int));
    cudaMalloc((void**)&d_pix_per_key_de, BLOCKSIZE * sizeof(int));
    cudaMalloc((void**)&d_block_per_key, IMAGESIZE/BLOCKSIZE* sizeof(int));
    cudaMalloc((void**)&d_block_per_key_de, IMAGESIZE/BLOCKSIZE * sizeof(int));

    for(int i = 0; i < num_streams; i++)
    {
    	cudaMemcpyAsync(d_input_image + offset * i, input_image + offset * i, offset, cudaMemcpyHostToDevice, streams[i]);
    	cudaMemcpyAsync(d_pixel_permutation_image + offset * i, pixel_permutation_image + offset * i, offset, cudaMemcpyHostToDevice, streams[i]);
    }
    cudaMemcpy(d_block_permutation_iamge, block_permutation_iamge, IMAGESIZE*3, cudaMemcpyHostToDevice);

    cudaMemcpy(d_substitution_key, substitution_key, BLOCKSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pix_per_key, pix_perm_key, BLOCKSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_pix_per_key_de, 0, BLOCKSIZE * sizeof(int));
    cudaMemcpy(d_block_per_key, block_perm_key, IMAGESIZE/BLOCKSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_block_per_key_de, 0,IMAGESIZE/BLOCKSIZE * sizeof(int));

    //define the dimension of CUDA grid and block
    dim3 gridDim(IMAGESIZE/BLOCKSIZE , 1, 1);
    dim3 blockDim(BLOCKSIZE, 1, 1);

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t end;
    cudaEventCreate(&end);
    float en_time;
    float de_time;

    //Image Encryption using three CUDA kernels
    cudaEventRecord(start, 0);
    for(int i = 0; i < num_streams; i++)
    {
    	substitution<<<128, blockDim, 1024, streams[i]>>>(d_input_image + offset * i,d_substitution_key, imagewidth);
    	pixel_permutation<<<128, blockDim, 1024, streams[i]>>>(d_input_image + offset * i, d_pixel_permutation_image + offset * i,d_pix_per_key, d_pix_per_key_de, imagewidth);
    }

	block_permutation<<<gridDim, blockDim>>>(d_pixel_permutation_image, d_block_permutation_iamge, d_block_per_key, d_block_per_key_de, imagewidth);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&en_time, start, end);
    printf("Encryption Time: %fms\n", en_time);

    cudaMemcpy(block_permutation_iamge, d_block_permutation_iamge, IMAGESIZE*3, cudaMemcpyDeviceToHost);
    SaveImage(argv[2], block_permutation_iamge, &bmpFileHeader, &bmpInfoHeader);

    //load encrypted image to array
    output_image = LoadImage(argv[2],&bmpInfoHeader1, &bmpFileHeader1);
    out_block_permutation_iamge = (unsigned char *) malloc(bmpInfoHeader.imagesize * sizeof(char));
    memset(out_block_permutation_iamge, 0, IMAGESIZE);
    out_pixel_permutation_image = (unsigned char *) malloc(bmpInfoHeader.imagesize * sizeof(char));
    memset(out_pixel_permutation_image, 0, IMAGESIZE);

    cudaMalloc((void**)&d_output_image, IMAGESIZE*3);
    cudaMalloc((void**)&d_out_block_permutation_iamge, IMAGESIZE*3);
    cudaMalloc((void**)&d_out_pixel_permutation_image, IMAGESIZE*3);

    cudaMemcpy(d_output_image, output_image, IMAGESIZE*3, cudaMemcpyHostToDevice);
    for(int i = 0; i < num_streams; i++)
    {
    	cudaMemcpyAsync(d_out_block_permutation_iamge + offset * i, out_block_permutation_iamge + offset * i, offset, cudaMemcpyHostToDevice, streams[i]);
    	cudaMemcpyAsync(d_out_pixel_permutation_image + offset * i, out_pixel_permutation_image + offset * i, offset, cudaMemcpyHostToDevice, streams[i]);
    }

    //Image Decryption using three CUDA kernels
    cudaEventRecord(start, 0);
    block_permutation<<<gridDim, blockDim>>>(d_output_image, d_out_block_permutation_iamge, d_block_per_key_de, d_block_per_key,imagewidth);
    for(int i = 0; i < num_streams; i++)
    {
    	pixel_permutation<<<128, blockDim,1024, streams[i]>>>(d_out_block_permutation_iamge + offset * i, d_out_pixel_permutation_image + offset * i, d_pix_per_key_de, d_pix_per_key,imagewidth);
    	substitution<<<128, blockDim,1024, streams[i]>>>(d_out_pixel_permutation_image + offset * i, d_substitution_key,imagewidth);
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&de_time, start, end);
    printf("Decryption Time: %fms\n", de_time);

    cudaMemcpy(out_pixel_permutation_image, d_out_pixel_permutation_image, IMAGESIZE*3, cudaMemcpyDeviceToHost);

    SaveImage(argv[3], out_pixel_permutation_image, &bmpFileHeader1, &bmpInfoHeader1);

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
    return 0;
}
