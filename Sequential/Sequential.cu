/***
 Sequential version of image encryption based on
 pixel substitution and permutation.
 ***/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "bitmap.h"

//define blocksize
#define BLOCKSIZE 1024
#define PIX_KEY_WIDTH 32

//Step 1: Pixel substitution
void substitution(unsigned char *inputImage, int size, int *key1, int image_width)
{
    for(int i = 0; i < size; i++)
    {
        int idx = (i - (i % 3)) / 3;
        int idx_x = idx % image_width;    					//col position in image
        int idx_y = idx / image_width;    					//row position in image
        int block_x = idx_x % PIX_KEY_WIDTH; 				//col position in block
        int block_y = idx_y % (BLOCKSIZE/PIX_KEY_WIDTH);	//row position in block
        
        inputImage[i] = inputImage[i] ^ key1[block_y * PIX_KEY_WIDTH + block_x];
    }
}

int col_calculator(int key, int col)
{   
    int N = col;
    return key % N;
}

int row_calculator(int key, int col)
{   
    int N = col;
    return key / N + 1;
}

int de_key_generator(int col, int row, int width)
{   
    return row * width + col + 1;
}

//Step 2: Pixel permutation within one block
void pixel_permutation(unsigned char *inputImage, unsigned char *outputImage,
		int size, int *key1, int *key2, int image_width)
{
    for(int i = 0; i < size; i++)
    {
        int idx = (i - (i % 3)) / 3;
        int idx_x = idx % image_width;         					//col position in image
        int idx_y = idx / image_width;         					//row position in image
        int block_x = idx_x % PIX_KEY_WIDTH;       				//col position in block
        int block_y = idx_y % (BLOCKSIZE/PIX_KEY_WIDTH);		//row position in block
        int block_idx_x = idx_x / PIX_KEY_WIDTH;   				//block position in image
        int block_idx_y = idx_y / (BLOCKSIZE/PIX_KEY_WIDTH);

        int key = key1[block_y * PIX_KEY_WIDTH + block_x];
        int new_col = col_calculator(key, PIX_KEY_WIDTH);
        int new_row = row_calculator(key, PIX_KEY_WIDTH);
        if(new_col == 0)
        {
            new_col = PIX_KEY_WIDTH;
            new_row -= 1;
        }

        int new_pos = (block_idx_y * (BLOCKSIZE/PIX_KEY_WIDTH) + new_row -1) * image_width + block_idx_x * PIX_KEY_WIDTH + new_col - 1;
        key2[(new_row - 1) * PIX_KEY_WIDTH + (new_col - 1)] = de_key_generator(block_x, block_y, PIX_KEY_WIDTH);
        outputImage[new_pos * 3 + (i % 3)] = inputImage[i];
    }
}

//Step 3: Block permutation
void block_permutation(unsigned char *inputImage, unsigned char *outputImage,
		int size, int *key1, int *key2, int image_width, int image_height)
{
    for(int i = 0; i < size; i++)
    {
        int idx = (i - (i % 3)) / 3;
        int idx_x = idx % image_width;
        int idx_y = idx / image_width;
        int block_x = idx_x % PIX_KEY_WIDTH;
        int block_y = idx_y % (BLOCKSIZE/PIX_KEY_WIDTH);
        int block_idx_x = idx_x / PIX_KEY_WIDTH;
        int block_idx_y = idx_y / (BLOCKSIZE/PIX_KEY_WIDTH);

        int key = key1[block_idx_y * (image_width/PIX_KEY_WIDTH) + block_idx_x];
        int new_col_block = col_calculator(key, image_width/PIX_KEY_WIDTH);
        int new_row_block = row_calculator(key, image_width/PIX_KEY_WIDTH);
        if(new_col_block == 0)
        {
            new_col_block = image_width/PIX_KEY_WIDTH;
            new_row_block -= 1;
        }

        int new_pos = ((new_row_block - 1) * (BLOCKSIZE/PIX_KEY_WIDTH)) * image_width + (new_col_block - 1) * PIX_KEY_WIDTH + (block_y * image_width + block_x);
        key2[(new_row_block - 1) * image_width/PIX_KEY_WIDTH + (new_col_block - 1)]
             = de_key_generator(block_idx_x, block_idx_y, image_width/PIX_KEY_WIDTH);

        outputImage[new_pos * 3 + (i % 3)] = inputImage[i];
    }
}

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

    //Loading input image and allocate enough memory for processing
    input_image = LoadImage(argv[1],&bmpInfoHeader, &bmpFileHeader);
    pixel_permutation_image = (unsigned char *) malloc(bmpInfoHeader.imagesize * sizeof(char));
    block_permutation_iamge = (unsigned char *) malloc(bmpInfoHeader.imagesize * sizeof(char));

    printf("Image size: %d\n", bmpInfoHeader.imagesize);
    printf("Image width: %d\n", bmpInfoHeader.width);
    printf("Image height: %d\n", bmpInfoHeader.height);

    //Generate key for the three step
    int substitution_key[BLOCKSIZE];
    substitution_key_generator(substitution_key);

    int pix_perm_key[BLOCKSIZE];
    int pix_perm_key_de[BLOCKSIZE];
    encryption_permutation_key_generator(pix_perm_key, BLOCKSIZE);
    randomize(pix_perm_key, BLOCKSIZE);

    int block_perm_key[bmpInfoHeader.width * bmpInfoHeader.height/BLOCKSIZE];
    int block_perm_key_de[bmpInfoHeader.width * bmpInfoHeader.height/BLOCKSIZE];
    encryption_permutation_key_generator(block_perm_key, bmpInfoHeader.width * bmpInfoHeader.height/BLOCKSIZE);
    randomize(block_perm_key, bmpInfoHeader.width * bmpInfoHeader.height/BLOCKSIZE);

    clock_t start, stop;
    double en_sub, en_pix_per, en_block_per;
    double de_sub, de_pix_per, de_block_per;

    //Image Encryption
    start = clock();
    substitution(input_image, bmpInfoHeader.imagesize, substitution_key, bmpInfoHeader.width);
    stop = clock();
    en_sub = ((double)(stop - start) / 1000);
    printf("Encryption substitution Time: %fms\n", en_sub);

    start = clock();
    pixel_permutation(input_image, pixel_permutation_image, bmpInfoHeader.imagesize,
    		pix_perm_key, pix_perm_key_de, bmpInfoHeader.width);
    stop = clock();
    en_pix_per = ((double)(stop - start) / 1000);
    printf("Encryption pixel permutation Time: %fms\n", en_pix_per);

    start = clock();
    block_permutation(pixel_permutation_image, block_permutation_iamge, bmpInfoHeader.imagesize,
    		block_perm_key, block_perm_key_de, bmpInfoHeader.width, bmpInfoHeader.height);
    stop = clock();
    en_block_per = ((double)(stop - start) / 1000);
    printf("Encryption block permutation Time: %fms\n", en_block_per);
    printf("Encryption time: %fms\n", en_sub + en_pix_per + en_block_per);
    SaveImage(argv[2], block_permutation_iamge, &bmpFileHeader, &bmpInfoHeader);

    //Loading output image and allocate enough memory for processing
    output_image = LoadImage(argv[2],&bmpInfoHeader1, &bmpFileHeader1);
    out_block_permutation_iamge = (unsigned char *) malloc(bmpInfoHeader.imagesize * sizeof(char));
    out_pixel_permutation_image = (unsigned char *) malloc(bmpInfoHeader.imagesize * sizeof(char));

    // Image Decryption
    start = clock();
    block_permutation(output_image, out_block_permutation_iamge, bmpInfoHeader.imagesize,
    		block_perm_key_de, block_perm_key, bmpInfoHeader.width, bmpInfoHeader.height);
    stop = clock();
    de_block_per = ((double)(stop - start) / 1000);
    printf("Decryption block permutation Time: %fms\n", de_block_per);

    start = clock();
    pixel_permutation(out_block_permutation_iamge, out_pixel_permutation_image, bmpInfoHeader.imagesize,
    		pix_perm_key_de, pix_perm_key, bmpInfoHeader.width);
    stop = clock();
    de_pix_per = ((double)(stop - start) / 1000);
    printf("Decryption pixel permutation Time: %fms\n", de_pix_per);

    start = clock();
    substitution(out_pixel_permutation_image, bmpInfoHeader.imagesize, substitution_key, bmpInfoHeader.width);
    stop = clock();
    de_sub = ((double)(stop - start) / 1000);
    printf("Decryption subtitution Time: %fms\n", de_sub);
    printf("Decryption time: %fms\n", de_sub + de_pix_per + de_block_per);
    SaveImage(argv[3], out_pixel_permutation_image, &bmpFileHeader1, &bmpInfoHeader1);

    free(input_image);
    free(pixel_permutation_image);
    free(block_permutation_iamge);
    free(output_image);
    free(out_block_permutation_iamge);
    free(out_pixel_permutation_image);

    return 0;
}
