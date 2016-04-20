#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "bitmap.h"

#define BLOCKSIZE 1024
#define PIX_KEY_WIDTH 32

void substitution(unsigned char *bitmapImage, int size, int *key1, int image_width)
{
    for(int i = 0; i < size; i++)
    {
        int idx = (i - (i % 3)) / 3;
        int idx_x = idx % image_width;    //col
        int idx_y = idx / image_width;    //row
        int block_x = idx_x % PIX_KEY_WIDTH;
        int block_y = idx_y % (BLOCKSIZE/PIX_KEY_WIDTH);
        
        bitmapImage[i] = bitmapImage[i] ^ key1[block_y * PIX_KEY_WIDTH + block_x];
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

void pixel_permutation(unsigned char *bitmapImage, unsigned char *bitmapImage1, int size, int *key1, int *key2, int image_width)
{
    for(int i = 0; i < size; i++)
    {
        int idx = (i - (i % 3)) / 3;
        int idx_x = idx % image_width;          //col
        int idx_y = idx / image_width;          //row
        int block_x = idx_x % PIX_KEY_WIDTH;       //pixel position in block
        int block_y = idx_y % (BLOCKSIZE/PIX_KEY_WIDTH);
        int block_idx_x = idx_x / PIX_KEY_WIDTH;   //block position
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
        bitmapImage1[new_pos * 3 + (i % 3)] = bitmapImage[i];    
    }
}

void block_permutation(unsigned char *bitmapImage1, unsigned char *bitmapImage2, int size, int *key1, int *key2, int image_width, int image_height)
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
        key2[(new_row_block - 1) * image_width/PIX_KEY_WIDTH + (new_col_block - 1)] = de_key_generator(block_idx_x, block_idx_y, image_width/PIX_KEY_WIDTH);

        bitmapImage2[new_pos * 3 + (i % 3)] = bitmapImage1[i];
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
    // Use a different seed value so that we don't get same
    // result each time we run this program
    srand ( time(NULL) );
 
    // Start from the last element and swap one by one. We don't
    // need to run for the first element that's why i > 0
    for (int i = n-1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        int j = rand() % (i+1);
        // Swap arr[i] with the element at random index
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
    BITMAPINFOHEADER bitmapInfoHeader;
    BITMAPFILEHEADER bitmapFileHeader;
    unsigned char *input_image;
    unsigned char *pixel_permutation_image;
    unsigned char *block_permutation_iamge;

    BITMAPINFOHEADER bitmapInfoHeader1;
    BITMAPFILEHEADER bitmapFileHeader1;
    unsigned char *output_image;
    unsigned char *out_block_permutation_iamge;
    unsigned char *out_pixel_permutation_image;


    input_image = LoadBitmapFile(argv[1],&bitmapInfoHeader, &bitmapFileHeader);
    pixel_permutation_image = (unsigned char *) malloc(bitmapInfoHeader.biSizeImage * sizeof(char));
    block_permutation_iamge = (unsigned char *) malloc(bitmapInfoHeader.biSizeImage * sizeof(char));

    printf("Image size: %d\n", bitmapInfoHeader.biSizeImage);
    printf("Image width: %d\n", bitmapInfoHeader.biWidth);
    printf("Image height: %d\n", bitmapInfoHeader.biHeight);

    int substitution_key[BLOCKSIZE];
    substitution_key_generator(substitution_key);

    int pix_perm_key[BLOCKSIZE];
    int pix_perm_key_de[BLOCKSIZE];
    encryption_permutation_key_generator(pix_perm_key, BLOCKSIZE);
    randomize(pix_perm_key, BLOCKSIZE);

    int block_perm_key[bitmapInfoHeader.biWidth * bitmapInfoHeader.biHeight/BLOCKSIZE];
    int block_perm_key_de[bitmapInfoHeader.biWidth * bitmapInfoHeader.biHeight/BLOCKSIZE];
    encryption_permutation_key_generator(block_perm_key, bitmapInfoHeader.biWidth * bitmapInfoHeader.biHeight/BLOCKSIZE);
    randomize(block_perm_key, bitmapInfoHeader.biWidth * bitmapInfoHeader.biHeight/BLOCKSIZE);

    clock_t start, stop;
    double en_sub, en_pix_per, en_block_per;
    double de_sub, de_pix_per, de_block_per;

    start = clock();
    substitution(input_image, bitmapInfoHeader.biSizeImage, substitution_key, bitmapInfoHeader.biWidth);
    stop = clock();
    en_sub = ((double)(stop - start) / 1000);
    printf("Encryption substitution Time: %fms\n", en_sub);

    start = clock();
    pixel_permutation(input_image, pixel_permutation_image, bitmapInfoHeader.biSizeImage, pix_perm_key, pix_perm_key_de, bitmapInfoHeader.biWidth);
    stop = clock();
    en_pix_per = ((double)(stop - start) / 1000);
    printf("Encryption pixel permutation Time: %fms\n", en_pix_per);

    start = clock();
    block_permutation(pixel_permutation_image, block_permutation_iamge, bitmapInfoHeader.biSizeImage, block_perm_key, block_perm_key_de, bitmapInfoHeader.biWidth, bitmapInfoHeader.biHeight);
    stop = clock();
    en_block_per = ((double)(stop - start) / 1000);
    printf("Encryption block permutation Time: %fms\n", en_block_per);
    printf("Encryption time: %fms\n", en_sub + en_pix_per + en_block_per);
    ReloadBitmapFile(argv[2], block_permutation_iamge, &bitmapFileHeader, &bitmapInfoHeader);

    // Image Decryption
    output_image = LoadBitmapFile(argv[2],&bitmapInfoHeader1, &bitmapFileHeader1);
    out_block_permutation_iamge = (unsigned char *) malloc(bitmapInfoHeader.biSizeImage * sizeof(char));
    out_pixel_permutation_image = (unsigned char *) malloc(bitmapInfoHeader.biSizeImage * sizeof(char));

    start = clock();
    block_permutation(output_image, out_block_permutation_iamge, bitmapInfoHeader.biSizeImage, block_perm_key_de, block_perm_key, bitmapInfoHeader.biWidth, bitmapInfoHeader.biHeight);
    stop = clock();
    de_block_per = ((double)(stop - start) / 1000);
    printf("Decryption block permutation Time: %fms\n", de_block_per);

    start = clock();
    pixel_permutation(out_block_permutation_iamge, out_pixel_permutation_image, bitmapInfoHeader.biSizeImage, pix_perm_key_de, pix_perm_key, bitmapInfoHeader.biWidth);
    stop = clock();
    de_pix_per = ((double)(stop - start) / 1000);
    printf("Decryption pixel permutation Time: %fms\n", de_pix_per);

    start = clock();
    substitution(out_pixel_permutation_image, bitmapInfoHeader.biSizeImage, substitution_key, bitmapInfoHeader.biWidth);
    stop = clock();
    de_sub = ((double)(stop - start) / 1000);
    printf("Decryption subtitution Time: %fms\n", de_sub);
    printf("Decryption time: %fms\n", de_sub + de_pix_per + de_block_per);
    ReloadBitmapFile(argv[3], out_pixel_permutation_image, &bitmapFileHeader1, &bitmapInfoHeader1);

    free(input_image);
    free(pixel_permutation_image);
    free(block_permutation_iamge);
    free(output_image);
    free(out_block_permutation_iamge);
    free(out_pixel_permutation_image);

    return 0;
}
