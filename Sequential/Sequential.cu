#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "bitmap.h"

void substitution(unsigned char *bitmapImage, int size, int *key1)
{
    for(int i = 0; i < size; i++)
    {
        int idx = (i - (i % 3)) / 3;
        int idx_x = idx % 512;    //col
        int idx_y = idx / 512;    //row
        int block_x = idx_x % 32;
        int block_y = idx_y % 32;
        
        bitmapImage[i] = bitmapImage[i] ^ key1[block_y * 32 + block_x];
    }
}

int col_calculator(int key, int size)
{   
    int N = sqrt(size);
    return key % N;
}

int row_calculator(int key, int size)
{   
    int N = sqrt(size);
    return key / N + 1;
}

int de_key_generator(int col, int row, int size)
{   
    return row * sqrt(size) + col + 1; 
}

void pixel_permutation(unsigned char *bitmapImage, unsigned char *bitmapImage1, int size, int *key1, int *key2)
{
    for(int i = 0; i < size; i++)
    {
        int idx = (i - (i % 3)) / 3;
        int idx_x = idx % 512;          //col
        int idx_y = idx / 512;          //row
        int block_x = idx_x % 32;       //pixel position in block
        int block_y = idx_y % 32;
        int block_idx_x = idx_x / 32;   //block position
        int block_idx_y = idx_y / 32;

        int key = key1[block_y * 32 + block_x];
        int new_col = col_calculator(key, 1024);
        int new_row = row_calculator(key, 1024);
        if(new_col == 0)
        {
            new_col = 32;
            new_row -= 1;
        }

        int new_pos = (block_idx_y * 32 + new_row -1) * 512 + block_idx_x * 32 + new_col - 1;
        key2[(new_row - 1) * 32 + (new_col - 1)] = de_key_generator(block_x, block_y, 1024);
        bitmapImage1[new_pos * 3 + (i % 3)] = bitmapImage[i];    
    }
}

void block_permutation(unsigned char *bitmapImage1, unsigned char *bitmapImage2, int size, int *key1, int *key2)
{
    for(int i = 0; i < size; i++)
    {
        int idx = (i - (i % 3)) / 3;
        int idx_x = idx % 512;
        int idx_y = idx / 512;
        int block_x = idx_x % 32;
        int block_y = idx_y % 32;
        int block_idx_x = idx_x / 32;
        int block_idx_y = idx_y / 32;

        int key = key1[block_idx_y * 16 + block_idx_x];
        int new_col_block = col_calculator(key, 256);
        int new_row_block = row_calculator(key, 256);
        if(new_col_block == 0)
        {
            new_col_block = 16;
            new_row_block -= 1;
        }

        int new_pos = ((new_row_block - 1) * 32) * 512 + (new_col_block - 1) * 32 + (block_y * 512 + block_x);
        key2[(new_row_block - 1) * 16 + (new_col_block - 1)] = de_key_generator(block_idx_x, block_idx_y, 256);

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
    for(int i = 0; i < 1024; i++){
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

    int substitution_key[32 * 32];
    substitution_key_generator(substitution_key);

    int pix_perm_key[32 * 32];
    int pix_perm_key_de[32 * 32];
    encryption_permutation_key_generator(pix_perm_key, 32 * 32);
    randomize(pix_perm_key, 32 * 32);
   
    int block_perm_key[bitmapInfoHeader.biWidth * bitmapInfoHeader.biHeight/(32*32)];
    int block_perm_key_de[bitmapInfoHeader.biWidth * bitmapInfoHeader.biHeight/(32*32)];
    encryption_permutation_key_generator(block_perm_key, bitmapInfoHeader.biWidth * bitmapInfoHeader.biHeight/(32*32) );
    randomize(block_perm_key, bitmapInfoHeader.biWidth * bitmapInfoHeader.biHeight/(32*32));

    clock_t start, stop;

    start = clock();
    substitution(input_image, bitmapInfoHeader.biSizeImage, substitution_key);
    stop = clock();
    printf("Encryption substitution Time: %fms\n", ((double)(stop - start) / 1000));



    start = clock();
    pixel_permutation(input_image, pixel_permutation_image, bitmapInfoHeader.biSizeImage, pix_perm_key, pix_perm_key_de);
    stop = clock();
    printf("Encryption pixel permutation Time: %fms\n", ((double)(stop - start) / 1000));

    start = clock();
    block_permutation(pixel_permutation_image, block_permutation_iamge, bitmapInfoHeader.biSizeImage, block_perm_key, block_perm_key_de);
    stop = clock();
    printf("Encryption block permutation Time: %fms\n", ((double)(stop - start) / 1000));
    ReloadBitmapFile(argv[2], block_permutation_iamge, &bitmapFileHeader, &bitmapInfoHeader);

    // Image Decryption
    output_image = LoadBitmapFile(argv[2],&bitmapInfoHeader1, &bitmapFileHeader1);
    out_block_permutation_iamge = (unsigned char *) malloc(bitmapInfoHeader.biSizeImage * sizeof(char));
    out_pixel_permutation_image = (unsigned char *) malloc(bitmapInfoHeader.biSizeImage * sizeof(char));

    start = clock();
    block_permutation(output_image, out_block_permutation_iamge, bitmapInfoHeader.biSizeImage, block_perm_key_de, block_perm_key);
    stop = clock();
    printf("Decryption block permutation Time: %fms\n", ((double)(stop - start) / 1000));

    start = clock();
    pixel_permutation(out_block_permutation_iamge, out_pixel_permutation_image, bitmapInfoHeader.biSizeImage, pix_perm_key_de, pix_perm_key);
    stop = clock();
    printf("Decryption pixel permutation Time: %fms\n", ((double)(stop - start) / 1000));

    start = clock();
    substitution(out_pixel_permutation_image, bitmapInfoHeader.biSizeImage, substitution_key);
    stop = clock();
    printf("Decryption subtitution Time: %fms\n", ((double)(stop - start) / 1000));
    ReloadBitmapFile(argv[3], out_pixel_permutation_image, &bitmapFileHeader1, &bitmapInfoHeader1);

    free(input_image);
    free(pixel_permutation_image);
    free(block_permutation_iamge);
    free(output_image);
    free(out_block_permutation_iamge);
    free(out_pixel_permutation_image);

    return 0;
}
