#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef short WORD;
typedef int DWORD;
typedef int LONG;

#pragma pack(push, 1)
typedef struct tagBITMAPFILEHEADER
{
    WORD bfType;  //specifies the file type
    DWORD bfSize;  //specifies the size in bytes of the bitmap file
    WORD bfReserved1;  //reserved; must be 0
    WORD bfReserved2;  //reserved; must be 0
    DWORD bOffBits;  //species the offset in bytes from the bitmapfileheader to the bitmap bits
}BITMAPFILEHEADER;
#pragma pack(pop)


#pragma pack(push, 1)
typedef struct tagBITMAPINFOHEADER
{
    DWORD biSize;  //specifies the number of bytes required by the struct
    LONG biWidth;  //specifies width in pixels
    LONG biHeight;  //species height in pixels
    WORD biPlanes; //specifies the number of color planes, must be 1
    WORD biBitCount; //specifies the number of bit per pixel
    DWORD biCompression;//spcifies the type of compression
    DWORD biSizeImage;  //size of image in bytes
    LONG biXPelsPerMeter;  //number of pixels per meter in x axis
    LONG biYPelsPerMeter;  //number of pixels per meter in y axis
    DWORD biClrUsed;  //number of colors used by th ebitmap
    DWORD biClrImportant;  //number of colors that are important
}BITMAPINFOHEADER;
#pragma pack(pop)

unsigned char *LoadBitmapFile(char *filename, BITMAPINFOHEADER *bitmapInfoHeader, BITMAPFILEHEADER *bitmapFileHeader)
{
    FILE *filePtr; //our file pointer
    unsigned char *bitmapImage;  //store image data
    int imageIdx=0;  //image index counter
    unsigned char tempRGB;  //our swap variable

    //open filename in read binary mode
    filePtr = fopen(filename,"rb");
    if (filePtr == NULL)
        return NULL;

    //read the bitmap file header
    fread(bitmapFileHeader, sizeof(BITMAPFILEHEADER),1,filePtr);

    
    //verify that this is a bmp file by check bitmap id
    if (bitmapFileHeader->bfType !=0x4D42)
    {
        fclose(filePtr);
        return NULL;
    }
    
    //read the bitmap info header
    fread(bitmapInfoHeader, sizeof(BITMAPINFOHEADER),1,filePtr); // small edit. forgot to add the closing bracket at sizeof

    //move file point to the begging of bitmap data
    fseek(filePtr, bitmapFileHeader->bOffBits, SEEK_SET);

    //allocate enough memory for the bitmap image data
    bitmapImage = (unsigned char*)malloc(bitmapInfoHeader->biSizeImage);

    //verify memory allocation
    if (!bitmapImage)
    {
        free(bitmapImage);
        fclose(filePtr);
        return NULL;
    }

    //read in the bitmap image data
    fread(bitmapImage,1,bitmapInfoHeader->biSizeImage,filePtr);

    //make sure bitmap image data was read
    if (bitmapImage == NULL)
    {
        fclose(filePtr);
        return NULL;
    }

    for (imageIdx = 0; imageIdx < bitmapInfoHeader->biSizeImage;imageIdx+=3)
    {
        tempRGB = bitmapImage[imageIdx];
        bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
        bitmapImage[imageIdx + 2] = tempRGB;
    }
    fclose(filePtr);
    return bitmapImage;
}

void ReloadBitmapFile(char *filename, unsigned char *bitmapImage, BITMAPFILEHEADER *bitmapFileHeader, BITMAPINFOHEADER *bitmapInfoHeader)
{
    FILE *filePtr; //our file pointer
    int imageIdx=0;  //image index counter
    unsigned char tempRGB;  //our swap variable

    //open filename in write binary mode
    filePtr = fopen(filename,"wb");
    if (filePtr == NULL)
    {
        printf("\nERROR: Cannot open file %s", filename);
        exit(1);
    }
        

    //write the bitmap file header
    fwrite(bitmapFileHeader, sizeof(BITMAPFILEHEADER),1,filePtr);

    //write the bitmap info header
    fwrite(bitmapInfoHeader, sizeof(BITMAPINFOHEADER),1,filePtr); // small edit. forgot to add the closing bracket at sizeof

    //swap the r and b values to get RGB (bitmap is BGR)
    for (imageIdx = 0; imageIdx < bitmapInfoHeader->biSizeImage;imageIdx+=3)
    {
        tempRGB = bitmapImage[imageIdx];
        bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
        bitmapImage[imageIdx + 2] = tempRGB;
    }

    //write in the bitmap image data
    fwrite(bitmapImage,bitmapInfoHeader->biSizeImage,1,filePtr);

    //close file
    fclose(filePtr);
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

__device__ int col_calculator(int, int);
__device__ int row_calculator(int, int);
__device__ int de_key_generator(int, int, int);

__global__ void substitution(unsigned char *bitmapImage, int *sub_key)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (index - index % 3) / 3;
    int idx_x = idx % 512;    //col
    int idx_y = idx / 512;    //row
    int block_x = idx_x % 32;
    int block_y = idx_y % 32;
    bitmapImage[index] = bitmapImage[index] ^ sub_key[block_y * 32 + block_x];
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

__device__ int de_key_generator(int col, int row, int size)
{   
    return row * size + col + 1; 
}

__global__ void pixel_permutation(unsigned char *bitmapImage, unsigned char *bitmapImage1, int *en_key, int *de_key)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (index - index % 3) / 3;
    int idx_x = idx % 512;  //col, 0-511
    int idx_y = idx / 512;  //row, 0-511
    int block_x = idx_x % 32;   //col in block, 0-31
    int block_y = idx_y % 32;   //row in block, 0-31
    int block_idx_x = idx_x / 32;   //block position, 0-15
    int block_idx_y = idx_y / 32;   //0-15

    int key = en_key[block_y * 32 + block_x];
    int new_col = col_calculator(key, 32);
    int new_row = row_calculator(key, 32);
    if(new_col == 0)
    {
        new_col = 32;
        new_row -= 1;
    }

    de_key[(new_row - 1) * 32 + (new_col - 1)] = de_key_generator(block_x, block_y, 32);
    bitmapImage1[((block_idx_y * 32 + new_row - 1) * 512 + block_idx_x * 32 + new_col - 1) * 3 + (index % 3)] = bitmapImage[index];
}


__global__ void block_permutation(unsigned char *bitmapImage1, unsigned char *bitmapImage2, int *en_key, int *de_key)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (index - index % 3) / 3;
    int idx_x = idx % 512;
    int idx_y = idx / 512;
    int block_x = idx_x % 32;
    int block_y = idx_y % 32;
    int block_idx_x = idx_x / 32;
    int block_idx_y = idx_y / 32;

    int key = en_key[block_idx_y * 16 + block_idx_x];
    int new_col_block = col_calculator(key, 16);
    int new_row_block = row_calculator(key, 16);
    if(new_col_block == 0)
    {
        new_col_block = 16;
        new_row_block -= 1;
    }

    de_key[(new_row_block - 1) * 16 + (new_col_block - 1)] = de_key_generator(block_idx_x, block_idx_y, 16);
    bitmapImage2[(((new_row_block-1) * 32) * 512 + (new_col_block-1) * 32 + (block_y * 512 + block_x)) * 3 + (index % 3)] = bitmapImage1[index];
}


int main()
{
    BITMAPINFOHEADER bitmapInfoHeader;
    BITMAPFILEHEADER bitmapFileHeader;
    BITMAPINFOHEADER bitmapInfoHeader1;
    BITMAPFILEHEADER bitmapFileHeader1;
    BITMAPINFOHEADER bitmapInfoHeader2;
    BITMAPFILEHEADER bitmapFileHeader2;
    BITMAPINFOHEADER bitmapInfoHeader3;
    BITMAPFILEHEADER bitmapFileHeader3;
    BITMAPINFOHEADER bitmapInfoHeader4;
    BITMAPFILEHEADER bitmapFileHeader4;
    BITMAPINFOHEADER bitmapInfoHeader5;
    BITMAPFILEHEADER bitmapFileHeader5;
    unsigned char *bitmapData;
    unsigned char *bitmapData1;
    unsigned char *bitmapData2;
    unsigned char *bitmapData3;
    unsigned char *bitmapData4;
    unsigned char *bitmapData5;

    unsigned char *d_bitmapData;
    unsigned char *d_bitmapData1;
    unsigned char *d_bitmapData2;
    unsigned char *d_bitmapData3;
    unsigned char *d_bitmapData4;
    unsigned char *d_bitmapData5;
    
    int substitution_key[32 * 32];
    substitution_key_generator(substitution_key);
    int *d_substitution_key;

    int pix_perm_key[32 * 32];
    int pix_perm_key_de[32 * 32];
    int *d_pix_per_key;
    int *d_pix_per_key_de;
    encryption_permutation_key_generator(pix_perm_key, 32 * 32);
    randomize(pix_perm_key, 32 * 32);

    int block_perm_key[16 * 16];
    int block_perm_key_de[16 * 16];
    int *d_block_per_key;
    int *d_block_per_key_de;
    encryption_permutation_key_generator(block_perm_key, 16 * 16);
    randomize(block_perm_key, 256);

    bitmapData = LoadBitmapFile("lena.bmp", &bitmapInfoHeader, &bitmapFileHeader);
    bitmapData1 = LoadBitmapFile("lena.bmp", &bitmapInfoHeader1, &bitmapFileHeader1);
    memset(bitmapData1, 0, bitmapInfoHeader.biSizeImage);
    bitmapData2 = LoadBitmapFile("lena.bmp",&bitmapInfoHeader2, &bitmapFileHeader2);
    memset(bitmapData2, 0, bitmapInfoHeader.biSizeImage);

    printf("Image size: %d\n", bitmapInfoHeader.biSizeImage);
    printf("Image widthh: %d\n", bitmapInfoHeader.biWidth);
    printf("Image height: %d\n", bitmapInfoHeader.biHeight);

    cudaMalloc((void**)&d_bitmapData, bitmapInfoHeader.biSizeImage);
    cudaMalloc((void**)&d_bitmapData1, bitmapInfoHeader.biSizeImage);
    cudaMalloc((void**)&d_bitmapData2, bitmapInfoHeader.biSizeImage);

    cudaMalloc((void**)&d_substitution_key, 1024 * sizeof(int));
    cudaMalloc((void**)&d_pix_per_key, 1024 * sizeof(int));
    cudaMalloc((void**)&d_pix_per_key_de, 1024 * sizeof(int));
    cudaMalloc((void**)&d_block_per_key, 256 * sizeof(int));
    cudaMalloc((void**)&d_block_per_key_de, 256 * sizeof(int));

    cudaMemcpy(d_bitmapData, bitmapData, bitmapInfoHeader.biSizeImage, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bitmapData1, bitmapData1, bitmapInfoHeader.biSizeImage, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bitmapData2, bitmapData2, bitmapInfoHeader.biSizeImage, cudaMemcpyHostToDevice);

    cudaMemcpy(d_substitution_key, substitution_key, 1024 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pix_per_key, pix_perm_key, 1024 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_pix_per_key_de, 0, 1024 * sizeof(int));
    cudaMemcpy(d_block_per_key, block_perm_key, 256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_block_per_key_de, 0, 256 * sizeof(int));

    dim3 gridDim(256 * 3, 1, 1);
    dim3 blockDim(1024, 1, 1);

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t end;
    cudaEventCreate(&end);
    float en_substitution_time, en_pixel_permutation_time, en_block_permutation_time;
    float de_substitution_time, de_pixel_permutation_time, de_block_permutation_time;

    cudaEventRecord(start, 0);
    substitution<<<gridDim, blockDim>>>(d_bitmapData, d_substitution_key);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&en_substitution_time, start, end);
    printf("Encryption substitution Time: %fms\n", en_substitution_time);

    cudaEventRecord(start, 0);
    pixel_permutation<<<gridDim, blockDim>>>(d_bitmapData, d_bitmapData1, d_pix_per_key, d_pix_per_key_de);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&en_pixel_permutation_time, start, end);
    printf("Encryption pixel permutation Time: %fms\n", en_pixel_permutation_time);

    cudaEventRecord(start, 0);
    block_permutation<<<gridDim, blockDim>>>(d_bitmapData1, d_bitmapData2, d_block_per_key, d_block_per_key_de);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&en_block_permutation_time, start, end);
    printf("Encryption block permutation Time: %fms\n", en_block_permutation_time);

    cudaMemcpy(bitmapData2, d_bitmapData2, bitmapInfoHeader.biSizeImage, cudaMemcpyDeviceToHost);
    ReloadBitmapFile("encrypted.bmp", bitmapData2, &bitmapFileHeader2, &bitmapInfoHeader2);

    //load encrypted image to array
    bitmapData3 = LoadBitmapFile("encrypted.bmp",&bitmapInfoHeader3, &bitmapFileHeader3);
    bitmapData4 = LoadBitmapFile("encrypted.bmp",&bitmapInfoHeader4, &bitmapFileHeader4);
    memset(bitmapData4, 0, bitmapInfoHeader.biSizeImage);
    bitmapData5 = LoadBitmapFile("encrypted.bmp", &bitmapInfoHeader5, &bitmapFileHeader5);
    memset(bitmapData5, 0, bitmapInfoHeader.biSizeImage);

    cudaMalloc((void**)&d_bitmapData3, bitmapInfoHeader.biSizeImage);
    cudaMalloc((void**)&d_bitmapData4, bitmapInfoHeader.biSizeImage);
    cudaMalloc((void**)&d_bitmapData5, bitmapInfoHeader.biSizeImage);

    cudaMemcpy(d_bitmapData3, bitmapData3, bitmapInfoHeader.biSizeImage, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bitmapData4, bitmapData4, bitmapInfoHeader.biSizeImage, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bitmapData5, bitmapData5, bitmapInfoHeader.biSizeImage, cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    block_permutation<<<gridDim, blockDim>>>(d_bitmapData3, d_bitmapData4, d_block_per_key_de, d_block_per_key);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&de_block_permutation_time, start, end);
    printf("Decryption block permutation Time: %fms\n", de_block_permutation_time);

    cudaEventRecord(start, 0);
    pixel_permutation<<<gridDim, blockDim>>>(d_bitmapData4, d_bitmapData5, d_pix_per_key_de, d_pix_per_key);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&de_pixel_permutation_time, start, end);
    printf("Decryption pixel permutation Time: %fms\n", de_pixel_permutation_time);

    cudaEventRecord(start, 0);
    substitution<<<gridDim, blockDim>>>(d_bitmapData5, d_substitution_key);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&de_substitution_time, start, end);
    printf("Decryption substitution Time: %fms\n", de_substitution_time);

    cudaMemcpy(bitmapData5, d_bitmapData5, bitmapInfoHeader.biSizeImage, cudaMemcpyDeviceToHost);
    ReloadBitmapFile("Decrypted.bmp", bitmapData5, &bitmapFileHeader5, &bitmapInfoHeader5);

    cudaFree(d_bitmapData);
    cudaFree(d_bitmapData1);
    cudaFree(d_bitmapData2);
    cudaFree(d_bitmapData3);
    cudaFree(d_bitmapData4);
    cudaFree(d_bitmapData5);

    cudaFree(d_substitution_key);
    cudaFree(d_pix_per_key);
    cudaFree(d_pix_per_key_de);
    cudaFree(d_block_per_key);
    cudaFree(d_block_per_key_de);

    return 0;
}