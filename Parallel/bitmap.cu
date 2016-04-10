#include <stdio.h>
#include <stdlib.h>
#include "bitmap.h"

unsigned char *LoadBitmapFile(char *filename, BITMAPINFOHEADER *bitmapInfoHeader, BITMAPFILEHEADER *bitmapFileHeader)
{
    FILE *filePtr; //our file pointer
    unsigned char *bitmapImage;  //store image data

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

    fclose(filePtr);
    return bitmapImage;
}

void ReloadBitmapFile(char *filename, unsigned char *bitmapImage, BITMAPFILEHEADER *bitmapFileHeader, BITMAPINFOHEADER *bitmapInfoHeader)
{
    FILE *filePtr; //our file pointer

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

    //write in the bitmap image data
    fwrite(bitmapImage,bitmapInfoHeader->biSizeImage,1,filePtr);

    //close file
    fclose(filePtr);
}

