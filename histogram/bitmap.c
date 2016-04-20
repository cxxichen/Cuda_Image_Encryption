#include <stdio.h>
#include <stdlib.h>
#include "bitmap.h"

unsigned char *LoadImage(char *filename, INFOHEADER *bitmapInfoHeader, FILEHEADER *FileHeader)
{
    FILE *fp;
    unsigned char *inputImage;

    //read the input image
    fp = fopen(filename,"rb");
    if (fp == NULL)
    {
        printf("\nERROR: Cannot open file %s", filename);
        return NULL;
    }

    //read the file header
    fread(FileHeader, sizeof(FILEHEADER),1,fp);

    //verify that this is a bmp file by check bitmap id
    if (FileHeader->type !=0x4D42)
    {
        printf("The image is not a bmp file");
        fclose(fp);
        return NULL;
    }
    
    //read the bmp info header
    fread(bitmapInfoHeader, sizeof(INFOHEADER),1,fp);


    fseek(fp, FileHeader->offset, SEEK_SET);
    //allocate enough memory for the input data
    inputImage = (unsigned char*)malloc(bitmapInfoHeader->imagesize);

    //verify memory allocation
    if (!inputImage)
    {
        free(inputImage);
        fclose(fp);
        return NULL;
    }

    //read in the bitmap image data
    fread(inputImage,1,bitmapInfoHeader->imagesize,fp);

    if (inputImage == NULL)
    {
        printf("Failed to load the input image");
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return inputImage;
}

void SaveImage(char *filename, unsigned char *outImage, FILEHEADER *FileHeader, INFOHEADER *bitmapInfoHeader)
{
    FILE *fp;
    fp = fopen(filename,"wb");
    if (fp == NULL)
    {
        printf("\nERROR: Cannot open file %s", filename);
        exit(1);
    }
        
    fwrite(FileHeader, sizeof(FILEHEADER),1,fp);
    fwrite(bitmapInfoHeader, sizeof(INFOHEADER),1,fp);
    fwrite(outImage,bitmapInfoHeader->imagesize,1,fp);
    fclose(fp);
}

