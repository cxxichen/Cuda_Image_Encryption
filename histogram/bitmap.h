/***
	bmp fie header
***/

#ifndef _BITMAP_H_
#define _BITMAP_H_

#pragma pack(push, 1)
typedef struct 
{
    unsigned short int type;
    unsigned int size;  //File size in bytes
    unsigned short int reserved1;
    unsigned short int reserved2;
    unsigned int offset;  //offset to image data
} FILEHEADER;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct 
{
    unsigned int size;  //Header size in bytes
    int width;  //Width of image
    int height;  //Height of image
    unsigned short int planes; //numbers of colours per plane
    unsigned short int bits; //number of bits per pixel
    unsigned int compression;//Compression type
    unsigned int imagesize;  //Image size in bytes
    int xresolution;  //Pixels per meter in x axis
    int yresolution;  //Pixels per meter in y axis
    unsigned int ncolours;  //Number of colors
    unsigned int importantcolours;  //Important colours
} INFOHEADER;
#pragma pack(pop)

extern unsigned char *LoadImage(char *filename, INFOHEADER *InfoHeader, FILEHEADER *FileHeader);
extern void SaveImage(char *filename, unsigned char *outImage, FILEHEADER *FileHeader, INFOHEADER *InfoHeader);

#endif
