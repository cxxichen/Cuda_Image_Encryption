#ifndef _BITMAP_H_
#define _BITMAP_H_

typedef short WORD;
typedef int DWORD;
typedef int LONG;

#pragma pack(push, 1)
typedef struct 
{
    WORD bfType;  //specifies the file type
    DWORD bfSize;  //specifies the size in bytes of the bitmap file
    WORD bfReserved1;  //reserved; must be 0
    WORD bfReserved2;  //reserved; must be 0
    DWORD bOffBits;  //species the offset in bytes from the bitmapfileheader to the bitmap bits
} BITMAPFILEHEADER;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct 
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
} BITMAPINFOHEADER;
#pragma pack(pop)

extern unsigned char *LoadBitmapFile(char *filename, BITMAPINFOHEADER *bitmapInfoHeader, BITMAPFILEHEADER *bitmapFileHeader);
extern void ReloadBitmapFile(char *filename, unsigned char *bitmapImage, BITMAPFILEHEADER *bitmapFileHeader, BITMAPINFOHEADER *bitmapInfoHeader);

#endif
