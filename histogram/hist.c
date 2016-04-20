#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bitmap.h"
#include <string.h>


int main(int argc, char *argv[])
{
    char filename[100], out_r[100], out_b[100], out_g[100];
    int n = 0;
    strcpy(filename, argv[1]);
    n = strlen(filename);
    strcpy(out_r, filename);
    strcpy(out_r+n-4, "_r.txt");
    strcpy(out_g, filename);
    strcpy(out_g+n-4, "_g.txt");
    strcpy(out_b, filename);
    strcpy(out_b+n-4, "_b.txt");

    INFOHEADER bmpInfoHeader;
    FILEHEADER bmpFileHeader;
    unsigned char *image;
    image = LoadImage(filename,&bmpInfoHeader, &bmpFileHeader);   

    int hist_R[256];
    int hist_G[256];
    int hist_B[256];
    for(int i = 0; i < 256; i++)
    {
        hist_R[i] = 0;
        hist_G[i] = 0;
        hist_B[i] = 0;
    }

    for(int i = 0; i < bmpInfoHeader.imagesize; i++)
    {
        int channel = i % 3;
        switch(channel)
        {
            case 0:
                hist_B[image[i]]++;
                break;
            case 1:
                hist_G[image[i]]++;
                break;
            case 2:
                hist_R[image[i]]++;
                break;
        }
    }

    FILE* fp;
    fp = fopen(out_r, "w+");
    for(int i = 0; i < 256; i++)
        fprintf(fp,"%d\t%d\n", i,hist_R[i]);
    fclose(fp);

    fp = fopen(out_g, "w+");
    for(int i = 0; i < 256; i++)
        fprintf(fp,"%d\t%d\n", i,hist_G[i]);
    fclose(fp);

    fp = fopen(out_b, "w+");
    for(int i = 0; i < 256; i++)
        fprintf(fp,"%d\t%d\n", i,hist_B[i]);
    fclose(fp);

    return 0;
}
