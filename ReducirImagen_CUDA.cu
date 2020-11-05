#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <omp.h>
#include <sys/time.h>
#include <fstream>
#include <string> 

using namespace std;
using namespace cv;

int BLOCKSPERGRID  = 32;
int NUMTHREADS = 128;

int **outR;
int **outG;
int **outB;
int **imgR;
int **imgG;
int **imgB;
int numeroColumnasImg = 0;

void matriz4x4Amatriz2x2(int imgR[4][4], int imgG[4][4], int imgB[4][4], int outR[2][2], int outG[2][2], int outB[2][2])
{ // recibo una matríz cuadrada con numero par de filas (nxn)
    /*cuadrantes
    --- ---
    |1 | 2|
    --- ---
    |3 | 4|
    --- ---*/
    int n = sizeof(*imgR) / sizeof(*imgR[0]);
    int centroMatriz = (n / 2);
    int cuadrante1R = 0, cuadrante2R = 0, cuadrante3R = 0, cuadrante4R = 0;
    int cuadrante1G = 0, cuadrante2G = 0, cuadrante3G = 0, cuadrante4G = 0;
    int cuadrante1B = 0, cuadrante2B = 0, cuadrante3B = 0, cuadrante4B = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < centroMatriz)
            {
                if (j < centroMatriz)
                {
                    //cuadrante 1
                    cuadrante1R += imgR[i][j];
                    cuadrante1G += imgG[i][j];
                    cuadrante1B += imgB[i][j];
                }
                else
                {
                    //cuadrante 2
                    cuadrante2R += imgR[i][j];
                    cuadrante2G += imgG[i][j];
                    cuadrante2B += imgB[i][j];
                }
            }
            else
            {
                if (j < centroMatriz)
                {
                    //cuadrante 3
                    cuadrante3R += imgR[i][j];
                    cuadrante3G += imgG[i][j];
                    cuadrante3B += imgB[i][j];
                }
                else
                {
                    //cuadrante 4
                    cuadrante4R += imgR[i][j];
                    cuadrante4G += imgG[i][j];
                    cuadrante4B += imgB[i][j];
                }
            }
        }
    }

    int divisor = centroMatriz * centroMatriz;

    outR[0][0] = cuadrante1R / divisor;
    outR[0][1] = cuadrante2R / divisor;
    outR[1][0] = cuadrante3R / divisor;
    outR[1][1] = cuadrante4R / divisor;

    outG[0][0] = cuadrante1G / divisor;
    outG[0][1] = cuadrante2G / divisor;
    outG[1][0] = cuadrante3G / divisor;
    outG[1][1] = cuadrante4G / divisor;

    outB[0][0] = cuadrante1B / divisor;
    outB[0][1] = cuadrante2B / divisor;
    outB[1][0] = cuadrante3B / divisor;
    outB[1][1] = cuadrante4B / divisor;
}

void algoritmo2Para4K(int imgR[8][8], int imgG[8][8], int imgB[8][8], int outR[2][2], int outG[2][2], int outB[2][2])
{ // recibo una matríz cuadrada con numero par de filas (nxn)
    /*cuadrantes
    --- ---
    |1 | 2|
    --- ---
    |3 | 4|
    --- ---*/
    int n = sizeof(*imgR) / sizeof(*imgR[0]);
    int centroMatriz = (n / 2);
    int cuadrante1R = 0, cuadrante2R = 0, cuadrante3R = 0, cuadrante4R = 0;
    int cuadrante1G = 0, cuadrante2G = 0, cuadrante3G = 0, cuadrante4G = 0;
    int cuadrante1B = 0, cuadrante2B = 0, cuadrante3B = 0, cuadrante4B = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < centroMatriz)
            {
                if (j < centroMatriz)
                {
                    //cuadrante 1
                    cuadrante1R += imgR[i][j];
                    cuadrante1G += imgG[i][j];
                    cuadrante1B += imgB[i][j];
                }
                else
                {
                    //cuadrante 2
                    cuadrante2R += imgR[i][j];
                    cuadrante2G += imgG[i][j];
                    cuadrante2B += imgB[i][j];
                }
            }
            else
            {
                if (j < centroMatriz)
                {
                    //cuadrante 3
                    cuadrante3R += imgR[i][j];
                    cuadrante3G += imgG[i][j];
                    cuadrante3B += imgB[i][j];
                }
                else
                {
                    //cuadrante 4
                    cuadrante4R += imgR[i][j];
                    cuadrante4G += imgG[i][j];
                    cuadrante4B += imgB[i][j];
                }
            }
        }
    }

    int divisor = centroMatriz * centroMatriz;

    outR[0][0] = cuadrante1R / divisor;
    outR[0][1] = cuadrante2R / divisor;
    outR[1][0] = cuadrante3R / divisor;
    outR[1][1] = cuadrante4R / divisor;

    outG[0][0] = cuadrante1G / divisor;
    outG[0][1] = cuadrante2G / divisor;
    outG[1][0] = cuadrante3G / divisor;
    outG[1][1] = cuadrante4G / divisor;

    outB[0][0] = cuadrante1B / divisor;
    outB[0][1] = cuadrante2B / divisor;
    outB[1][0] = cuadrante3B / divisor;
    outB[1][1] = cuadrante4B / divisor;
}

void algoritmo2Para1080p(int imgR[8][8], int imgG[8][8], int imgB[8][8], int outR[4][4], int outG[4][4], int outB[4][4])
{ // recibo matriz de 8x8 y tamaño de matriz resultante (nxn)
    int subMatriz1R[4][4];
    int subMatriz2R[4][4];
    int subMatriz3R[4][4];
    int subMatriz4R[4][4];

    int subMatriz1G[4][4];
    int subMatriz2G[4][4];
    int subMatriz3G[4][4];
    int subMatriz4G[4][4];

    int subMatriz1B[4][4];
    int subMatriz2B[4][4];
    int subMatriz3B[4][4];
    int subMatriz4B[4][4];
    int centroMatriz = 4;

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            if (i < centroMatriz)
            {
                if (j < centroMatriz)
                {
                    //cuadrante 1
                    subMatriz1R[i][j] = imgR[i][j];
                    subMatriz1G[i][j] = imgG[i][j];
                    subMatriz1B[i][j] = imgB[i][j];
                }
                else
                {
                    //cuadrante 2
                    subMatriz2R[i][j - 4] = imgR[i][j];
                    subMatriz2G[i][j - 4] = imgG[i][j];
                    subMatriz2B[i][j - 4] = imgB[i][j];
                }
            }
            else
            {
                if (j < centroMatriz)
                {
                    //cuadrante 3
                    subMatriz3R[i - 4][j] = imgR[i][j];
                    subMatriz3G[i - 4][j] = imgG[i][j];
                    subMatriz3B[i - 4][j] = imgB[i][j];
                }
                else
                {
                    //cuadrante 4
                    subMatriz4R[i - 4][j - 4] = imgR[i][j];
                    subMatriz4G[i - 4][j - 4] = imgG[i][j];
                    subMatriz4B[i - 4][j - 4] = imgB[i][j];
                }
            }
        }
    }

    int matrizCuadrante1R[2][2];
    int matrizCuadrante2R[2][2];
    int matrizCuadrante3R[2][2];
    int matrizCuadrante4R[2][2];

    int matrizCuadrante1G[2][2];
    int matrizCuadrante2G[2][2];
    int matrizCuadrante3G[2][2];
    int matrizCuadrante4G[2][2];

    int matrizCuadrante1B[2][2];
    int matrizCuadrante2B[2][2];
    int matrizCuadrante3B[2][2];
    int matrizCuadrante4B[2][2];

    matriz4x4Amatriz2x2(subMatriz1R, subMatriz1G, subMatriz1B, matrizCuadrante1R, matrizCuadrante1G, matrizCuadrante1B);
    matriz4x4Amatriz2x2(subMatriz2R, subMatriz2G, subMatriz2B, matrizCuadrante2R, matrizCuadrante2G, matrizCuadrante2B);
    matriz4x4Amatriz2x2(subMatriz3R, subMatriz3G, subMatriz3B, matrizCuadrante3R, matrizCuadrante3G, matrizCuadrante3B);
    matriz4x4Amatriz2x2(subMatriz4R, subMatriz4G, subMatriz4B, matrizCuadrante4R, matrizCuadrante4G, matrizCuadrante4B);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            outR[i][j] = matrizCuadrante1R[i][j];
            outR[i][j + 2] = matrizCuadrante2R[i][j];
            outR[i + 2][j] = matrizCuadrante3R[i][j];
            outR[i + 2][j + 2] = matrizCuadrante4R[i][j];

            outG[i][j] = matrizCuadrante1G[i][j];
            outG[i][j + 2] = matrizCuadrante2G[i][j];
            outG[i + 2][j] = matrizCuadrante3G[i][j];
            outG[i + 2][j + 2] = matrizCuadrante4G[i][j];

            outB[i][j] = matrizCuadrante1B[i][j];
            outB[i][j + 2] = matrizCuadrante2B[i][j];
            outB[i + 2][j] = matrizCuadrante3B[i][j];
            outB[i + 2][j + 2] = matrizCuadrante4B[i][j];
        }
    }
}

Mat cambiarTamanoImagen(Mat img, int row, int nuevoNcolumnas)
{
    Mat imgAux(row, nuevoNcolumnas, CV_8UC3);

    for (int i = 0; i < imgAux.rows; i++)
    {
        for (int j = 0; j < imgAux.cols; j++)
        {
            imgAux.at<cv::Vec3b>(i, j)[0] = img.at<cv::Vec3b>(i, j)[0];
            imgAux.at<cv::Vec3b>(i, j)[1] = img.at<cv::Vec3b>(i, j)[1];
            imgAux.at<cv::Vec3b>(i, j)[2] = img.at<cv::Vec3b>(i, j)[2];
        }
    }
    return imgAux;
}

__global__ void reducirMatriz3x3a2x2(int imgR[3][3], int imgG[3][3], int imgB[3][3], int outR[2][2], int outG[2][2], int outB[2][2])
{
    double R[3][2];
    double G[3][2];
    double B[3][2];

    for (int k = 0; k < 3; k++)
    {
        for (int l = 0; l < 2; l++)
        {
            R[k][l] = (double)((imgR[k][l] + imgR[k][l + 1]) / 2);
            G[k][l] = (double)((imgG[k][l] + imgG[k][l + 1]) / 2);
            B[k][l] = (double)((imgB[k][l] + imgB[k][l + 1]) / 2);
        }
    }

    for (int k = 0; k < 2; k++)
    {
        for (int l = 0; l < 2; l++)
        {
            outR[k][l] = ceil((double)((R[k][l] + R[k + 1][l]) / 2));
            outG[k][l] = ceil((double)((G[k][l] + G[k + 1][l]) / 2));
            outB[k][l] = ceil((double)((B[k][l] + B[k + 1][l]) / 2));
        }
    }
}

void reducirMatriz9x9a4x4(int imgR[9][9], int imgG[9][9], int imgB[9][9], int outR[4][4], int outG[4][4], int outB[4][4])
{
    double R[9][9];
    double G[9][9];
    double B[9][9];

    int R8x8[8][8];
    int G8x8[8][8];
    int B8x8[8][8];

    for (int k = 0; k < 9; k++)
    {
        for (int l = 0; l < 8; l++)
        {
            R[k][l] = (double)((imgR[k][l] + imgR[k][l + 1]) / 2);
            G[k][l] = (double)((imgG[k][l] + imgG[k][l + 1]) / 2);
            B[k][l] = (double)((imgB[k][l] + imgB[k][l + 1]) / 2);
        }
    }

    for (int k = 0; k < 8; k++)
    {
        for (int l = 0; l < 8; l++)
        {
            R8x8[k][l] = ceil((double)((R[k][l] + R[k + 1][l]) / 2));
            G8x8[k][l] = ceil((double)((G[k][l] + G[k + 1][l]) / 2));
            B8x8[k][l] = ceil((double)((B[k][l] + B[k + 1][l]) / 2));
        }
    }

    algoritmo2Para1080p(R8x8, G8x8, B8x8, outR, outG, outB);
}

void reducirMatriz9x9a2x2(int imgR[9][9], int imgG[9][9], int imgB[9][9], int outR[2][2], int outG[2][2], int outB[2][2])
{
    double R[9][9];
    double G[9][9];
    double B[9][9];

    int R8x8[8][8];
    int G8x8[8][8];
    int B8x8[8][8];

    for (int k = 0; k < 9; k++)
    {
        for (int l = 0; l < 8; l++)
        {
            R[k][l] = (double)((imgR[k][l] + imgR[k][l + 1]) / 2);
            G[k][l] = (double)((imgG[k][l] + imgG[k][l + 1]) / 2);
            B[k][l] = (double)((imgB[k][l] + imgB[k][l + 1]) / 2);
        }
    }

    for (int k = 0; k < 8; k++)
    {
        for (int l = 0; l < 8; l++)
        {
            R8x8[k][l] = ceil((double)((R[k][l] + R[k + 1][l]) / 2));
            G8x8[k][l] = ceil((double)((G[k][l] + G[k + 1][l]) / 2));
            B8x8[k][l] = ceil((double)((B[k][l] + B[k + 1][l]) / 2));
        }
    }

    algoritmo2Para4K(R8x8, G8x8, B8x8, outR, outG, outB);
}

__global__ void reduccion720(int **imgR, int **imgG, int **imgB, int **outR, int **outG, int **outB, int *numeroColumnasImg, int *NUMTHREADS)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if (*NUMTHREADS<=240){
        int filaInicial, filaFinal; //, threadId = *(int *)args;

        int numeroFilasImg = 240; // 720/3
        filaInicial = (numeroFilasImg / *NUMTHREADS) * threadId;
        filaFinal = filaInicial + ((numeroFilasImg / *NUMTHREADS) - 1);

        for (int i = filaInicial; i <= filaFinal; i++)
        {
            for (int j = 0; j < *numeroColumnasImg; j++)
            {
                int R3x3[3][3];
                int G3x3[3][3];
                int B3x3[3][3];

                int indexFilaActual = (i * 3);
                int indexColumnaActual = (j * 3);

                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        R3x3[k][l] = imgR[indexFilaActual + k][indexColumnaActual + l];
                        G3x3[k][l] = imgG[indexFilaActual + k][indexColumnaActual + l];
                        B3x3[k][l] = imgB[indexFilaActual + k][indexColumnaActual + l];
                    }
                }

                int R2x2[2][2];
                int G2x2[2][2];
                int B2x2[2][2];

                reducirMatriz3x3a2x2(R3x3, G3x3, B3x3, R2x2, G2x2, B2x2);

                int indexFilaActualOUT = (i * 2);
                int indexColumnaActualOUT = (j * 2);

                for (int k = 0; k < 2; k++)
                {
                    for (int l = 0; l < 2; l++)
                    {
                        outR[indexFilaActualOUT + k][indexColumnaActualOUT + l] = R2x2[k][l];
                        outG[indexFilaActualOUT + k][indexColumnaActualOUT + l] = G2x2[k][l];
                        outB[indexFilaActualOUT + k][indexColumnaActualOUT + l] = B2x2[k][l];
                    }
                }
            }
        }
    }else{

    }
}

void *reduccion1080(void *args)
{
    int filaInicial, filaFinal, threadId = *(int *)args;

    int numeroFilasImg = 120; // 1080/9

    filaInicial = (numeroFilasImg / NUMTHREADS) * threadId;
    filaFinal = filaInicial + ((numeroFilasImg / NUMTHREADS) - 1);

    for (int i = filaInicial; i <= filaFinal; i++)
    {
        for (int j = 0; j < numeroColumnasImg; j++)
        {
            int R9x9[9][9];
            int G9x9[9][9];
            int B9x9[9][9];

            int indexFilaActual = (i * 9);
            int indexColumnaActual = (j * 9);

            for (int k = 0; k < 9; k++)
            {
                for (int l = 0; l < 9; l++)
                {
                    R9x9[k][l] = imgR[indexFilaActual + k][indexColumnaActual + l];
                    G9x9[k][l] = imgG[indexFilaActual + k][indexColumnaActual + l];
                    B9x9[k][l] = imgB[indexFilaActual + k][indexColumnaActual + l];
                }
            }

            int R4x4[4][4];
            int G4x4[4][4];
            int B4x4[4][4];

            reducirMatriz9x9a4x4(R9x9, G9x9, B9x9, R4x4, G4x4, B4x4);

            int indexFilaActualOUT = (i * 4);
            int indexColumnaActualOUT = (j * 4);

            for (int k = 0; k < 4; k++)
            {
                for (int l = 0; l < 4; l++)
                {
                    outR[indexFilaActualOUT + k][indexColumnaActualOUT + l] = R4x4[k][l];
                    outG[indexFilaActualOUT + k][indexColumnaActualOUT + l] = G4x4[k][l];
                    outB[indexFilaActualOUT + k][indexColumnaActualOUT + l] = B4x4[k][l];
                }
            }
        }
    }
}

void *reduccion4k(void *args)
{
    int filaInicial, filaFinal, threadId = *(int *)args;

    int numeroFilasImg = 240; // 2160/9
    filaInicial = (numeroFilasImg / NUMTHREADS) * threadId;
    filaFinal = filaInicial + ((numeroFilasImg / NUMTHREADS) - 1);

    for (int i = filaInicial; i <= filaFinal; i++)
    {
        for (int j = 0; j < numeroColumnasImg; j++)
        {
            int R9x9[9][9];
            int G9x9[9][9];
            int B9x9[9][9];

            int indexFilaActual = (i * 9);
            int indexColumnaActual = (j * 9);

            for (int k = 0; k < 9; k++)
            {
                for (int l = 0; l < 9; l++)
                {
                    R9x9[k][l] = imgR[indexFilaActual + k][indexColumnaActual + l];
                    G9x9[k][l] = imgG[indexFilaActual + k][indexColumnaActual + l];
                    B9x9[k][l] = imgB[indexFilaActual + k][indexColumnaActual + l];
                }
            }

            int R2x2[2][2];
            int G2x2[2][2];
            int B2x2[2][2];

            reducirMatriz9x9a2x2(R9x9, G9x9, B9x9, R2x2, G2x2, B2x2);

            int indexFilaActualOUT = (i * 2);
            int indexColumnaActualOUT = (j * 2);

            for (int k = 0; k < 2; k++)
            {
                for (int l = 0; l < 2; l++)
                {
                    outR[indexFilaActualOUT + k][indexColumnaActualOUT + l] = R2x2[k][l];
                    outG[indexFilaActualOUT + k][indexColumnaActualOUT + l] = G2x2[k][l];
                    outB[indexFilaActualOUT + k][indexColumnaActualOUT + l] = B2x2[k][l];
                }
            }
        }
    }
}

int main(int argc, char **argv)
{    
    /* char* nombreEntrada = argv[1];
    char* nombreSalida = argv[2];
    NUMTHREADS = atoi(argv[3]); */

    string nombreEntrada = "imagen720p.jpg";
    string nombreSalida = "imagen720-a480CUDAAAAAAA.jpg";
    NUMTHREADS=2;

    //ofstream file;

    // Leer imágen
    Mat img = imread(nombreEntrada, CV_LOAD_IMAGE_COLOR);

    //cout << "La imagen tiene " << img.rows << " pixeles de alto x "<< img.cols << " pixeles de ancho" << endl;

    // Inicio Correción tamaño (si necesita)
    int nuevoNcolumnas = 0;

    if (img.rows == 720)
    {
        if (img.cols % 3 != 0)
        {
            nuevoNcolumnas = (img.cols / 3) * 3;
            img = cambiarTamanoImagen(img, img.rows, nuevoNcolumnas);
        }
    }
    else if (img.rows == 1080)
    {
        if (img.cols % 9 != 0)
        {
            nuevoNcolumnas = (img.cols / 9) * 9;
            img = cambiarTamanoImagen(img, img.rows, nuevoNcolumnas);
        }
    }
    else if (img.rows == 2160)
    {
        if (img.cols % 9 != 0)
        {
            nuevoNcolumnas = (img.cols / 9) * 9;
            img = cambiarTamanoImagen(img, img.rows, nuevoNcolumnas);
        }
    }
    else
    {
        cout << "Resolución no permitida" << endl;
    }
    // Fin Correción tamaño (si necesita)

    // cout << "La imagen tiene " << img.rows << " pixeles de alto x " << img.cols << " pixeles de ancho" << endl;

    // Comienzo creación de matrices
    const int rows = img.rows;
    const int cols = img.cols;

    imgR = new int *[rows];
    for (size_t i = 0; i < rows; ++i)
        imgR[i] = new int[cols];

    imgG = new int *[rows];
    for (size_t i = 0; i < rows; ++i)
        imgG[i] = new int[cols];

    imgB = new int *[rows];
    for (size_t i = 0; i < rows; ++i)
        imgB[i] = new int[cols];

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            imgR[i][j] = img.at<cv::Vec3b>(i, j)[2];
            imgG[i][j] = img.at<cv::Vec3b>(i, j)[1];
            imgB[i][j] = img.at<cv::Vec3b>(i, j)[0];
        }
    }

    const int outRows = 480;
    const int outCols = (outRows * img.cols) / img.rows;

    // Output image
    Mat imgOut(outRows, outCols, CV_8UC3);

    outR = new int *[outRows];
    for (size_t i = 0; i < outRows; ++i)
        outR[i] = new int[outCols];

    outG = new int *[outRows];
    for (size_t i = 0; i < outRows; ++i)
        outG[i] = new int[outCols];

    outB = new int *[outRows];
    for (size_t i = 0; i < outRows; ++i)
        outB[i] = new int[outCols];

    // Fin creación de matrices

    //************************** CUDA **********************************

    int **d_imgR;
    int **d_imgG;
    int **d_imgB;
    int **d_outR;
    int **d_outG;
    int **d_outB;
    int *d_numeroColumnasImg;
    int *d_NUMTHREADS;
    

    int sizeIn = sizeof(imgR); // Size sirve para todas las img
    int sizeOut = sizeof(outR); // Size sirve para todas las out
    int sizeNumeroColumnasImg = sizeof(numeroColumnasImg); 
    int sizeNUMTHREADS = sizeof(NUMTHREADS); 

    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_imgR, sizeIn);
    cudaMalloc((void **)&d_imgG, sizeIn);
    cudaMalloc((void **)&d_imgB, sizeIn);

    cudaMalloc((void **)&d_outR, sizeOut);
    cudaMalloc((void **)&d_outG, sizeOut);
    cudaMalloc((void **)&d_outB, sizeOut);

    cudaMalloc((void **)&d_numeroColumnasImg, sizeNumeroColumnasImg);
    cudaMalloc((void **)&d_NUMTHREADS, sizeNUMTHREADS);


    numeroColumnasImg = cols / 3;


    // Copy inputs to device
    cudaMemcpy(d_imgR, imgR, sizeIn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgG, imgG, sizeIn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgB, imgB, sizeIn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_numeroColumnasImg, numeroColumnasImg, sizeNumeroColumnasImg, cudaMemcpyHostToDevice);
    cudaMemcpy(d_NUMTHREADS, NUMTHREADS, sizeNUMTHREADS, cudaMemcpyHostToDevice);

    int NUMTHREADSPerBlock = NUMTHREADS/BLOCKSPERGRID;
    // Launch add() kernel on GPU with N blocks
    reduccion720<<<BLOCKSPERGRID, NUMTHREADSPerBlock>>>(d_imgR, d_imgG, d_imgB, d_outR, d_outG, d_outB, d_numeroColumnasImg, d_NUMTHREADS);

    // Copy result back to host
    cudaMemcpy(outR, d_outR, sizeOut, cudaMemcpyDeviceToHost);
    cudaMemcpy(outG, d_outG, sizeOut, cudaMemcpyDeviceToHost);
    cudaMemcpy(outB, d_outB, sizeOut, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_imgR); cudaFree(d_imgG); cudaFree(d_imgB);cudaFree(d_outR); cudaFree(d_outG); cudaFree(d_outB); cudaFree(d_numeroColumnasImg); cudaFree(d_NUMTHREADS);


    //************************** CUDA **********************************


    /*

    //Inicio Conversión**********************************
    int numeroFilasImg = 0;

    //  Creación hilos y empezar toma de tiempo
    int threadId[NUMTHREADS], i, *retval;
    pthread_t thread[NUMTHREADS];

    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    //  **************

    if (rows == 720)
    {
        numeroColumnasImg = cols / 3;

        for (i = 0; i < NUMTHREADS; i++)
        {
            threadId[i] = i;
            pthread_create(&thread[i], NULL, reduccion720, &threadId[i]);
        }
    }
    else if (rows == 1080)
    {

        numeroColumnasImg = cols / 9;

        for (i = 0; i < NUMTHREADS; i++)
        {
            threadId[i] = i;
            pthread_create(&thread[i], NULL, reduccion1080, &threadId[i]);
        }
    }
    else if (rows == 2160)
    {

        numeroColumnasImg = cols / 9;

        for (i = 0; i < NUMTHREADS; i++)
        {
            threadId[i] = i;
            pthread_create(&thread[i], NULL, reduccion4k, &threadId[i]);
        }
    }
    else
    {
        cout << "Resolución no permitida" << endl;
    }
    //Fin Conversión*******************

    //Recolección Hilos y finalización toma de tiempo
    for (i = 0; i < NUMTHREADS; i++)
    {
        pthread_join(thread[i], (void **)&retval);
    }

    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);

    

    if (rows == 720){
        ofstream file;
        file.open("./720.txt", ofstream::app);
        file << NUMTHREADS << " HILOS: " << (long double)tval_result.tv_sec + (long double)(tval_result.tv_usec)/1000000 << endl;
        file.close();
    }else if (rows == 1080){
        ofstream file;
        file.open("./1080.txt", ofstream::app);
        file << NUMTHREADS << " HILOS: " << (long double)tval_result.tv_sec + (long double)(tval_result.tv_usec)/1000000 << endl;
        file.close();
    }else{
        ofstream file;
        file.open("./4k.txt", ofstream::app);
        file << NUMTHREADS << " HILOS: " << (long double)tval_result.tv_sec + (long double)(tval_result.tv_usec)/1000000 << endl;
        file.close();
    }
    //Fin Recolección Hilos y finalización toma de tiempo*****
    */

    //Pasar matrices resultantes a Imagen de salida
    for (int i = 0; i < outRows; i++)
    {
        for (int j = 0; j < outCols; j++)
        {
            imgOut.at<cv::Vec3b>(i, j)[0] = outB[i][j];
            imgOut.at<cv::Vec3b>(i, j)[1] = outG[i][j];
            imgOut.at<cv::Vec3b>(i, j)[2] = outR[i][j];
        }
    }
    //Fin Pasar matrices resultantes a Imagen de salida

    //Inicio borrar matrices
    for (size_t i = 0; i < rows; ++i)
        delete imgR[i];
    delete imgR;

    for (size_t i = 0; i < rows; ++i)
        delete imgG[i];
    delete imgG;

    for (size_t i = 0; i < rows; ++i)
        delete imgB[i];
    delete imgB;

    for (size_t i = 0; i < outRows; ++i)
        delete outR[i];
    delete outR;

    for (size_t i = 0; i < outRows; ++i)
        delete outG[i];
    delete outG;

    for (size_t i = 0; i < outRows; ++i)
        delete outB[i];
    delete outB;

    //Fin Borrar matrices

    // Imprimir Imagen original y convertida. DESCOMENTAR LAS SIGUIENTES LINEAS SI SE QUIEREN VER LAS IMAGENES DE ENTRADA Y SALIDA RESPECTIVAMENTE
    //imshow(nombreEntrada, img);
    //imshow(nombreSalida, imgOut);

    //Guarda imágen de salida en directorio local
    imwrite(nombreSalida, imgOut);
    waitKey(0);

    return 1;
}
