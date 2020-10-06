#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

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

//CAMBIAR POR COMPLETO LA SIGUIENTE FUNCIÓN************************************************
//CAMBIAR POR COMPLETO LA SIGUIENTE FUNCIÓN************************************************
//CAMBIAR POR COMPLETO LA SIGUIENTE FUNCIÓN************************************************
void reducirMatriz9x9a4x4(int imgR[9][9], int imgG[9][9], int imgB[9][9], int outR[4][4], int outG[4][4], int outB[4][4])
{
    for (int k = 0; k < 4; k++)
    {
        for (int l = 0; l < 4; l++)
        {
            outR[k][l] = imgR[k][l];
            outG[k][l] = imgG[k][l];
            outB[k][l] = imgB[k][l];
        }
    }
}

//CAMBIAR POR COMPLETO LA SIGUIENTE FUNCIÓN************************************************
//CAMBIAR POR COMPLETO LA SIGUIENTE FUNCIÓN************************************************
//CAMBIAR POR COMPLETO LA SIGUIENTE FUNCIÓN************************************************
void reducirMatriz9x9a2x2(int imgR[9][9], int imgG[9][9], int imgB[9][9], int outR[2][2], int outG[2][2], int outB[2][2])
{
    for (int k = 0; k < 2; k++)
    {
        for (int l = 0; l < 2; l++)
        {
            outR[k][l] = imgR[k][l];
            outG[k][l] = imgG[k][l];
            outB[k][l] = imgB[k][l];
        }
    }
}

int main(int argc, char **argv)
{
    // Read image
    Mat img = imread("imagen4k.jpg", CV_LOAD_IMAGE_COLOR);

    cout << "La imagen tiene " << img.rows << " pixeles de alto x "
         << img.cols << " pixeles de ancho" << endl;

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

    cout << "La imagen tiene " << img.rows << " pixeles de alto x " << img.cols << " pixeles de ancho" << endl;

    //Comienzo creación de matrices
    const int rows = img.rows;
    const int cols = img.cols;

    int **imgR = new int *[rows];
    for (size_t i = 0; i < rows; ++i)
        imgR[i] = new int[cols];

    int **imgG = new int *[rows];
    for (size_t i = 0; i < rows; ++i)
        imgG[i] = new int[cols];

    int **imgB = new int *[rows];
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

    int **outR = new int *[outRows];
    for (size_t i = 0; i < outRows; ++i)
        outR[i] = new int[outCols];

    int **outG = new int *[outRows];
    for (size_t i = 0; i < outRows; ++i)
        outG[i] = new int[outCols];

    int **outB = new int *[outRows];
    for (size_t i = 0; i < outRows; ++i)
        outB[i] = new int[outCols];

    //Fin creación de matrices

    //Inicio Conversión**********************************
    int numeroFilasImg = 0;
    int numeroColumnasImg = 0;
    if (rows == 720)
    {
        //reducirUnPixel(imgR, imgG, imgB, outR, outG, outB);
    }
    else if (rows == 1080)
    {
        numeroFilasImg = 120; // 1080/9
        numeroColumnasImg = cols / 9;

        for (int i = 0; i < numeroFilasImg; i++)
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
    else if (rows == 2160)
    {
        numeroFilasImg = 240; // 2160/9
        numeroColumnasImg = cols / 9;

        for (int i = 0; i < numeroFilasImg; i++)
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
    else
    {
        cout << "Resolución no permitida" << endl;
    }
    //Fin Conversión*******************

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
    imshow("Imagen Entrada", img);
    imshow("Imagen Salida", imgOut);
    waitKey(0);

    return 1;
}
