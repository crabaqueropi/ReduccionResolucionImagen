#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void matriz4x4Amatriz2x2(int matriz[4][4], int matrizResultante[2][2])
{ // recibo una matríz cuadrada con numero par de filas (nxn)
    /*cuadrantes
    --- ---
    |1 | 2|
    --- ---
    |3 | 4|
    --- ---*/
    int n = sizeof(*matriz) / sizeof(*matriz[0]);
    int centroMatriz = (n / 2);
    int cuadrante1 = 0, cuadrante2 = 0, cuadrante3 = 0, cuadrante4 = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < centroMatriz)
            {
                if (j < centroMatriz)
                {
                    //cuadrante 1
                    cuadrante1 += matriz[i][j];
                }
                else
                {
                    //cuadrante 2
                    cuadrante2 += matriz[i][j];
                }
            }
            else
            {
                if (j < centroMatriz)
                {
                    //cuadrante 3
                    cuadrante3 += matriz[i][j];
                }
                else
                {
                    //cuadrante 4
                    cuadrante4 += matriz[i][j];
                }
            }
        }
    }

    int divisor = centroMatriz * centroMatriz;

    matrizResultante[0][0] = cuadrante1 / divisor;
    matrizResultante[0][1] = cuadrante2 / divisor;
    matrizResultante[1][0] = cuadrante3 / divisor;
    matrizResultante[1][1] = cuadrante4 / divisor;
}

void algoritmo2Para4K(int matriz[8][8], int matrizResultante[2][2])
{ // recibo una matríz cuadrada con numero par de filas (nxn)
    /*cuadrantes
    --- ---
    |1 | 2|
    --- ---
    |3 | 4|
    --- ---*/
    int n = sizeof(*matriz) / sizeof(*matriz[0]);
    int centroMatriz = (n / 2);
    int cuadrante1 = 0, cuadrante2 = 0, cuadrante3 = 0, cuadrante4 = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < centroMatriz)
            {
                if (j < centroMatriz)
                {
                    //cuadrante 1
                    cuadrante1 += matriz[i][j];
                }
                else
                {
                    //cuadrante 2
                    cuadrante2 += matriz[i][j];
                }
            }
            else
            {
                if (j < centroMatriz)
                {
                    //cuadrante 3
                    cuadrante3 += matriz[i][j];
                }
                else
                {
                    //cuadrante 4
                    cuadrante4 += matriz[i][j];
                }
            }
        }
    }

    int divisor = centroMatriz * centroMatriz;

    matrizResultante[0][0] = cuadrante1 / divisor;
    matrizResultante[0][1] = cuadrante2 / divisor;
    matrizResultante[1][0] = cuadrante3 / divisor;
    matrizResultante[1][1] = cuadrante4 / divisor;
}

void algoritmo2Para1080p(int matriz[8][8], int matrizResultante[4][4])
{ // recibo matriz de 8x8 y tamaño de matriz resultante (nxn)
    int subMatriz1[4][4];
    int subMatriz2[4][4];
    int subMatriz3[4][4];
    int subMatriz4[4][4];
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
                    subMatriz1[i][j] = matriz[i][j];
                }
                else
                {
                    //cuadrante 2
                    subMatriz2[i][j - 4] = matriz[i][j];
                }
            }
            else
            {
                if (j < centroMatriz)
                {
                    //cuadrante 3
                    subMatriz3[i - 4][j] = matriz[i][j];
                }
                else
                {
                    //cuadrante 4
                    subMatriz4[i - 4][j - 4] = matriz[i][j];
                }
            }
        }
    }

    int matrizCuadrante1[2][2];
    int matrizCuadrante2[2][2];
    int matrizCuadrante3[2][2];
    int matrizCuadrante4[2][2];

    matriz4x4Amatriz2x2(subMatriz1, matrizCuadrante1);
    matriz4x4Amatriz2x2(subMatriz2, matrizCuadrante2);
    matriz4x4Amatriz2x2(subMatriz3, matrizCuadrante3);
    matriz4x4Amatriz2x2(subMatriz4, matrizCuadrante4);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            matrizResultante[i][j] = matrizCuadrante1[i][j];
            matrizResultante[i][j + 2] = matrizCuadrante2[i][j];
            matrizResultante[i + 2][j] = matrizCuadrante3[i][j];
            matrizResultante[i + 2][j + 2] = matrizCuadrante4[i][j];
        }
    }
}

int main(int argc, char **argv)
{
    // Read image
    Mat img = imread("red_eyes.jpg", CV_LOAD_IMAGE_COLOR);

    // Output image
    // Mat imgOut = img.clone();
    Mat imgOut(480, 480, CV_8UC3);

    cout << "La imagen tiene " << img.rows << " pixeles de alto x "
         << img.cols << " pixeles de ancho" << endl;

    /*for (int i = 0; i < img.cols; i++)
    {
        img.at<cv::Vec3b>(511,i)[0] = 0; //B
        img.at<cv::Vec3b>(511,i)[1] = 0;  //G
        img.at<cv::Vec3b>(511,i)[2] = 255; //R

        img.at<cv::Vec3b>(512,i)[0] = 0;
        img.at<cv::Vec3b>(512,i)[1] = 0;
        img.at<cv::Vec3b>(512,i)[2] = 255;

        img.at<cv::Vec3b>(513,i)[0] = 0;
        img.at<cv::Vec3b>(513,i)[1] = 0;
        img.at<cv::Vec3b>(513,i)[2] = 255;
    }*/

    /*for (int i = 0; i < imgOut.cols; i++)
    {
        imgOut.at<cv::Vec3b>(240,i)[0] = 0; //B
        imgOut.at<cv::Vec3b>(240,i)[1] = 0;  //G
        imgOut.at<cv::Vec3b>(240,i)[2] = 255; //R

        imgOut.at<cv::Vec3b>(241,i)[0] = 0;
        imgOut.at<cv::Vec3b>(241,i)[1] = 0;
        imgOut.at<cv::Vec3b>(241,i)[2] = 255;

        imgOut.at<cv::Vec3b>(242,i)[0] = 0;
        imgOut.at<cv::Vec3b>(242,i)[1] = 0;
        imgOut.at<cv::Vec3b>(242,i)[2] = 255;
    }*/

    
    for (int i = 0; i < imgOut.rows; i++)
    {
        for (int j = 0; j < imgOut.cols; j++)
        {
            imgOut.at<cv::Vec3b>(i, j)[0] = img.at<cv::Vec3b>(i, j)[0];
            imgOut.at<cv::Vec3b>(i, j)[1] = img.at<cv::Vec3b>(i, j)[1];
            imgOut.at<cv::Vec3b>(i, j)[2] = img.at<cv::Vec3b>(i, j)[2];
        }
    }

    imshow("Imagen Entrada", img);
    imshow("Imagen Salida", imgOut);
    waitKey(0);
    
    return 1;
}
