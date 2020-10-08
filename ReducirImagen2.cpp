#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;


const int rows = 410;
const int cols = 410;
const int outRows = 409;
const int outCols = 409;

void reducirUnPixel(int imgR[rows][cols], int imgG[rows][cols], int imgB[rows][cols], 
                    int outR[outRows][outCols], int outG[outRows][outCols], int outB[outRows][outCols]){
    double R[rows][cols];
    double G[rows][cols];
    double B[rows][cols];

    for (int i=0; i < rows; i++){
        for ( int j=0; j < outCols; j++){
            R[i][j] = (double)((imgR[i][j] + imgR[i][j+1])/2);
            G[i][j] = (double)((imgG[i][j] + imgG[i][j+1])/2);
            B[i][j] = (double)((imgB[i][j] + imgB[i][j+1])/2);
        }
    }

    for (int i=0; i < outRows; i++){
        for (int j=0; j < outCols; j++){
            outR[i][j] = ceil((double)((R[i][j] + R[i+1][j])/2));
            outG[i][j] = ceil((double)((G[i][j] + G[i+1][j])/2));
            outB[i][j] = ceil((double)((B[i][j] + B[i+1][j])/2));
        }
    }
}

int main(int argc, char** argv )
{
    // Read image
    Mat img = imread("red_eyes.jpg",CV_LOAD_IMAGE_COLOR);

    // Output image
    // Mat imgOut = img.clone();
    Mat imgOut(outRows,outCols,CV_8UC3);

    
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

    
    int imgR[rows][cols];
    int imgG[rows][cols];
    int imgB[rows][cols];
    int outR[outRows][outCols];
    int outG[outRows][outCols];
    int outB[outRows][outCols];

    for (int i=0; i < rows; i++){
        for (int j=0; j < cols; j++){
            imgR[i][j] = img.at<cv::Vec3b>(i,j)[2];
            imgG[i][j] = img.at<cv::Vec3b>(i,j)[1];
            imgB[i][j] = img.at<cv::Vec3b>(i,j)[0];
        }
    }

    reducirUnPixel(imgR, imgG, imgB, outR, outG, outB);



    for (int i = 0; i < outRows; i++)
    {
        for (int j = 0; j < outCols; j++)
        {
            imgOut.at<cv::Vec3b>(i,j)[0] = outB[i][j];
            imgOut.at<cv::Vec3b>(i,j)[1] = outG[i][j];
            imgOut.at<cv::Vec3b>(i,j)[2] = outR[i][j];
        }
    } 

    imshow("Imagen Entrada", img);
    imshow("Imagen Salida", imgOut);
    waitKey(0);
    return 1;
}
