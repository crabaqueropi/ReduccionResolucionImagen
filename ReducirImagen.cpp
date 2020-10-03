#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv )
{
    // Read image
    Mat img = imread("red_eyes.jpg",CV_LOAD_IMAGE_COLOR);

    // Output image
    // Mat imgOut = img.clone();
    Mat imgOut(480,480,CV_8UC3);

    
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
            imgOut.at<cv::Vec3b>(i,j)[0] = img.at<cv::Vec3b>(i,j)[0];
            imgOut.at<cv::Vec3b>(i,j)[1] = img.at<cv::Vec3b>(i,j)[1];
            imgOut.at<cv::Vec3b>(i,j)[2] = img.at<cv::Vec3b>(i,j)[2];
        }
    }

    imshow("Imagen Entrada", img);
    imshow("Imagen Salida", imgOut);
    waitKey(0);
    return 1;
}
