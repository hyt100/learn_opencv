#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char *argv[])
{
    cv::Mat img_bgr = cv::imread("../data/lena.jpg");
    cv::Mat img_gray;

    #if 0
    cv::cvtColor(img_bgr, img_gray, cv::COLOR_BGR2GRAY);
    #else
    img_gray = cv::imread("../data/lena.jpg", cv::IMREAD_GRAYSCALE);
    #endif
    
    cv::imwrite("lena_gray.jpg", img_gray);

    cv::imshow("bgr", img_bgr);
    cv::imshow("gray", img_gray);
    cv::waitKey(0);
    return 0;
}