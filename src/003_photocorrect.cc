#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

void photo_correct(cv::Mat &imgIn, cv::Mat &imgOut, std::vector<cv::Point2f> &src, float fx, float fy, float cx, float cy)
{
    //TODO
}

int main(int argc, char *argv[])
{
    cv::Mat imgIn = cv::imread("../data/photo_correct_640x480.jpg");

    // 角点坐标
    // The point order is:
    //     1 ---> 2
    //     |      |
    //     4 <--- 3
    cv::Point2f p1(184, 55);
    cv::Point2f p2(471, 38);
    cv::Point2f p3(520, 449);
    cv::Point2f p4(132, 452);
    std::vector<cv::Point2f> points;
    points.push_back(p1);
    points.push_back(p2);
    points.push_back(p3);
    points.push_back(p4);

    // camera instricts 相机内参
    float fx = 533.18;
    float fy = 533.18;
    float cx = 342.22;
    float cy = 235.01;

    cv::Mat imgOut;
    photo_correct(imgIn, imgOut, points, fx, fy, cx, cy);

    cv::imshow("imgIn", imgIn);
    cv::imshow("imgOut", imgOut);
    cv::waitKey(0);
    return 0;
}