#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

int main()
{
    cv::Mat m = cv::imread("../data/persp_504x378.jpg");
    int w = m.cols, h = m.rows;
    cv::imshow("img", m);

    // 方法一： 通过仿射变换
    cv::Mat m1;
    // 绕原点顺时针旋转90度
    cv::Point2f center = {(float)0, (float)0}; 
    cv::Mat rotate = cv::getRotationMatrix2D(center, -90, 1);
    // 增加一个位移到矩阵上
    rotate.at<double>(0, 2) += h;
    cv::warpAffine(m, m1, rotate, cv::Size{h, w});
    cv::imshow("case1", m1);

    // 方法二： 通过转置矩阵+Y轴翻转
    cv::Mat m2, m22;
    cv::transpose(m, m2);
    cv::flip(m2, m22, 1);
    cv::imshow("case2", m22);

    // 方法三： 旋转接口(支持90/180/270)
    cv::Mat m3;
    cv::rotate(m, m3, cv::ROTATE_90_CLOCKWISE);
    cv::imshow("case3", m3);

    cv::waitKey(0);

    return 0;
}