// 参考文章：  opencv关于对比度和亮度的误解 (https://blog.csdn.net/abc20002929/article/details/40474807)
//
// 亮度调整：图像像素强度整体变高/变低
// 对比度调整：图像暗处像素强度变低，图像亮处像素强度变高，从而拉大中间某个区域范围的显示精度
//           (注意：对比度调整会同时改变亮度值)
//
#include <stdio.h>
#include <stdint.h>
#include <cmath>
#include <algorithm>  //clamp since c++17
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

float g_light = 0.0f;

// 这里是简单的调节，更好的方式是将RGB转换到HSL颜色空间，然后对L通道调节，得到新的HSL，最后转换回RGB
void TrackbarCallbackLight(int pos, void* userdata)
{
    cv::Mat *img = (cv::Mat *)userdata;
    float light = pos * 2.0 - 100; // [0:100] => [-100.0:100.0]
    cv::Mat imgNew;
    img->convertTo(imgNew, CV_8U, 1.0f, light);
    cv::imshow("img", imgNew);
    g_light = light;  //对比度在此亮度上进行调节
}


// 我们在128上进行对比度调节，计算公式如下：
//    y = (x-128)*alpha + 128
// 等价于：
//    y = x * alpha + beta, beta = (1 - alpha)*128
void TrackbarCallbackContrast(int pos, void* userdata)
{
    cv::Mat *img = (cv::Mat *)userdata;
    float alpha = pos / 50.0; // [0:100] => [0.0:2.0]
    float beta = (1 - alpha) * 128 + g_light;
    cv::Mat imgNew;
    img->convertTo(imgNew, CV_8U, alpha, beta);
    cv::imshow("img", imgNew);
}

// 这里是简单的方法：增加像素强度值，然后减去亮度值
// 更好的方式是将RGB转换到HSL颜色空间，然后对S通道调节，得到新的HSL，最后转换回RGB
void TrackbarCallbackSaturation(int pos, void* userdata)
{
    cv::Mat *img = (cv::Mat *)userdata;
    cv::Mat imgNew = img->clone();
    float saturation = pos / 50.0; // [0:100] => [0.0:2.0]

    for (int i = 0; i < img->cols; ++i) {
        for (int j = 0; j < img->rows; ++j) {
            uchar R = img->at<cv::Vec3b>(j, i)[2];
            uchar G = img->at<cv::Vec3b>(j, i)[1];
            uchar B = img->at<cv::Vec3b>(j, i)[0];
            float gray = 0.3*R + 0.59*G + 0.11*B;

            for (int k = 0; k < 3; ++k) {
                imgNew.at<cv::Vec3b>(j, i)[k] = cv::saturate_cast<uchar>(img->at<cv::Vec3b>(j, i)[k] * saturation + gray * (1 - saturation));
            }
        }
    }
    cv::imshow("img", imgNew);
}

int main(int argc, char **argv)
{
    cv::Mat img = cv::imread("../data/lena.jpg");
    if (img.channels() != 3) {
        printf("only test bgr \n");
        return -1;
    }
    int light = 50, contrast = 50, saturation = 50; // 0~100

    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    cv::imshow("img", img);
    cv::createTrackbar("light", "img", &light, 100, TrackbarCallbackLight, &img);
    cv::createTrackbar("contr", "img", &contrast, 100, TrackbarCallbackContrast, &img);
    cv::createTrackbar("satur", "img", &saturation, 100, TrackbarCallbackSaturation, &img);

    cv::waitKey();
    return 0;
}