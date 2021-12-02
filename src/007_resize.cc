#include <stdio.h>
#include <stdint.h>
#include <cmath>
#include <algorithm>  //clamp since c++17
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void resize_nearest(cv::Mat &src, cv::Mat &dst, cv::Size dsize)
{
    cv::Size ssize(src.cols, src.rows);
    dst = cv::Mat(dsize, CV_8UC3);
    float scale_x = 1.0 * dsize.width / ssize.width;
    float scale_y = 1.0 * dsize.height / ssize.height;

    for (int x = 0; x < dsize.width; ++x) {
        float x0 = x / scale_x;
        for (int y = 0; y < dsize.height; ++y) {
            float y0 = y / scale_y;
            int i = std::clamp((int)std::round(x0), 0, ssize.width - 1);
            int j = std::clamp((int)std::round(y0), 0, ssize.height - 1);
            
            dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(j, i);
        }
    }
}

void resize_linear(cv::Mat &src, cv::Mat &dst, cv::Size dsize)
{
    cv::Size ssize(src.cols, src.rows);
    dst = cv::Mat(dsize, CV_8UC3);
    float scale_x = 1.0 * dsize.width / ssize.width;
    float scale_y = 1.0 * dsize.height / ssize.height;

    for (int x = 0; x < dsize.width; ++x) {
        float x0 = x / scale_x;
        if (x0 > ssize.width - 1) x0 = ssize.width - 1;

        for (int y = 0; y < dsize.height; ++y) {
            float y0 = y / scale_y;
            if (y0 > ssize.height - 1) y0 = ssize.height - 1;
            
            int u0 = std::clamp((int)std::floor(x0), 0, ssize.width - 1);
            int v0 = std::clamp((int)std::floor(y0), 0, ssize.height - 1);
            int u1 = std::clamp(u0 + 1, 0, ssize.width - 1);
            int v1 = std::clamp(v0 + 1, 0, ssize.height - 1);
            
            float kx = std::clamp(u1 - x0, 0.0f, 1.0f);
            float ky = std::clamp(v1 - y0, 0.0f, 1.0f);

            uint8_t b0 = (uint8_t)(src.at<cv::Vec3b>(v0, u0)[0] * kx + src.at<cv::Vec3b>(v0, u1)[0] * (1 - kx));
            uint8_t b1 = (uint8_t)(src.at<cv::Vec3b>(v1, u0)[0] * kx + src.at<cv::Vec3b>(v1, u1)[0] * (1 - kx));
            uint8_t b2 = (uint8_t)(b0 * ky + b1 * (1 - ky));

            uint8_t g0 = (uint8_t)(src.at<cv::Vec3b>(v0, u0)[1] * kx + src.at<cv::Vec3b>(v0, u1)[1] * (1 - kx));
            uint8_t g1 = (uint8_t)(src.at<cv::Vec3b>(v1, u0)[1] * kx + src.at<cv::Vec3b>(v1, u1)[1] * (1 - kx));
            uint8_t g2 = (uint8_t)(g0 * ky + g1 * (1 - ky));

            uint8_t r0 = (uint8_t)(src.at<cv::Vec3b>(v0, u0)[2] * kx + src.at<cv::Vec3b>(v0, u1)[2] * (1 - kx));
            uint8_t r1 = (uint8_t)(src.at<cv::Vec3b>(v1, u0)[2] * kx + src.at<cv::Vec3b>(v1, u1)[2] * (1 - kx));
            uint8_t r2 = (uint8_t)(r0 * ky + r1 * (1 - ky));

            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(b2, g2, r2);
        }
    }
}

int main(int argc, char *argv[])
{
    cv::Mat origin = cv::imread("../data/lena.jpg");
    int width = origin.cols, height = origin.rows;
    std::cout << "image size: " << width << "x" << height << std::endl;
    cv::imshow("origin", origin);

    // nearest
    cv::Mat nearest;
    resize_nearest(origin, nearest, cv::Size(1024, 1024));
    cv::imshow("nearest", nearest);

    // linear
    cv::Mat linear;
    resize_linear(origin, linear, cv::Size(1024, 1024));
    cv::imshow("linear", linear);

    cv::waitKey();
    return 0;
}