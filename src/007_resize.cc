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

void resize_bilinear(cv::Mat &src, cv::Mat &dst, cv::Size dsize)
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

// 权重计算公式：
//              /   (a+2)|x|^3 - (a+3)|x|^2 + 1          |x|<=1
//      W(x) = |    a|x|^3 - 5a|x|^2 + 8a|x| -4a       1<|x|<2
//              \   0                                    else
//      其中a通常取-0.5
float cal_weight(float x)
{
    float a = -0.5f;
    float x_abs = std::fabs(x);
    if (x_abs <= 1.0f) {
        return (float)((a+2)*std::pow(x_abs, 3) - (a+3)*std::pow(x_abs, 2) + 1);
    } else if (x_abs < 2.0f) {
        return (float)(a*std::pow(x_abs, 3) - 5*a*std::pow(x_abs, 2) + 8*a*x_abs - 4*a);
    } else {
        return 0.0f;
    }
}

struct PixelPoint {
    int x;
    int y;
    float weight;
};

std::vector<PixelPoint> generate16(float x, float y, int width, int height)
{
    std::vector<PixelPoint> vec;
    vec.resize(16);

    int i = (int)std::floor(x) - 1;
    int j = (int)std::floor(y) - 1;

    //calcurate weight
    for (int m = 0; m < 16; ++m) {
        vec[m].x = i + m % 4;
        vec[m].y = j + m / 4;
        vec[m].weight = cal_weight(x - vec[m].x) * cal_weight(y - vec[m].y);
    }
    //fixed coordinate
    for (auto &p: vec) {
        p.x = std::clamp(p.x, 0, width - 1);
        p.y = std::clamp(p.y, 0, height - 1);
    }

    return vec;
}

void resize_bicubic(cv::Mat &src, cv::Mat &dst, cv::Size dsize)
{
    cv::Size ssize(src.cols, src.rows);
    dst = cv::Mat(dsize, CV_8UC3);
    float scale_x = 1.0 * dsize.width / ssize.width;
    float scale_y = 1.0 * dsize.height / ssize.height;

    for (int x = 0; x < dsize.width; ++x) {
        float x0 = x / scale_x;
        for (int y = 0; y < dsize.height; ++y) {
            float y0 = y / scale_y;
            
            std::vector<PixelPoint> vec = generate16(x0, y0, ssize.width, ssize.height);
            cv::Vec3f color(0.0f, 0.0f, 0.0f);
            for (auto &p: vec) {
                color[0] += p.weight * src.at<cv::Vec3b>(p.y, p.x)[0];
                color[1] += p.weight * src.at<cv::Vec3b>(p.y, p.x)[1];
                color[2] += p.weight * src.at<cv::Vec3b>(p.y, p.x)[2];
            }
            color[0] = std::clamp(color[0], 0.0f, 255.0f);
            color[1] = std::clamp(color[1], 0.0f, 255.0f);
            color[2] = std::clamp(color[2], 0.0f, 255.0f);

            dst.at<cv::Vec3b>(y, x) = cv::Vec3b((uint8_t)color[0], (uint8_t)color[1], (uint8_t)color[2]);
        }
    }
}

// 根据面积变化比例划分原图，并进行求平均：仅支持缩小
void resize_area(cv::Mat &src, cv::Mat &dst, cv::Size dsize)
{
    cv::Size ssize(src.cols, src.rows);
    if (dsize.width > ssize.width || dsize.height > ssize.height) {
        std::cout << "not support zoom " << std::endl;
        return;
    }

    dst = cv::Mat(dsize, CV_8UC3);
    float delta_x = 1.0 * ssize.width / dsize.width;
    float delta_y = 1.0 * ssize.height / dsize.height;

    for (int x = 0; x < dsize.width; ++x) {
        for (int y = 0; y < dsize.height; ++y) {
            float u0 = delta_x * x, v0 = delta_y * y;
            float u1 = u0 + delta_x, v1 = v0 + delta_y;
            int u0_int = (int)u0, v0_int = (int)v0;
            int u1_int = (int)std::ceil(u1), v1_int = (int)std::ceil(v1);
            
            cv::Vec3f color(0.0f, 0.0f, 0.0f);
            int pixel_num = 0;
            for (int u = u0_int; u <= u1_int; ++u) {
                for (int v = v0_int; v <= v1_int; ++v) {
                    float m = 1.0f, n = 1.0f;
                    if (u == u0_int)
                        m = u0_int + 1 - u0;
                    if (v == v0_int)
                        n = v0_int + 1 - v0;
                    if (u == u1_int)
                        m = u1 - (u1_int - 1);
                    if (v == v1_int)
                        n = v1 - (v1_int - 1);
                    
                    float k = m * n;
                    color += src.at<cv::Vec3b>(v, u) * k;
                    pixel_num++;
                }
            }
            color /= pixel_num;

            color[0] = std::clamp(color[0], 0.0f, 255.0f);
            color[1] = std::clamp(color[1], 0.0f, 255.0f);
            color[2] = std::clamp(color[2], 0.0f, 255.0f);
            dst.at<cv::Vec3b>(y, x) = cv::Vec3b((uint8_t)color[0], (uint8_t)color[1], (uint8_t)color[2]);
        }
    }
}

int main(int argc, char *argv[])
{
    cv::Mat origin = cv::imread("../data/lena.jpg");
    int width = origin.cols, height = origin.rows;
    std::cout << "image size: " << width << "x" << height << std::endl;
    cv::imshow("origin", origin);

    // area
    cv::Mat area;
    resize_area(origin, area, cv::Size(128, 128));
    cv::imshow("area", area);

    // nearest
    cv::Mat nearest;
    resize_nearest(origin, nearest, cv::Size(1024, 1024));
    cv::imshow("nearest", nearest);

    // bilinear
    cv::Mat bilinear;
    resize_bilinear(origin, bilinear, cv::Size(1024, 1024));
    cv::imshow("bilinear", bilinear);

    // bicubic
    cv::Mat bicubic;
    resize_bicubic(origin, bicubic, cv::Size(1024, 1024));
    cv::imshow("bicubic", bicubic);

    cv::waitKey();
    return 0;
}