#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// crop:
//    ffmpeg -i input.jpg -vf crop=1160:400 feathering_1160x400.jpg   (居中裁剪)
//    ffmpeg -i input.jpg -vf crop=580:400:0:0 out.jpg   (指定起始位置)

uint8_t add_delta(uint8_t a, uint8_t b)
{
    int sum = (int)a + b;
    sum = sum > 255 ? 255 : sum;
    return (uint8_t)sum;
}

void change_color(const char *file_in, const char *file_out)
{
    cv::Mat img_in = cv::imread(file_in);
    cv::Mat img_out(img_in.rows, img_in.cols, CV_8UC3);

    for (int row = 0; row < img_in.size().height; ++row) {
        for (int col = 0; col < img_in.size().width; ++col) {
            img_out.at<cv::Vec3b>(row, col)[0] = add_delta(img_in.at<cv::Vec3b>(row, col)[0], 30); //B
            img_out.at<cv::Vec3b>(row, col)[1] = add_delta(img_in.at<cv::Vec3b>(row, col)[1], 0); //G
            img_out.at<cv::Vec3b>(row, col)[2] = add_delta(img_in.at<cv::Vec3b>(row, col)[2], 0); //R
        }
    }

    cv::imwrite(file_out, img_out);
}

// 根据当前位置与两个图像边界的距离，进行加权平均
uint8_t CalValue(int start, int end, int cur, uint8_t val1, uint8_t val2)
{
    int w = end - start; //重叠宽度
    float k = 1.0 * (end - cur)/w; //计算k
    int val = (int)(val1 * k + val2 * (1 - k));

    val = val > 255 ? 255 : val; //确保输出范围为0-255
    return (uint8_t)val;
}

int main(int argc, char *argv[])
{
    // generate test file
    change_color("../data/feathering0.jpg", "../data/feathering1.jpg");

    cv::Mat img_dst(400, 1100, CV_8UC3);
    cv::Mat img_src1 = cv::imread("../data/feathering1.jpg");
    cv::Mat img_src2 = cv::imread("../data/feathering2.jpg");

    for (int row = 0; row < 400; ++row) {
        // 0-470: 图像1独立区域
        for (int col = 0; col < 470; ++col) {
            img_dst.at<cv::Vec3b>(row, col) = img_src1.at<cv::Vec3b>(row, col);
        }
        // 630-1100: 图像2独立区域
        for (int col = 630; col < 1100; ++col) {
            img_dst.at<cv::Vec3b>(row, col) = img_src2.at<cv::Vec3b>(row, col-470);
        }
        // 470-630: 重叠区域，重叠宽度为160
        for (int col = 470; col < 630; ++col) {
            img_dst.at<cv::Vec3b>(row, col)[0] = CalValue(470, 630, col, 
                img_src1.at<cv::Vec3b>(row, col)[0], img_src2.at<cv::Vec3b>(row, col-470)[0]);
            img_dst.at<cv::Vec3b>(row, col)[1] = CalValue(470, 630, col, 
                img_src1.at<cv::Vec3b>(row, col)[1], img_src2.at<cv::Vec3b>(row, col-470)[1]);
            img_dst.at<cv::Vec3b>(row, col)[2] = CalValue(470, 630, col, 
                img_src1.at<cv::Vec3b>(row, col)[2], img_src2.at<cv::Vec3b>(row, col-470)[2]);
        }
    }
    cv::imwrite("../data/feathering3.jpg", img_dst);

    cv::imshow("img1", img_src1);
    cv::imshow("img2", img_src2);
    cv::imshow("img3", img_dst);
    cv::waitKey(0);
    return 0;
}