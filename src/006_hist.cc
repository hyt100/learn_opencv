#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Histogram
{
public:
    // 调用OpenCV接口计算直方图
    static cv::Mat GetHistMat(const cv::Mat &image, int chn)
    {
        cv::Mat hist;
        int histSize[1] = {256}; // 项的数量
        float hranges[2] = {0.0f, 255.0f}; // 统计像素的最大值和最小值
        const float* ranges[1] = {hranges};
        int channels[1] = {chn}; // 仅计算一个通道

        // 计算直方图
        cv::calcHist(&image, // 要计算图像的
            1,               // 只计算一幅图像的直方图
            channels,        // 通道数量
            cv::Mat(),       // 不使用掩码
            hist,            // 存放直方图
            1,               // 1D直方图
            histSize,        // 统计的灰度的个数
            ranges);         // 灰度值的范围
        return hist;
    }

    // 直接计算直方图
    static cv::Mat GetHistMat2(const cv::Mat &image, int chn)
    {
        int histSize = 256;
        cv::Mat hist = cv::Mat::zeros(1, histSize, CV_32FC1);
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                int location;

                if (image.channels() == 1) { //灰度图，单通道
                    uchar val = image.at<uchar>(i, j);
                    location = val;
                } else { //彩色图，三通道
                    cv::Vec3b val = image.at<cv::Vec3b>(i, j);
                    location = val[chn];
                }

                hist.at<float>(location) ++;
            }
        }
        return hist;
    }

    static cv::Mat GetHistImage(const cv::Mat &image, int chn, const cv::Scalar& color)
    {
        cv::Mat hist = GetHistMat(image, chn);
        int histSize = 256;

        // 最大值，最小值
        double maxVal = 0.0f;
        double minVal = 0.0f;
        cv::minMaxLoc(hist, &minVal, &maxVal);

        //显示直方图的图像
        cv::Mat histImg = cv::Mat::zeros(histSize, histSize, CV_8UC3);

        //每个条目绘制一条垂直线
        for (int h = 0; h < histSize; h++)
        {
            float binVal = hist.at<float>(h); //注意hist中是float类型
            int intensity = static_cast<int>(binVal / maxVal * histSize);  //先将值范围转换到0~1，然后乘以histSize
            // 两点之间绘制一条直线
            cv::line(histImg, cv::Point(h, histSize), cv::Point(h, histSize - intensity), color);
        }
        return histImg;
    }
};

int main(int argc, char *argv[])
{
    // 彩色图像直方图
    cv::Mat image = cv::imread("../data/lena.jpg");
    cv::Mat hist_b = Histogram::GetHistImage(image, 0, CV_RGB(0, 0, 255));
    cv::Mat hist_g = Histogram::GetHistImage(image, 1, CV_RGB(0, 255, 0));
    cv::Mat hist_r = Histogram::GetHistImage(image, 2, CV_RGB(255, 0, 0));

    // 灰度图像直方图
    cv::Mat image_gray = cv::imread("../data/lena.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat hist_gray = Histogram::GetHistImage(image_gray, 0, CV_RGB(255, 255, 255));

    // 灰度图像均衡化
    cv::Mat image_equal;
    cv::equalizeHist(image_gray, image_equal); // equalizeHist接口只能处理单通道图像
    cv::Mat hist_equal = Histogram::GetHistImage(image_equal, 0, CV_RGB(255, 255, 255));

    // 彩色图像均衡化
    // (注意：彩色图像做直方图均衡化不能对三个分量分别均衡化，这么搞可能会导致颜色畸变。
    //       最好将彩色图像转化为HLS颜色模型，然后单独对L分量做均衡化，最后再转换回BGR颜色模型)

    cv::imshow("image", image);
    cv::imshow("hist_r", hist_r);
    cv::imshow("hist_g", hist_g);
    cv::imshow("hist_b", hist_b);

    cv::imshow("image_gray", image_gray);
    cv::imshow("hist_gray", hist_gray);
    cv::imshow("image_equal", image_equal);
    cv::imshow("hist_equal", hist_equal);

    cv::waitKey(0);
    return 0;
}