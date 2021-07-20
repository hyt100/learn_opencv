#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

int main()
{
    // 打印opencv版本信息
    printf("%d.%d.%d \n", cv::getVersionMajor(), cv::getVersionMinor(), cv::getVersionRevision());

    // 打印opencv编译信息
    printf("%s \n", cv::getBuildInformation().c_str());
    return 0;
}