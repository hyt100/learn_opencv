#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

int main()
{
    printf("%d.%d.%d \n", cv::getVersionMajor(), cv::getVersionMinor(), cv::getVersionRevision());
    return 0;
}