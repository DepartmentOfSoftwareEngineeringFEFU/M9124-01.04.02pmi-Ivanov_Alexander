#ifndef DISPARITY_MAP_H
#define DISPARITY_MAP_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Rect
{
    int x1;
    int x2;
    int y1;
    int y2;
    int vol;
};

bool WriteMatBinary(const std::string& filename, const cv::Mat& output);
bool ReadMatBinary(const std::string& filename, cv::Mat& output);

void CalculateDisparityForSegment(
    cv::Mat img1g,
    cv::Mat img2g,
    std::vector<Rect> windows,
    cv::Mat& bestShift,
    bool reverse,
    int l,
    int r
);

bool BuildDisparityMap(
    cv::Mat img1g,
    cv::Mat img2g,
    cv::Mat& bestShift,
    bool reverse,
    const std::string& filepath
);

cv::Mat ReadDisparityMap(int rows, int cols, const std::string& filepath);

#endif // DISPARITY_MAP_H