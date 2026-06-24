#ifndef REMOVE_BACKGROUND_H
#define REMOVE_BACKGROUND_H

#include <opencv2/opencv.hpp>
#include <vector>

double VectorDistance(cv::Vec3b v1, cv::Vec3b v2);

void ConsumeRegion(
    int y,
    int x,
    std::vector<std::vector<int>>& mask,
    cv::Mat& img,
    int depth,
    int consumeThreshold
);

void ConsumeRegionHSV(
    int y,
    int x,
    std::vector<std::vector<int>>& mask,
    cv::Mat& img,
    int depth,
    int consumeHue,
    int consumeSaturation,
    int consumeValue
);

std::vector<std::vector<int>> RemoveBackgroundWithDisparity(
    cv::Mat& img,
    const cv::Mat& dis,
    int disBottomBound = 20,
    int disFloorRange = 10,
    int consumeThreshold = 3,
    cv::Vec3b bgColor = cv::Vec3b(109, 153, 148)
);

std::vector<std::vector<int>> RemoveBackgroundHSV(
    const cv::Mat& img,
    cv::Mat& out,
    int hue = 10,
    int saturation = 20,
    int value = 20,
    int consumeHue = 5,
    int consumeSaturation = 10,
    int consumeValue = 10,
    int saturationCutoff = 30
);

std::vector<std::vector<int>> ColorSegmentationAdaptiveThreshold(
    const cv::Mat& imgGray,
    cv::Mat& out,
    int thresh1 = 1,
    int thresh2 = 2
);

int ColorSegmentationKMeans(const cv::Mat& img, cv::Mat& out, int K = 3);

#endif // REMOVE_BACKGROUND_H