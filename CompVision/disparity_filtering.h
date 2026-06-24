#ifndef DISPARITY_FILTERING_H
#define DISPARITY_FILTERING_H

#include <opencv2/core.hpp>
#include <string>

int ColorToInt(const cv::Scalar& a);

cv::Scalar IntToColor(int a);

void FilterDisparityMap(
    int frameIndex,
    const std::string& basePath,
    const std::string& disPath,
    const std::string& disPathOut,
    double focusDist = 642.7,
    double camDist = 0.3
);

#endif // DISPARITY_FILTERING_H