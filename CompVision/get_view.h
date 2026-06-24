#ifndef GET_VIEW_H
#define GET_VIEW_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "load_view.h"

std::vector<double> GetCameraParams(
    const std::string& filePath
);

View GetView(
    int frameIndex,
    bool rebuild,
    const std::string& disPath,
    const std::string& basePath,
    double focusDist,
    double baseline
);

#endif // GET_VIEW_H