#include "get_view.h"
#include "disparity_map.h"
#include <iostream>
#include <fstream>
#include <cmath>

std::vector<double> GetCameraParams(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filePath);
    }

    std::string line1, line2;
    if (!std::getline(file, line1)) {
        throw std::runtime_error("Failed to read the first line");
    }
    if (!std::getline(file, line2)) {
        throw std::runtime_error("Failed to read the second line");
    }

    double val1, val2;
    std::istringstream ss1(line1), ss2(line2);
    if (!(ss1 >> val1)) {
        throw std::runtime_error("First line does not contain a valid double");
    }
    if (!(ss2 >> val2)) {
        throw std::runtime_error("Second line does not contain a valid double");
    }

    std::vector<double> params;
    params.reserve(2);
    params.push_back(val1);
    params.push_back(val2);
    return params;
}

View GetView(
    int frameIndex,
    bool rebuild,
    const std::string& disPath,
    const std::string& basePath,
    double focusDist,
    double camDist
)
{
    cv::Mat img1 = GetImage(basePath, frameIndex, 0);
    cv::Mat img2 = GetImage(basePath, frameIndex, 1);

    if (img1.empty() || img2.empty())
        throw std::runtime_error("Не удалось загрузить изображения для кадра " + std::to_string(frameIndex));

    int n = img1.rows;
    int m = img1.cols;

    cv::Mat img1g, img2g;
    cv::cvtColor(img1, img1g, cv::COLOR_RGB2GRAY);
    cv::cvtColor(img2, img2g, cv::COLOR_RGB2GRAY);

    cv::Mat bestShift = cv::Mat::zeros(n, m, CV_32F);
    cv::Mat bestShift2 = cv::Mat::zeros(n, m, CV_32F);

    if (rebuild)
    {
        std::string file0 = disPath + std::to_string(frameIndex) + "_0.txt";
        std::string file1 = disPath + std::to_string(frameIndex) + "_1.txt";
        //std::string disPathOut = basePath + "disF/";

        if (!BuildDisparityMap(img1g, img2g, bestShift, false, file0))
            throw std::runtime_error("Ошибка построения карты диспаратности для левого изображения.");
        //FilterDisparityMap(frameIndex, basePath, disPath, disPathOut, focusDist, camDist);
        if (!BuildDisparityMap(img2g, img1g, bestShift2, true, file1))
            throw std::runtime_error("Ошибка построения карты диспаратности для правого изображения.");
    }

    bestShift = ReadDisparityMap(n, m, disPath + std::to_string(frameIndex) + "_0.txt");
    bestShift2 = ReadDisparityMap(n, m, disPath + std::to_string(frameIndex) + "_1.txt");

    if (bestShift.empty() || bestShift2.empty())
        throw std::runtime_error("Не удалось прочитать карты диспаратности.");

    View fullView;
    fullView.cloud = std::make_shared<open3d::geometry::PointCloud>();

    if (!LoadViewSimple(basePath, frameIndex, false, bestShift, focusDist, camDist, fullView))
        throw std::runtime_error("Ошибка загрузки вида для кадра " + std::to_string(frameIndex));

    return fullView;
}