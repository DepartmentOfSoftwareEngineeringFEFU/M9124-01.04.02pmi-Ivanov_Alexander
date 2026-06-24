#ifndef LOAD_VIEW_H
#define LOAD_VIEW_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <Open3D/Open3D.h>
#include <string>
#include <vector>
#include <utility>
#include <memory>

struct View
{
    int number;
    std::string name;
    Eigen::Matrix4f unProj;
    Eigen::Matrix4f model;
    Eigen::Matrix4f unProj_i;
    Eigen::Matrix4f model_i;
    cv::Mat pic;
    cv::Mat pic2;
    std::pair<int, int> picSize;              // (rows, cols)
    std::vector<Eigen::Vector3d> bordPoints;
    std::shared_ptr<open3d::geometry::PointCloud> cloud;
};

Eigen::Matrix4f GetFirstModelMatrix(const std::string& basePath, int frameIndex);

cv::Mat GetImage(const std::string& basePath, int frameIndex, int stereoIndex);

bool LoadViewSimple(
    const std::string& basePath,
    int frameIndex,
    bool MSK,
    const cv::Mat& img1d,
    double focusDist,
    double camDist,
    View& outView
);

#endif // LOAD_VIEW_H