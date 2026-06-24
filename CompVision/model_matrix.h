#ifndef MODEL_MATRIX_H
#define MODEL_MATRIX_H

#include <Eigen/Dense>
#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "load_view.h"

struct PointPair
{
    Eigen::Vector3d p1;
    Eigen::Vector3d p2;
};

double Distance3D(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2);

void DrawRegistrationResult(
    const open3d::geometry::PointCloud& source,
    const open3d::geometry::PointCloud& target,
    const Eigen::Matrix4d& transformation,
    bool colored
);

Eigen::Matrix4d GetModelMatrix(const std::vector<PointPair>& corPoints3ds);

int CountCorPointsMatches(const cv::Mat& img1, const cv::Mat& img2);

std::vector<PointPair> GetCorPointsFiltered(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const View& view1,
    const View& view2,
    const cv::Mat& dis1,
    const cv::Mat& dis2,
    const cv::Mat& dis1rev,
    const cv::Mat& dis2rev
);

Eigen::Matrix4d GetMatrix4Way(
    const View& view1,
    const View& view2,
    const Eigen::Matrix4d& transformMatrix,
    double focusDist,
    double baseline
);

std::vector<Eigen::Matrix4d> ReadModelMatrices(const std::string& filepath);

void BuildModelMatrices(
    const std::vector<int>& frameSequence,
    const std::string& basePath,
    const std::string& disPath,
    double focusDist,
    double baseline,
    const std::string& outputPath
);

open3d::pipelines::registration::RegistrationResult RefineRegistration(
    const open3d::geometry::PointCloud& source,
    const open3d::geometry::PointCloud& target,
    std::shared_ptr<open3d::pipelines::registration::Feature> source_fpfh,
    std::shared_ptr<open3d::pipelines::registration::Feature> target_fpfh,
    float voxel_size,
    const open3d::pipelines::registration::RegistrationResult& resultRansac
);

#endif // MODEL_MATRIX_H