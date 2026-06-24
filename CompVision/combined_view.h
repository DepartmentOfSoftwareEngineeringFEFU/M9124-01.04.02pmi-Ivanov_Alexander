#ifndef COMBINED_VIEW_H
#define COMBINED_VIEW_H

#include <Eigen/Dense>
#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>
#include <string>
#include "load_view.h"
#include "model_matrix.h"
#include "plan_frame_sequence.h"

std::vector<size_t> GetBackgroundIndices(const View& view,
    const cv::Mat& img,
    const cv::Mat& dis,
    int rows,
    int cols);

int CombinedView(const std::vector<int>& sparseSeq,
    const std::vector<int>& reconstructionSeq,
    const std::vector<Eigen::Matrix4d>& transformMatrices,
    const std::string& disPath,
    const std::string& basePath,
    double focusDist,
    double baseline,
    const std::string& outputObjPath);

int reconstructFromFrameSequence(const std::vector<int>& frameSequence,
    int totalFrames,
    int maxStep,
    const std::string basePath,
    const std::string cameraParamsPath);

#endif // COMBINED_VIEW_H