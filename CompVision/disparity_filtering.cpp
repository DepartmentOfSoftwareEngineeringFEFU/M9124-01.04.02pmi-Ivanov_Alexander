#include "disparity_filtering.h"
#include "load_view.h"
#include "disparity_map.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ximgproc.hpp>
#include <Open3D/Open3D.h>
#include <iostream>
#include <stdexcept>

int ColorToInt(const cv::Scalar& a)
{
    return static_cast<int>(a[0] * 256 * 256 + a[1] * 256 + a[2]);
}

cv::Scalar IntToColor(int a)
{
    return cv::Scalar(a / (256 * 256), (a % (256 * 256)) / 256, a % 256);
}

void FilterDisparityMap(
    int frameIndex,
    const std::string& basePath,
    const std::string& disPath,
    const std::string& disPathOut,
    double focusDist,
    double camDist)
{
    cv::Mat img1 = GetImage(basePath, frameIndex, 0);
    cv::Mat img2 = GetImage(basePath, frameIndex, 1);
    if (img1.empty() || img2.empty())
        throw std::runtime_error("Не удалось загрузить изображения для кадра " + std::to_string(frameIndex));

    int n = img1.rows, m = img1.cols;

    cv::Mat left_disp = ReadDisparityMap(n, m, disPath + std::to_string(frameIndex) + "_0.txt");
    cv::Mat right_disp = ReadDisparityMap(n, m, disPath + std::to_string(frameIndex) + "_1.txt");
    if (left_disp.empty() || right_disp.empty())
        throw std::runtime_error("Не удалось прочитать карты диспаратности для кадра " + std::to_string(frameIndex));

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            left_disp.at<float>(i, j) *= 16.0f;
            right_disp.at<float>(i, j) *= 16.0f;
            int idx = j - static_cast<int>(left_disp.at<float>(i, j) / 16.0f);
            if (idx < 0 || idx >= m ||
                std::abs(left_disp.at<float>(i, j) + right_disp.at<float>(i, idx)) > 0.0f)
            {
                left_disp.at<float>(i, j) = -1.0f;
            }
        }
    }

    int windowSize = 10;
    cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(
        20,                                  // minDisparity
        50,                                  // numDisparities
        windowSize,                          // blockSize
        8 * 3 * windowSize * windowSize / 2, // P1
        32 * 3 * windowSize * windowSize / 2,// P2
        8,                                   // disp12MaxDiff
        0,                                   // preFilterCap
        10,                                  // uniquenessRatio
        10,                                  // speckleWindowSize
        10,                                  // speckleRange
        cv::StereoSGBM::MODE_HH
    );
    cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter =
        cv::ximgproc::createDisparityWLSFilter(left_matcher);
    wls_filter->setLambda(8000.0);
    wls_filter->setSigmaColor(2.45);
    wls_filter->setLRCthresh(8);
    wls_filter->setDepthDiscontinuityRadius(1);

    cv::Mat filtered_left, filtered_right;
    wls_filter->filter(left_disp, img1, filtered_left, right_disp);
    wls_filter->filter(right_disp, img2, filtered_right, left_disp);

    filtered_left /= 16.0f;
    filtered_right /= 16.0f;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            float valL = filtered_left.at<float>(i, j);
            float valR = filtered_right.at<float>(i, j);
            if (valL < 20.0f || valL > 255.0f)
                filtered_left.at<float>(i, j) = 0.0f;
            if (valR < 20.0f || valR > 255.0f)
                filtered_right.at<float>(i, j) = 0.0f;
        }
    }

    if (!WriteMatBinary(disPathOut + std::to_string(frameIndex) + "_0.txt", filtered_left))
        throw std::runtime_error("Не удалось сохранить левую отфильтрованную карту для кадра " + std::to_string(frameIndex));
    if (!WriteMatBinary(disPathOut + std::to_string(frameIndex) + "_1.txt", filtered_right))
        throw std::runtime_error("Не удалось сохранить правую отфильтрованную карту для кадра " + std::to_string(frameIndex));

    cv::Mat bestShift = ReadDisparityMap(n, m, disPathOut + std::to_string(frameIndex) + "_0.txt");
    if (bestShift.empty())
        throw std::runtime_error("Не удалось прочитать сохранённую отфильтрованную карту для кадра " + std::to_string(frameIndex));

    View fullView;
    fullView.cloud = std::make_shared<open3d::geometry::PointCloud>();
    if (!LoadViewSimple(basePath, frameIndex, false, bestShift, focusDist, camDist, fullView))
        throw std::runtime_error("Не удалось загрузить вид для кадра " + std::to_string(frameIndex));

    open3d::visualization::DrawGeometries({ fullView.cloud }, "Open3D", 640, 480, 50, 50);
    cv::waitKey(0);
}