#include "model_matrix.h"
#include "disparity_map.h"
#include "remove_background.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <stdexcept>

double Distance3D(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2)
{
    return (v1 - v2).norm();
}

void DrawRegistrationResult(
    const open3d::geometry::PointCloud& source,
    const open3d::geometry::PointCloud& target,
    const Eigen::Matrix4d& transformation,
    bool colored)
{
    open3d::geometry::PointCloud sourceTemp = source;
    open3d::geometry::PointCloud targetTemp = target;
    if (colored)
    {
        sourceTemp.PaintUniformColor({ 1, 0, 0 });
        targetTemp.PaintUniformColor({ 0, 0, 1 });
    }
    targetTemp.Transform(transformation.inverse());
    auto cloud1 = std::make_shared<open3d::geometry::PointCloud>(sourceTemp);
    auto cloud2 = std::make_shared<open3d::geometry::PointCloud>(targetTemp);
    open3d::visualization::DrawGeometries({ cloud1, cloud2 }, "DrawRegistrationResult", 640, 480, 50, 50);
}

Eigen::Matrix4d GetModelMatrix(const std::vector<PointPair>& corPoints3ds)
{
    if (corPoints3ds.empty())
        throw std::runtime_error("Нет соответствующих 3D-точек для вычисления матрицы.");

    auto pcl1 = std::make_shared<open3d::geometry::PointCloud>();
    auto pcl2 = std::make_shared<open3d::geometry::PointCloud>();
    std::vector<Eigen::Vector2i> corres;
    for (size_t i = 0; i < corPoints3ds.size(); ++i)
    {
        pcl1->points_.push_back(corPoints3ds[i].p1);
        pcl2->points_.push_back(corPoints3ds[i].p2);
        corres.push_back(Eigen::Vector2i(static_cast<int>(i), static_cast<int>(i)));
    }

    open3d::pipelines::registration::TransformationEstimationPointToPoint transfEst;
    Eigen::Matrix4d result = transfEst.ComputeTransformation(*pcl1, *pcl2, corres);

    // Отображаем исходное и преобразованное облака
    DrawRegistrationResult(*pcl1, *pcl2, Eigen::Matrix4d::Identity(), true);
    DrawRegistrationResult(*pcl1, *pcl2, result, true);

    return result;
}

int CountCorPointsMatches(const cv::Mat& img1, const cv::Mat& img2)
{
    int minHessian = 100;
    auto detector = cv::xfeatures2d::SURF::create(minHessian);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    detector->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    detector->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(desc1, desc2, knnMatches, 2);

    const float ratioThresh = 0.20f;
    std::vector<cv::DMatch> goodMatches;
    for (const auto& match : knnMatches)
    {
        if (match.size() >= 2 && match[0].distance < ratioThresh * match[1].distance)
            goodMatches.push_back(match[0]);
    }
    return static_cast<int>(goodMatches.size());
}

std::vector<PointPair> GetCorPointsFiltered(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const View& view1,
    const View& view2,
    const cv::Mat& dis1,
    const cv::Mat& dis2,
    const cv::Mat& dis1rev,
    const cv::Mat& dis2rev)
{
    double disErrorTolerance = 0.7;
    float ratioThresh = 0.15f;
    int minHessian = 500;
    std::vector<PointPair> corPoints;

    while (corPoints.size() < 20)
    {
        corPoints.clear();

        auto detector = cv::xfeatures2d::SURF::create(minHessian);
        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat desc1, desc2;
        detector->detectAndCompute(img1, cv::noArray(), kp1, desc1);
        detector->detectAndCompute(img2, cv::noArray(), kp2, desc2);

        auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(desc1, desc2, knnMatches, 2);

        std::vector<cv::DMatch> goodMatches;
        for (const auto& match : knnMatches)
        {
            if (match.size() >= 2 && match[0].distance < ratioThresh * match[1].distance)
                goodMatches.push_back(match[0]);
        }

        int n = img1.rows, m = img1.cols;
        for (const auto& match : goodMatches)
        {
            cv::Point2f pt1 = kp1[match.queryIdx].pt;
            cv::Point2f pt2 = kp2[match.trainIdx].pt;
            int x1 = static_cast<int>(std::round(pt1.x));
            int y1 = static_cast<int>(std::round(pt1.y));
            int x2 = static_cast<int>(std::round(pt2.x));
            int y2 = static_cast<int>(std::round(pt2.y));

            if (x1 < 0 || y1 < 0 || x1 >= m || y1 >= n ||
                x2 < 0 || y2 < 0 || x2 >= m || y2 >= n)
                continue;

            PointPair pair;
            pair.p1 = view1.cloud->points_[y1 * m + x1];
            pair.p2 = view2.cloud->points_[y2 * m + x2];

            float d1 = dis1.at<float>(y1, x1);
            float d2 = dis2.at<float>(y2, x2);
            if (d1 > 0 && d2 > 0 && x1 - d1 > 0 && x2 - d2 > 0)
            {
                int idx1 = x1 - static_cast<int>(d1);
                int idx2 = x2 - static_cast<int>(d2);
                if (std::abs(d1 + dis1rev.at<float>(y1, idx1)) <= disErrorTolerance &&
                    std::abs(d2 + dis2rev.at<float>(y2, idx2)) <= disErrorTolerance)
                {
                    corPoints.push_back(pair);
                }
            }
        }

        if (corPoints.size() >= 20)
            break;

        if (disErrorTolerance < 4.0)
            disErrorTolerance *= 2.0;
        else if (minHessian > 50)
            minHessian /= 2;
        else if (ratioThresh < 0.35f)
            ratioThresh += 0.01f;
        else
            break;
    }
    return corPoints;
}

Eigen::Matrix4d GetMatrix4Way(
    const View& view1,
    const View& view2,
    const Eigen::Matrix4d& transformMatrix,
    double focusDist,
    double baseline)
{
    cv::Mat img1L = view1.pic;
    cv::Mat img1R = view1.pic2;
    cv::Mat img2L = view2.pic;
    cv::Mat img2R = view2.pic2;

    float ratioThresh = 0.15f;
    int minHessian = 500;
    std::vector<PointPair> corPoints;

    while (corPoints.size() < 25)
    {
        corPoints.clear();

        auto detector = cv::xfeatures2d::SURF::create(minHessian);
        std::vector<cv::KeyPoint> kp1L, kp1R, kp2L, kp2R;
        cv::Mat desc1L, desc1R, desc2L, desc2R;
        detector->detectAndCompute(img1L, cv::noArray(), kp1L, desc1L);
        detector->detectAndCompute(img1R, cv::noArray(), kp1R, desc1R);
        detector->detectAndCompute(img2L, cv::noArray(), kp2L, desc2L);
        detector->detectAndCompute(img2R, cv::noArray(), kp2R, desc2R);

        auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector<std::vector<cv::DMatch>> knn1L_R, knn2L_R, knn1L_2L, knn1R_2R;
        matcher->knnMatch(desc1L, desc1R, knn1L_R, 2);
        matcher->knnMatch(desc2L, desc2R, knn2L_R, 2);
        matcher->knnMatch(desc1L, desc2L, knn1L_2L, 2);
        matcher->knnMatch(desc1R, desc2R, knn1R_2R, 2);

        auto filter = [&](const std::vector<std::vector<cv::DMatch>>& knn, float ratio)
            {
                std::vector<cv::DMatch> good;
                for (const auto& m : knn)
                    if (m.size() >= 2 && m[0].distance < ratio * m[1].distance)
                        good.push_back(m[0]);
                return good;
            };

        std::vector<cv::DMatch> goodL_R = filter(knn1L_R, ratioThresh);
        std::vector<cv::DMatch> good2L_R = filter(knn2L_R, ratioThresh);
        std::vector<cv::DMatch> goodL_2L = filter(knn1L_2L, ratioThresh);
        std::vector<cv::DMatch> goodR_2R = filter(knn1R_2R, ratioThresh);

        int n = img1L.rows, m = img1L.cols;

        for (const auto& m1 : goodL_R)
        {
            const cv::KeyPoint& k1L = kp1L[m1.queryIdx];
            const cv::KeyPoint& k1R = kp1R[m1.trainIdx];
            for (const auto& mL : goodL_2L)
            {
                if (m1.queryIdx == mL.queryIdx)
                {
                    const cv::KeyPoint& k2L = kp2L[mL.trainIdx];
                    for (const auto& m2 : good2L_R)
                    {
                        if (mL.trainIdx == m2.queryIdx)
                        {
                            const cv::KeyPoint& k2R = kp2R[m2.trainIdx];
                            double x1L = k1L.pt.x, y1L = k1L.pt.y;
                            double x1R = k1R.pt.x, y1R = k1R.pt.y;
                            Eigen::Vector3d P1;
                            P1(0) = -m / 2.0 + x1L + 0.5;
                            P1(1) = -n / 2.0 + y1L + 0.5;
                            P1(2) = -focusDist * baseline / (x1L - x1R);
                            P1(0) = -P1(0) * P1(2) / focusDist;
                            P1(1) = P1(1) * P1(2) / focusDist;

                            double x2L = k2L.pt.x, y2L = k2L.pt.y;
                            double x2R = k2R.pt.x, y2R = k2R.pt.y;
                            Eigen::Vector3d P2;
                            P2(0) = -m / 2.0 + x2L + 0.5;
                            P2(1) = -n / 2.0 + y2L + 0.5;
                            P2(2) = -focusDist * baseline / (x2L - x2R);
                            P2(0) = -P2(0) * P2(2) / focusDist;
                            P2(1) = P2(1) * P2(2) / focusDist;

                            if (std::abs(y1L - y1R) < 0.1 && std::abs(y2L - y2R) < 0.1)
                                corPoints.push_back({ P1, P2 });
                        }
                    }
                }
            }
        }

        if (corPoints.size() >= 10)
            break;

        if (minHessian > 50)
            minHessian /= 2;
        else if (ratioThresh < 1.0f)
            ratioThresh += 0.1f;
        else
            break;
    }

    if (corPoints.empty())
        throw std::runtime_error("Не найдено достаточно 3D-соответствий для вычисления матрицы.");

    auto pcl1 = std::make_shared<open3d::geometry::PointCloud>();
    auto pcl2 = std::make_shared<open3d::geometry::PointCloud>();
    std::vector<Eigen::Vector2i> corres;
    for (size_t i = 0; i < corPoints.size(); ++i)
    {
        pcl1->points_.push_back(corPoints[i].p1);
        pcl2->points_.push_back(corPoints[i].p2);
        corres.push_back(Eigen::Vector2i(static_cast<int>(i), static_cast<int>(i)));
    }

    pcl1->Transform(transformMatrix.inverse());
    open3d::pipelines::registration::TransformationEstimationPointToPoint transfEst;
    Eigen::Matrix4d result = transfEst.ComputeTransformation(*pcl1, *pcl2, corres);


    //DrawRegistrationResult(*pcl1, *pcl2, Eigen::Matrix4d::Identity(), true);
    //DrawRegistrationResult(*pcl1, *pcl2, result, true);

    return result;
}

std::vector<Eigen::Matrix4d> ReadModelMatrices(const std::string& filepath)
{
    std::ifstream infile(filepath);
    if (!infile.is_open())
        throw std::runtime_error("Не удалось открыть файл: " + filepath);

    int n;
    infile >> n;
    if (infile.fail())
        throw std::runtime_error("Ошибка чтения количества матриц.");

    std::vector<Eigen::Matrix4d> matrices(n);
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                infile >> matrices[k](i, j);

    infile.close();
    return matrices;
}

void BuildModelMatrices(
    const std::vector<int>& frameSequence,
    const std::string& basePath,
    const std::string& disPath,
    double focusDist,
    double baseline,
    const std::string& outputPath)
{
    int frameCount = static_cast<int>(frameSequence.size());
    if (frameCount < 2)
        throw std::runtime_error("Последовательность должна содержать минимум два кадра.");

    std::ofstream outfile(outputPath);
    if (!outfile.is_open())
        throw std::runtime_error("Не удалось открыть выходной файл: " + outputPath);

    outfile << (frameCount - 1) << "\n";

    cv::Mat img1L = GetImage(basePath, frameSequence[0], 0);
    cv::Mat img1R = GetImage(basePath, frameSequence[0], 1);
    if (img1L.empty() || img1R.empty())
        throw std::runtime_error("Не удалось загрузить изображения для кадра " + std::to_string(frameSequence[0]));

    View view1;
    view1.pic = img1L;
    view1.pic2 = img1R;

    for (int i = 0; i < frameCount - 1; ++i)
    {
        auto start = std::chrono::system_clock::now();

        cv::Mat img2L = GetImage(basePath, frameSequence[i + 1], 0);
        cv::Mat img2R = GetImage(basePath, frameSequence[i + 1], 1);
        if (img2L.empty() || img2R.empty())
            throw std::runtime_error("Не удалось загрузить изображения для кадра " + std::to_string(frameSequence[i + 1]));

        View view2;
        view2.pic = img2L;
        view2.pic2 = img2R;

        Eigen::Matrix4d transform = GetMatrix4Way(view1, view2, Eigen::Matrix4d::Identity(), focusDist, baseline);
        outfile << transform << "\n";

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cerr << "Время для матрицы " << i << ": " << elapsed.count() << " с\n";

        view1 = view2;
    }

    outfile.close();
}

open3d::pipelines::registration::RegistrationResult RefineRegistration(
    const open3d::geometry::PointCloud& source,
    const open3d::geometry::PointCloud& target,
    std::shared_ptr<open3d::pipelines::registration::Feature> source_fpfh,
    std::shared_ptr<open3d::pipelines::registration::Feature> target_fpfh,
    float voxel_size,
    const open3d::pipelines::registration::RegistrationResult& resultRansac)
{
    float distanceThreshold = voxel_size * 0.4f;
    open3d::pipelines::registration::RegistrationResult result =
        open3d::pipelines::registration::RegistrationICP(
            source, target, distanceThreshold, resultRansac.transformation_,
            open3d::pipelines::registration::TransformationEstimationPointToPlane());
    return result;
}