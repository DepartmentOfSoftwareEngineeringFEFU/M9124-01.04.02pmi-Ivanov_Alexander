#include "load_view.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

static std::string FormatFrameIndex(int frameIndex)
{
    if (frameIndex < 10)
        return "00" + std::to_string(frameIndex);
    else if (frameIndex < 100)
        return "0" + std::to_string(frameIndex);
    else
        return std::to_string(frameIndex);
}

Eigen::Matrix4f GetFirstModelMatrix(const std::string& basePath, int frameIndex)
{
    std::string frameStr = FormatFrameIndex(frameIndex);
    std::string path = basePath + "/frame_00" + frameStr + "_0.info"; // предположим, что в пути уже есть разделитель

    std::ifstream infofile(path.c_str(), std::ios::in);
    if (!infofile.is_open())
        throw std::runtime_error("Не удалось открыть файл: " + path);

    Eigen::Matrix4f mModel_i, mModel;
    char temp[255];
    infofile.getline(temp, 255);

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            infofile >> mModel_i(i, j);

    infofile.getline(temp, 255);
    infofile.getline(temp, 255);

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            infofile >> mModel(i, j);

    infofile.close();
    return mModel;
}

cv::Mat GetImage(const std::string& basePath, int frameIndex, int stereoIndex)
{
    std::string frameStr = FormatFrameIndex(frameIndex);
    std::string path = basePath + "/frame_00" + frameStr + "_" + std::to_string(stereoIndex) + ".bmp";
    cv::Mat img = cv::imread(path);
    return img;
}

bool LoadViewSimple(
    const std::string& basePath,
    int frameIndex,
    bool MSK,
    const cv::Mat& img1d,
    double focusDist,
    double camDist,
    View& outView
)
{
    outView.name = std::to_string(frameIndex);
    outView.number = frameIndex;

    // Загрузка матриц модели и проекции из .info
    std::string frameStr = FormatFrameIndex(frameIndex);
    std::string infoPath = basePath + "/frame_00" + frameStr + "_0.info";
    std::ifstream infofile(infoPath.c_str(), std::ios::in);

    if (!infofile.is_open())
    {
        std::cerr << "Ошибка: не удалось открыть файл " << infoPath << std::endl;
        return false;
    }

    Eigen::Matrix4f mModel_i, mModel, mUnProj_i, mUnProj;
    char temp[255];
    infofile.getline(temp, 255);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            infofile >> mModel_i(i, j);

    infofile.getline(temp, 255);
    infofile.getline(temp, 255);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            infofile >> mModel(i, j);

    infofile.getline(temp, 255);
    infofile.getline(temp, 255);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            infofile >> mUnProj_i(i, j);

    infofile.getline(temp, 255);
    infofile.getline(temp, 255);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            infofile >> mUnProj(i, j);

    infofile.close();

    outView.model = mModel;
    outView.model_i = mModel_i;
    outView.unProj = mUnProj;
    outView.unProj_i = mUnProj_i;

    // Загрузка изображений
    outView.pic = GetImage(basePath, frameIndex, 0);
    outView.pic2 = GetImage(basePath, frameIndex, 1);

    if (outView.pic.empty())
    {
        std::cerr << "Ошибка: не удалось загрузить изображение для кадра " << frameIndex << std::endl;
        return false;
    }

    outView.cloud = std::make_shared<open3d::geometry::PointCloud>();

    int sizeX = outView.pic.cols;
    int sizeY = outView.pic.rows;
    outView.picSize = std::make_pair(sizeY, sizeX);

    Eigen::Matrix4f mModelOrig;
    /*try
    {
        mModelOrig = GetFirstModelMatrix(basePath, 0);
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "Ошибка при загрузке матрицы модели для кадра 0: " << e.what() << std::endl;
        return false;
    }*/

    // Заполнение облака точек
    for (int y = 0; y < sizeY; ++y)
    {
        for (int x = 0; x < sizeX; ++x)
        {
            Eigen::Vector4f P(-sizeX / 2 + x + 0.5f, -sizeY / 2 + y + 0.5f, 0.0f, 1.0f);
            float depth = img1d.at<float>(y, x);
            if (depth > 0.1f)
                P(2) = -focusDist * camDist / depth;

            P(0) = -P(0) * P(2) / focusDist;
            P(1) = P(1) * P(2) / focusDist;

            Eigen::Vector4f P1 = P;

            //Eigen::Vector4f P2 = mModelOrig * P1;
            Eigen::Vector4f P2 = P1;
            P2 /= P2(3);
            outView.cloud->points_.push_back(Eigen::Vector3d(P2(0), P2(1), P2(2)));

            cv::Vec3b color = outView.pic.at<cv::Vec3b>(y, x);
            outView.cloud->colors_.push_back(Eigen::Vector3d(
                (double)color[2] / 255.0,
                (double)color[1] / 255.0,
                (double)color[0] / 255.0
            ));

            Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();
            outView.cloud->covariances_.push_back(cov);
            outView.cloud->normals_.push_back(Eigen::Vector3d(0, 0, 1));
        }
    }

    return true;
}