#include "remove_background.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <limits>

double VectorDistance(cv::Vec3b v1, cv::Vec3b v2)
{
    double d0 = static_cast<double>(v1[0]) - v2[0];
    double d1 = static_cast<double>(v1[1]) - v2[1];
    double d2 = static_cast<double>(v1[2]) - v2[2];
    return std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
}

void ConsumeRegion(
    int y,
    int x,
    std::vector<std::vector<int>>& mask,
    cv::Mat& img,
    int depth,
    int consumeThreshold)
{
    if (depth > 250) return;
    for (int i = y - 2; i <= y + 2; ++i)
    {
        for (int j = x - 2; j <= x + 2; ++j)
        {
            if (i >= 0 && j >= 0 && i < img.rows && j < img.cols && mask[i][j] < 1)
            {
                if (VectorDistance(img.at<cv::Vec3b>(y, x), img.at<cv::Vec3b>(i, j)) < consumeThreshold)
                {
                    mask[i][j] = 2;
                    ConsumeRegion(i, j, mask, img, depth + 1, consumeThreshold);
                }
            }
        }
    }
}

void ConsumeRegionHSV(
    int y,
    int x,
    std::vector<std::vector<int>>& mask,
    cv::Mat& img,
    int depth,
    int consumeHue,
    int consumeSaturation,
    int consumeValue)
{
    if (depth > 250) return;
    for (int i = y - 2; i <= y + 2; ++i)
    {
        for (int j = x - 2; j <= x + 2; ++j)
        {
            if (i >= 0 && j >= 0 && i < img.rows && j < img.cols && mask[i][j] < 1)
            {
                cv::Vec3b cur = img.at<cv::Vec3b>(y, x);
                cv::Vec3b neigh = img.at<cv::Vec3b>(i, j);
                int hDif = std::abs(neigh[0] - cur[0]);
                int sDif = std::abs(neigh[1] - cur[1]);
                int vDif = std::abs(neigh[2] - cur[2]);
                if (hDif < consumeHue && sDif < consumeSaturation && vDif < consumeValue)
                {
                    mask[i][j] = 2;
                    ConsumeRegionHSV(i, j, mask, img, depth + 1, consumeHue, consumeSaturation, consumeValue);
                }
            }
        }
    }
}

std::vector<std::vector<int>> RemoveBackgroundWithDisparity(
    cv::Mat& img,
    const cv::Mat& dis,
    int disBottomBound,
    int disFloorRange,
    int consumeThreshold,
    cv::Vec3b bgColor)
{
    if (img.empty())
        throw std::runtime_error("Изображение пустое.");
    if (dis.empty())
        throw std::runtime_error("Карта диспаратности пустая.");
    if (img.size() != dis.size())
        throw std::runtime_error("Размеры изображения и карты диспаратности не совпадают.");

    int n = img.rows, m = img.cols;
    std::vector<std::vector<int>> mask(n, std::vector<int>(m, 0));

    int disMin = std::numeric_limits<int>::max();
    const int border = 50;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if (dis.at<float>(i, j) < disBottomBound)
                mask[i][j] = -1;
        }
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = border; j < m - border; ++j)
        {
            if (mask[i][j] != -1 && dis.at<float>(i, j) < disMin)
                disMin = static_cast<int>(dis.at<float>(i, j));
        }
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if (mask[i][j] != -1 && std::abs(dis.at<float>(i, j) - disMin) < disFloorRange)
            {
                double dist = cv::norm(img.at<cv::Vec3b>(i, j), bgColor, cv::NORM_L2);
                if (dist < 15.0)
                    mask[i][j] = 1;
            }
        }
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if (mask[i][j] >= 1)
                ConsumeRegion(i, j, mask, img, 0, consumeThreshold);
        }
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            img.at<cv::Vec3b>(i, j) = cv::Vec3b(100, 100, 100);
            if (dis.at<float>(i, j) < 1.0f / 100.0f)
                mask[i][j] = -1;

            if (mask[i][j] == 1)
                img.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            else if (mask[i][j] == 2)
                img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 125); 
            else if (mask[i][j] == -1)
                img.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);  
        }
    }

    return mask;
}

std::vector<std::vector<int>> RemoveBackgroundHSV(
    const cv::Mat& img,
    cv::Mat& out,
    int hue,
    int saturation,
    int value,
    int consumeHue,
    int consumeSaturation,
    int consumeValue,
    int saturationCutoff)
{
    if (img.empty())
        throw std::runtime_error("Изображение пустое.");

    cv::Mat imgHSV;
    cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);
    int n = img.rows, m = img.cols;
    std::vector<std::vector<int>> mask(n, std::vector<int>(m, 0));

    std::vector<int> colorStat(256 * 256 * 256, 0);
    int maxIdx = 0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            cv::Vec3b pix = imgHSV.at<cv::Vec3b>(i, j);
            int idx = 256 * 256 * pix[0] + 256 * pix[1] + pix[2];
            ++colorStat[idx];
            if (colorStat[idx] > colorStat[maxIdx])
                maxIdx = idx;
        }
    }
    int h = maxIdx / (256 * 256);
    int s = (maxIdx - 256 * 256 * h) / 256;
    int v = maxIdx - 256 * 256 * h - 256 * s;
    cv::Vec3b dominantColor(h, s, v);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            cv::Vec3b pix = imgHSV.at<cv::Vec3b>(i, j);
            int hDif = std::abs(pix[0] - dominantColor[0]);
            int sDif = std::abs(pix[1] - dominantColor[1]);
            int vDif = std::abs(pix[2] - dominantColor[2]);

            if (dominantColor[1] < saturationCutoff || pix[1] < saturationCutoff)
            {
                if (vDif < value)
                    mask[i][j] = 1;
            }
            else if (hDif < hue && sDif < saturation && vDif < value)
            {
                mask[i][j] = 1;
            }
        }
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if (mask[i][j] >= 1)
                ConsumeRegionHSV(i, j, mask, imgHSV, 0, consumeHue, consumeSaturation, consumeValue);
        }
    }

    cv::Mat result = cv::Mat::zeros(img.size(), img.type());
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 50);
            if (mask[i][j] == 1)
                result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            else if (mask[i][j] == 2)
                result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 200);
        }
    }
    result.copyTo(out);

    return mask;
}

std::vector<std::vector<int>> ColorSegmentationAdaptiveThreshold(
    const cv::Mat& imgGray,
    cv::Mat& out,
    int thresh1,
    int thresh2)
{
    if (imgGray.empty())
        throw std::runtime_error("Изображение пустое.");
    if (imgGray.channels() != 1)
        throw std::runtime_error("Ожидается одноканальное изображение.");

    cv::Mat th0, th1, th2;
    cv::adaptiveThreshold(imgGray, th0, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, 0);
    cv::adaptiveThreshold(imgGray, th1, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, thresh1 * 3);
    cv::adaptiveThreshold(imgGray, th2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, thresh2 * 3);

    cv::Mat sum16;
    cv::add(th0, th1, sum16, cv::noArray(), CV_16U);
    cv::add(sum16, th2, sum16, cv::noArray(), CV_16U);
    sum16.convertTo(out, CV_8U, 1.0 / 4.0);

    int n = imgGray.rows, m = imgGray.cols;
    std::vector<std::vector<int>> mask(n, std::vector<int>(m, 0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            mask[i][j] = static_cast<int>(sum16.at<ushort>(i, j) / 255);

    return mask;
}

int ColorSegmentationKMeans(const cv::Mat& img, cv::Mat& out, int K)
{
    if (img.empty())
        throw std::runtime_error("Изображение пустое.");

    cv::Mat imgHSV;
    cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

    int n = img.rows, m = img.cols;
    cv::Mat pixelArray = cv::Mat::zeros(n * m, 5, CV_32F);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            cv::Vec3b pix = imgHSV.at<cv::Vec3b>(i, j);
            float row = static_cast<float>(i) / n;
            float col = static_cast<float>(j) / m;
            pixelArray.at<float>(i * m + j, 0) = row;
            pixelArray.at<float>(i * m + j, 1) = col;
            pixelArray.at<float>(i * m + j, 2) = pix[0] / 255.0f;
            pixelArray.at<float>(i * m + j, 3) = pix[1] / 255.0f;
            pixelArray.at<float>(i * m + j, 4) = pix[2] / 255.0f;
        }
    }

    cv::Mat bestLabels, centers;
    cv::kmeans(pixelArray, K, bestLabels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0),
        3, cv::KMEANS_PP_CENTERS, centers);

    std::vector<int> colors(K);
    for (int i = 0; i < K; ++i)
        colors[i] = 255 / (i + 1);

    cv::Mat clustered = cv::Mat::zeros(n, m, CV_32F);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            int label = bestLabels.at<int>(i * m + j, 0);
            clustered.at<float>(i, j) = static_cast<float>(colors[label]);
        }
    }
    clustered.convertTo(out, CV_8U);
    out.copyTo(out);
    return 0;
}