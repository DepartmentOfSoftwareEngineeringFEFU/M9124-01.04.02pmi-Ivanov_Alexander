#include "disparity_map.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <cmath>

static const int THREAD_COUNT = 12;

bool WriteMatBinary(const std::string& filename, const cv::Mat& output)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open())
    {
        std::cerr << "Ошибка: не удалось открыть файл для записи: " << filename << std::endl;
        return false;
    }

    if (output.empty())
    {
        int zero = 0;
        ofs.write(reinterpret_cast<const char*>(&zero), sizeof(int));
        return true;
    }

    int rows = output.rows;
    int cols = output.cols;
    int type = output.type();
    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&type), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(output.data), output.elemSize() * output.total());

    return true;
}

bool ReadMatBinary(const std::string& filename, cv::Mat& output)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open())
    {
        std::cerr << "Ошибка: не удалось открыть файл для чтения: " << filename << std::endl;
        return false;
    }

    int rows, cols, type;
    ifs.read(reinterpret_cast<char*>(&rows), sizeof(int));
    if (rows == 0)
    {
        output.release();
        return true;
    }
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&type), sizeof(int));

    output.release();
    output.create(rows, cols, type);
    ifs.read(reinterpret_cast<char*>(output.data), output.elemSize() * output.total());

    return true;
}

void CalculateDisparityForSegment(
    cv::Mat img1g,
    cv::Mat img2g,
    std::vector<Rect> windows,
    cv::Mat& bestShift,
    bool reverse,
    int l,
    int r
)
{
    int n = img1g.rows;
    int m = img1g.cols;

    for (int i = l; i < r; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            double bestWindowScore = 1e9;
            int bestShiftValue = 0;

            int shiftMin = 20;
            int shiftMax = std::min(j, 70);
            if (reverse)
            {
                shiftMin = std::max(-(m - j - 1), -70);
                shiftMax = -20;
            }

            for (const Rect& window : windows)
            {
                double windowBestScore = 1e9;
                int shiftForPoint = 0;

                for (int shift = shiftMin; shift <= shiftMax; ++shift)
                {
                    long long count = 0;
                    double windowScore = 0.0;
                    int windowVol = (window.x2 - window.x1) * (window.y2 - window.y1);

                    for (int y = window.y1; y <= window.y2; ++y)
                    {
                        int yy = i + y;
                        if (yy < 0 || yy >= n) continue;

                        for (int x = window.x1; x <= window.x2; ++x)
                        {
                            int xx = j + x;
                            int xxShift = xx - shift;
                            if (xx < 0 || xx >= m || xxShift < 0 || xxShift >= m) continue;

                            ++count;
                            uchar a = img1g.at<uchar>(yy, xx);
                            uchar b = img2g.at<uchar>(yy, xxShift);
                            windowScore += (a - b) * (a - b);
                        }
                    }

                    if (count >= (3 * windowVol) / 4)
                    {
                        windowScore += std::abs(shift);
                        windowScore /= (windowVol * windowVol);
                        if (windowBestScore > windowScore)
                        {
                            windowBestScore = windowScore;
                            shiftForPoint = shift;
                        }
                    }
                }

                if (bestWindowScore > windowBestScore)
                {
                    bestWindowScore = windowBestScore;
                    bestShiftValue = shiftForPoint;
                }
            }

            bestShift.at<float>(i, j) = static_cast<float>(bestShiftValue);
        }
    }
}

bool BuildDisparityMap(
    cv::Mat img1g,
    cv::Mat img2g,
    cv::Mat& bestShift,
    bool reverse,
    const std::string& filepath
)
{
    if (img1g.empty() || img2g.empty())
    {
        std::cerr << "Ошибка: одно из изображений пустое." << std::endl;
        return false;
    }

    if (img1g.rows != img2g.rows || img1g.cols != img2g.cols)
    {
        std::cerr << "Ошибка: размеры изображений не совпадают." << std::endl;
        return false;
    }

    int n = img1g.rows;
    int m = img1g.cols;

    if (bestShift.empty() || bestShift.rows != n || bestShift.cols != m || bestShift.type() != CV_32F)
    {
        bestShift = cv::Mat::zeros(n, m, CV_32F);
    }

    int w1h = 1, w1w = 5;
    int w2h = 5, w2w = 1;
    std::vector<Rect> windows;

    for (int i = 0; i <= 2; i += 2)
    {
        windows.push_back({ (i - 2) * w1w, i * w1w, (1 - 2) * w1h, 1 * w1h, (2 * w1h + 1) * (2 * w1w + 1) });
        windows.push_back({ (1 - 2) * w2w, 1 * w2w, (i - 2) * w2h, i * w2h, (2 * w2h + 1) * (2 * w2w + 1) });
    }

    int threadAmount = THREAD_COUNT;
    std::vector<std::thread> threads;
    for (int t = 0; t < threadAmount; ++t)
    {
        int l = t * (n / threadAmount);
        int r = (t + 1) * (n / threadAmount);
        if (t == threadAmount - 1) r = n;
        threads.emplace_back(CalculateDisparityForSegment, img1g, img2g, windows, std::ref(bestShift), reverse, l, r);
    }

    for (auto& th : threads) th.join();

    if (!WriteMatBinary(filepath, bestShift))
    {
        std::cerr << "Ошибка: не удалось сохранить карту диспаратности в файл." << std::endl;
        return false;
    }

    return true;
}

cv::Mat ReadDisparityMap(int rows, int cols, const std::string& filepath)
{
    cv::Mat disparity;
    if (!ReadMatBinary(filepath, disparity))
    {
        std::cerr << "Ошибка: не удалось прочитать карту диспаратности из файла." << std::endl;
        return cv::Mat();
    }

    if (disparity.rows != rows || disparity.cols != cols)
    {
        std::cerr << "Ошибка: размеры прочитанной карты не соответствуют ожидаемым." << std::endl;
        return cv::Mat();
    }

    return disparity;
}