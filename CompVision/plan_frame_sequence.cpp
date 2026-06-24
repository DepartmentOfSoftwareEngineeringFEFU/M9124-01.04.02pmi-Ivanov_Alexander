#include "plan_frame_sequence.h"
#include "load_view.h"
#include "model_matrix.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <iostream>

std::vector<int> PlanFrameSequence(const std::string& basePath,
    int firstIndex,
    int lastIndex,
    int maxStep)
{
    if (firstIndex >= lastIndex)
        throw std::runtime_error("Начальный индекс должен быть меньше конечного.");
    if (maxStep <= 0)
        throw std::runtime_error("Максимальный шаг должен быть положительным.");

    std::vector<int> sparseSeq;
    sparseSeq.push_back(firstIndex);

    int left = firstIndex;
    int right = firstIndex + 1;

    std::cout << "Начало планирования последовательности." << std::endl;
    while (left < lastIndex)
    {
        cv::Mat img1 = GetImage(basePath, left, 0);
        if (img1.empty())
            throw std::runtime_error("Не удалось загрузить изображение для кадра " + std::to_string(left));

        cv::Mat img2 = GetImage(basePath, right, 0);
        if (img2.empty())
            throw std::runtime_error("Не удалось загрузить изображение для кадра " + std::to_string(right));

        while (right < lastIndex &&
            (right - left) < maxStep &&
            CountCorPointsMatches(img1, img2) > 25)
        {
            ++right;
            img2 = GetImage(basePath, right, 0);
            if (img2.empty())
                throw std::runtime_error("Не удалось загрузить изображение для кадра " + std::to_string(right));
        }

        if (left == right)
            ++right; 

        std::cout << "Добавлен кадр: " << right << std::endl;
        sparseSeq.push_back(right);
        left = right;
    }

    if (sparseSeq.back() != lastIndex)
        sparseSeq.push_back(lastIndex);

    return sparseSeq;
}

std::vector<int> ComputeReconstructionSequence(const std::vector<int>& sparseSeq,
    const std::vector<int>& frameSeq)
{
    if (sparseSeq.empty() || frameSeq.empty())
        return {};

    std::set<int> resultSet;

    for (int f : frameSeq)
    {
        auto it = std::upper_bound(sparseSeq.begin(), sparseSeq.end(), f);
        if (it != sparseSeq.end() && it != sparseSeq.begin())
        {
            int j = static_cast<int>(it - sparseSeq.begin()) - 1;
            if (it != sparseSeq.end() && f < *it) 
            {
                resultSet.insert(sparseSeq[j]);
                resultSet.insert(*it);
            }
        }
    }

    return std::vector<int>(resultSet.begin(), resultSet.end());
}

std::vector<int> ReadFrameSequence(const std::string& filepath)
{
    std::ifstream infile(filepath);
    if (!infile.is_open())
        throw std::runtime_error("Не удалось открыть файл: " + filepath);

    int n;
    if (!(infile >> n))
        throw std::runtime_error("Ошибка чтения количества кадров из файла: " + filepath);

    std::vector<int> frames(n);
    for (int i = 0; i < n; ++i)
    {
        if (!(infile >> frames[i]))
            throw std::runtime_error("Ошибка чтения кадра " + std::to_string(i) + " из файла: " + filepath);
    }

    infile.close();
    return frames;
}