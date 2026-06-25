#include "combined_view.h"
#include "load_view.h"
#include "get_view.h"
#include "disparity_map.h"
#include "remove_background.h"
#include "model_matrix.h"
#include "plan_frame_sequence.h"
#include <unordered_set>
#include <iostream>
#include <stdexcept>

std::vector<size_t> GetBackgroundIndices(const View& view,
    const cv::Mat& img,
    const cv::Mat& dis,
    int rows,
    int cols)
{
    (void)view; // не используется
    std::vector<std::vector<int>> mask = RemoveBackgroundWithDisparity(
        const_cast<cv::Mat&>(img), dis, 20, 10, 3, cv::Vec3b(109, 153, 148));
    std::vector<size_t> indices;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (mask[i][j] != 0)
                indices.push_back(static_cast<size_t>(i) * cols + j);
    return indices;
}

int CombinedView(const std::vector<int>& sparseSeq,
    const std::vector<int>& reconstructionSeq,
    const std::vector<Eigen::Matrix4d>& transformMatrices,
    const std::string& disPath,
    const std::string& basePath,
    double focusDist,
    double baseline,
    const std::string& outputObjPath)
{
    if (sparseSeq.size() < 2)
        throw std::runtime_error(u8"Разреженная последовательность должна содержать минимум 2 кадра.");

    if (transformMatrices.size() != sparseSeq.size() - 1)
        throw std::runtime_error(u8"Количество матриц преобразований не соответствует sparseSeq.");

    std::unordered_set<int> reconstSet(reconstructionSeq.begin(), reconstructionSeq.end());

    int refIndex = -1;
    for (size_t i = 0; i < sparseSeq.size(); ++i)
    {
        if (reconstSet.count(sparseSeq[i]))
        {
            refIndex = static_cast<int>(i);
            break;
        }
    }
    if (refIndex == -1)
        throw std::runtime_error(u8"Ни один кадр из разреженной последовательности не входит в reconstructionSeq.");

    int refFrame = sparseSeq[refIndex];

    View refView = GetView(refFrame, true, disPath, basePath, focusDist, baseline);
    if (!refView.cloud || refView.cloud->points_.empty())
        throw std::runtime_error(u8"Облако эталонного кадра пустое.");

    cv::Mat refImg = GetImage(basePath, refFrame, 0);
    if (refImg.empty())
        throw std::runtime_error(u8"Не удалось загрузить изображение для кадра " + std::to_string(refFrame));
    int rows = refImg.rows, cols = refImg.cols;
    cv::Mat refDis = ReadDisparityMap(rows, cols, disPath + std::to_string(refFrame) + "_0.txt");
    if (refDis.empty())
        throw std::runtime_error(u8"Не удалось прочитать карту диспаратности для кадра " + std::to_string(refFrame));

    std::vector<size_t> bgIdx = GetBackgroundIndices(refView, refImg, refDis, rows, cols);
    *refView.cloud = *refView.cloud->SelectByIndex(bgIdx, true);

    auto combinedCloud = std::make_shared<open3d::geometry::PointCloud>(*refView.cloud);

    Eigen::Matrix4d accum = Eigen::Matrix4d::Identity();

    for (size_t i = refIndex + 1; i < sparseSeq.size(); ++i)
    {
        accum = accum * transformMatrices[i - 1];
        int frame = sparseSeq[i];

        if (!reconstSet.count(frame))
        {
            std::cout << u8"Пропускаем кадр " << frame << u8" (не входит в набор)" << std::endl;
            continue;
        }

        View view = GetView(frame, true, disPath, basePath, focusDist, baseline);
        if (!view.cloud || view.cloud->points_.empty())
            throw std::runtime_error(u8"Облако кадра " + std::to_string(frame) + u8" пустое.");

        cv::Mat img = GetImage(basePath, frame, 0);
        if (img.empty())
            throw std::runtime_error(u8"Не удалось загрузить изображение для кадра " + std::to_string(frame));
        cv::Mat dis = ReadDisparityMap(rows, cols, disPath + std::to_string(frame) + "_0.txt");
        if (dis.empty())
            throw std::runtime_error(u8"Не удалось прочитать карту диспаратности для кадра " + std::to_string(frame));

        std::vector<size_t> bg = GetBackgroundIndices(view, img, dis, rows, cols);
        *view.cloud = *view.cloud->SelectByIndex(bg, true);

        //open3d::visualization::DrawGeometries({ view.cloud, refView.cloud }, "Open3D", 640, 480, 50, 50);
        //cv::waitKey(0);
        view.cloud->Transform(accum.inverse());
        //open3d::visualization::DrawGeometries({ view.cloud, refView.cloud }, "Open3D", 640, 480, 50, 50);
        //cv::waitKey(0);

        open3d::geometry::KDTreeFlann kdtree(*combinedCloud);
        std::vector<size_t> pointsToDelete;
        for (size_t j = 0; j < view.cloud->points_.size(); ++j)
        {
            Eigen::Vector3d p = view.cloud->points_[j];
            std::vector<int> found;
            std::vector<double> dist;
            kdtree.SearchRadius(p, 0.01, found, dist);
            if (!found.empty())
                pointsToDelete.push_back(j);
        }
        *view.cloud = *view.cloud->SelectByIndex(pointsToDelete, true);

        *combinedCloud += *view.cloud;
        std::cout << u8"Добавлен кадр " << frame << std::endl;
    }

    auto result = combinedCloud->RemoveStatisticalOutliers(8, 0.8, false);
    combinedCloud = std::get<std::shared_ptr<open3d::geometry::PointCloud>>(result);

    std::shared_ptr<open3d::geometry::TriangleMesh> mesh;
    std::vector<double> densities;
    std::tie(mesh, densities) = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(*combinedCloud, 11);
    if (!mesh)
        throw std::runtime_error(u8"Ошибка при реконструкции поверхности методом Пуассона.");

    std::vector<size_t> verticesToDelete;
    for (size_t i = 0; i < densities.size(); ++i)
        if (densities[i] < 10.0)
            verticesToDelete.push_back(i);
    mesh->RemoveVerticesByIndex(verticesToDelete);
    mesh->ComputeVertexNormals();

    open3d::visualization::DrawGeometries({ mesh }, "Result", 640, 480, 50, 50);
    cv::waitKey(0);

    // Сохранение OBJ
    if (!open3d::io::WriteTriangleMeshToOBJ(outputObjPath, *mesh, false, false, true, true, true, true))
        throw std::runtime_error(u8"Не удалось сохранить модель в " + outputObjPath);

    return 0;
}

int reconstructFromFrameSequence(const std::vector<int>& frameSequence,
    int totalFrames,
    int maxStep,
    std::string basePath,
    std::string cameraParamsPath)
{
    try
    {
        std::string disPathRaw = basePath + "/dmaps/"; 
        std::string disPathFiltered = basePath + "/dmapsF/";
        if (!std::filesystem::exists(disPathRaw))
            std::filesystem::create_directory(disPathRaw);
        if (!std::filesystem::exists(disPathFiltered))
            std::filesystem::create_directory(disPathFiltered);
        std::string outputObjPath = basePath + "/out.obj";
        std::string matFilePath = basePath + "/mat.txt";
        // 1. Планирование разреженной последовательности
        //std::vector<int> sparseSeq = { 57, 59 };
        std::vector<int> sparseSeq = PlanFrameSequence(basePath, 0, totalFrames - 1, maxStep);
        std::cout << u8"Разреженная последовательность (" << sparseSeq.size() << "): ";
        for (int s : sparseSeq) std::cout << s << " ";
        std::cout << std::endl;

        // 2. Вычисление последовательности для реконструкции (опорные пары)
        //std::vector<int> reconstructionSeq = { 57, 59 };
        std::vector<int> reconstructionSeq = ComputeReconstructionSequence(sparseSeq, frameSequence);
        std::cout << u8"Последовательность реконструкции (" << reconstructionSeq.size() << "): ";
        for (int r : reconstructionSeq) std::cout << r << " ";
        std::cout << std::endl;

        // 3. Получение параметров камер
        std::vector<double> camParams = GetCameraParams(cameraParamsPath);
        /*double focusDist = 642.666603034;
        double baseline = 0.3;*/
        double focusDist = camParams[0];
        double baseline = camParams[1];

        // 4. Построение матриц преобразований для разреженной последовательности
        BuildModelMatrices(sparseSeq, basePath, disPathFiltered, focusDist, baseline, matFilePath);

        // 5. Чтение матриц
        std::vector<Eigen::Matrix4d> transformMatrices = ReadModelMatrices(matFilePath);
        if (transformMatrices.size() != sparseSeq.size() - 1)
            throw std::runtime_error(u8"Количество прочитанных матриц не совпадает с размером sparseSeq.");

        // 6. Вызов объединения облаков
        return CombinedView(sparseSeq, reconstructionSeq, transformMatrices,
            disPathFiltered, basePath, focusDist, baseline,
            outputObjPath);
    }
    catch (const std::exception& e)
    {
        std::cerr << u8"Ошибка в RunReconstruction: " << e.what() << std::endl;
        return -1;
    }
}