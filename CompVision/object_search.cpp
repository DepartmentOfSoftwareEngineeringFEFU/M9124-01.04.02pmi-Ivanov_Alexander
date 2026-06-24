#include "object_search.h"

#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <regex>

namespace fs = std::filesystem;

void colorSegmentationThresh(const cv::Mat& img_gray, int thresh1, int thresh2,
    cv::Mat& out, std::vector<std::vector<int>>& mask) {
    cv::Mat img_thresh0, img_thresh1, img_thresh2;
    cv::adaptiveThreshold(img_gray, img_thresh0, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, 0);
    cv::adaptiveThreshold(img_gray, img_thresh1, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, thresh1 * 3);
    cv::adaptiveThreshold(img_gray, img_thresh2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, thresh2 * 3);

    cv::Mat sum16;
    cv::add(img_thresh2, img_thresh1, sum16, cv::noArray(), CV_16U);
    out = sum16 / 4.0;
    out.convertTo(out, CV_8U);

    cv::Mat sum_div = sum16 / 255;
    sum_div.convertTo(sum_div, CV_32S);
    int rows = sum_div.rows;
    int cols = sum_div.cols;
    mask.resize(rows, std::vector<int>(cols));
    for (int y = 0; y < rows; ++y) {
        int* row_ptr = sum_div.ptr<int>(y);
        for (int x = 0; x < cols; ++x) {
            mask[y][x] = row_ptr[x];
        }
    }
}

cv::Mat crop(const cv::Mat& img, int x1, int x2, int y1, int y2) {
    return img(cv::Range(y1, y2), cv::Range(x1, x2)).clone();
}

std::pair<cv::Point2f, cv::Point2f> getCrossPoints(const cv::Point2f& p1, const cv::Point2f& p2,
    float cross_len, bool round_result = false) {
    cv::Point2f pm = (p1 + p2) * 0.5f;
    cv::Point2f diff = p1 - pm;
    float norm_val = cv::norm(diff);
    if (norm_val == 0.0f) {
        return { pm, pm };
    }
    cv::Point2f perp(-diff.y, diff.x);
    perp *= (cross_len / norm_val);
    cv::Point2f c1 = pm - perp;
    cv::Point2f c2 = pm + perp;
    if (round_result) {
        c1.x = std::round(c1.x);
        c1.y = std::round(c1.y);
        c2.x = std::round(c2.x);
        c2.y = std::round(c2.y);
    }
    return { c1, c2 };
}

void drawLineWithCrossingEdge(const cv::Vec4i& line, cv::Mat& img, int cross_len) {
    int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
    cv::Point2f p1(x1, y1), p2(x2, y2), pm = (p1 + p2) * 0.5f;
    cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    if (cross_len > 0) {
        auto drawCross = [&](const cv::Point2f& a, const cv::Point2f& b) {
            auto [c1, c2] = getCrossPoints(a, b, static_cast<float>(cross_len), true);
            cv::line(img, cv::Point(static_cast<int>(c1.x), static_cast<int>(c1.y)),
                cv::Point(static_cast<int>(c2.x), static_cast<int>(c2.y)),
                cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            };
        drawCross(p1, p2);
        drawCross(p1, pm);
        drawCross(pm, p2);
    }
}

int checkPointPairDifference(const cv::Point2f& p1, const cv::Point2f& p2,
    const std::vector<std::vector<int>>& segments) {
    int rows = static_cast<int>(segments.size());
    if (rows == 0) return 0;
    int cols = static_cast<int>(segments[0].size());
    int x1 = static_cast<int>(std::round(p1.x));
    int y1 = static_cast<int>(std::round(p1.y));
    int x2 = static_cast<int>(std::round(p2.x));
    int y2 = static_cast<int>(std::round(p2.y));
    if (x1 >= 0 && x1 < cols && y1 >= 0 && y1 < rows &&
        x2 >= 0 && x2 < cols && y2 >= 0 && y2 < rows) {
        int v1 = segments[y1][x1];
        int v2 = segments[y2][x2];
        if (v1 != v2) {
            return v1 - v2;
        }
    }
    return 0;
}

std::vector<std::pair<cv::Point2f, cv::Point2f>> gen_segments(const cv::Point2f& p1, const cv::Point2f& p2, int depth) {
    std::vector<std::pair<cv::Point2f, cv::Point2f>> segs;
    if (depth < 0) return segs;
    segs.emplace_back(p1, p2);
    if (depth == 0) return segs;
    cv::Point2f mid = (p1 + p2) * 0.5f;
    auto left = gen_segments(p1, mid, depth - 1);
    auto right = gen_segments(mid, p2, depth - 1);
    segs.insert(segs.end(), left.begin(), left.end());
    segs.insert(segs.end(), right.begin(), right.end());
    return segs;
}

bool checkEdge(const cv::Vec4i& line, const std::vector<std::vector<int>>& segments_map,
    int cross_len, int split_depth = 2) {
    if (cross_len == -1) return true;
    cv::Point2f p1(line[0], line[1]), p2(line[2], line[3]);
    auto parts = gen_segments(p1, p2, split_depth);
    int diff_direction = 0;
    for (const auto& [start, end] : parts) {
        auto [c1, c2] = getCrossPoints(start, end, static_cast<float>(cross_len), true);
        int diff = checkPointPairDifference(c1, c2, segments_map);
        if (diff_direction == 0) diff_direction = diff;
        if (diff == 0 || diff * diff_direction <= 0) {
            return false;
        }
    }
    return true;
}

std::pair<std::vector<cv::Vec4i>, int> detectLines(const cv::Mat& canny_edges, const cv::Mat& img,
    const std::vector<std::vector<int>>& segment_map,
    int length, int cross_len, int split_depth = 2,
    bool visualization = false) {
    int thresh = 25;
    int gap = length / 5;
    std::vector<cv::Vec4i> lines_raw;
    cv::HoughLinesP(canny_edges, lines_raw, 1, CV_PI / 180, thresh, length, gap);

    std::vector<cv::Vec4i> lines_edge;
    if (visualization) {
        cv::Mat cdst, cdstP;
        cv::cvtColor(img, cdst, cv::COLOR_GRAY2BGR);
        cdstP = cdst.clone();
        for (const auto& l : lines_raw) {
            drawLineWithCrossingEdge(l, cdst, cross_len);
            if (checkEdge(l, segment_map, cross_len, split_depth)) {
                drawLineWithCrossingEdge(l, cdstP, cross_len);
                lines_edge.push_back(l);
            }
        }
        cv::imshow(u8"Линии Хафа", cdst);
        cv::imshow(u8"Обнаруженные линии (красным) - вероятностное преобразование Хафа", cdstP);
        cv::waitKey(0);
    }
    else {
        for (const auto& l : lines_raw) {
            if (checkEdge(l, segment_map, cross_len, split_depth)) {
                lines_edge.push_back(l);
            }
        }
    }
    return { lines_edge, static_cast<int>(lines_edge.size()) };
}


cv::Mat preprocessImage(const cv::Mat& img) {
    int h = img.rows;
    int w = img.cols;
    int new_w = static_cast<int>(256.0 * w / h);

    cv::Mat denoised;
    if (img.channels() == 3) {
        cv::fastNlMeansDenoisingColored(img, denoised, 10, 10, 7, 21);
    }
    else {
        cv::fastNlMeansDenoising(img, denoised, 10, 7, 21);
    }

    cv::Mat resized;
    cv::resize(denoised, resized, cv::Size(new_w, 256), 0, 0, cv::INTER_LANCZOS4);

    cv::Mat gray;
    if (resized.channels() == 3) {
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = resized.clone();
    }

    cv::Mat enhanced;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(6, 6));
    clahe->apply(gray, enhanced);

    cv::Mat normalized;
    enhanced.convertTo(normalized, CV_32F, 1.0 / 255.0);
    cv::Mat blurred;
    cv::GaussianBlur(normalized, blurred, cv::Size(0, 0), 40.0);
    normalized = normalized - blurred + 0.5;
    cv::normalize(normalized, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);

    return normalized;
}

ImageFeatures extractFeatures(const cv::Mat& img) {
    ImageFeatures f;
    CV_Assert(!img.empty());

    std::vector<cv::Mat> bgr;
    cv::split(img, bgr);
    cv::Mat mag2_sum = cv::Mat::zeros(img.size(), CV_32F);
    for (const auto& ch : bgr) {
        cv::Mat gx, gy;
        cv::Sobel(ch, gx, CV_32F, 1, 0);
        cv::Sobel(ch, gy, CV_32F, 0, 1);
        cv::Mat mag;
        cv::magnitude(gx, gy, mag);
        cv::accumulateSquare(mag, mag2_sum);
    }
    cv::Mat color_grad;
    cv::sqrt(mag2_sum, color_grad);
    cv::Scalar mean_val, std_val;
    cv::meanStdDev(color_grad, mean_val, std_val);
    f.color_grad_mean = mean_val[0];
    f.color_grad_std = std_val[0];

    double edge_thresh = 50.0;
    cv::Mat color_edges = (color_grad > edge_thresh);
    color_edges.convertTo(color_edges, CV_8U);
    int edge_pix = cv::countNonZero(color_edges);
    f.color_edge_density = static_cast<double>(edge_pix) / (img.rows * img.cols);

    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;
    int canny_thresh = 80;
    int kernel_size = 3;
    int rat = 3;
    cv::Mat edges;
    cv::Canny(gray, edges, canny_thresh, canny_thresh * rat, kernel_size);
    f.edge_density = static_cast<double>(cv::countNonZero(edges)) / (gray.rows * gray.cols);

    cv::Mat img_segmented;
    std::vector<std::vector<int>> segment_map;
    colorSegmentationThresh(gray, -5, 5, img_segmented, segment_map);

    const std::vector<std::pair<int, int>> line_scales = { {15,2},{30,4},{60,8},{120,16} };
#pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(line_scales.size()); ++idx) {
        int length = line_scales[idx].first;
        int cross_len = line_scales[idx].second;

        auto [lines_edge, count] = detectLines(edges, img_segmented, segment_map, length, cross_len, 2, false);
        *(&f.num_lines_edge1 + idx) = count;
        double total_len = 0.0;
        for (const auto& l : lines_edge) {
            total_len += std::hypot(l[2] - l[0], l[3] - l[1]);
        }
        *(&f.total_line_length_edge1 + idx) = total_len;

        int cross_len_half = cross_len / 2;
        auto [lines_edge_half, count_half] = detectLines(edges, img_segmented, segment_map, length, cross_len_half, 2, false);
        *(&f.num_lines_edge_half1 + idx) = count_half;
        double total_len_half = 0.0;
        for (const auto& l : lines_edge_half) {
            total_len_half += std::hypot(l[2] - l[0], l[3] - l[1]);
        }
        *(&f.total_line_length_edge_half1 + idx) = total_len_half;

        auto [lines_all, cnt_all] = detectLines(edges, img_segmented, segment_map, length, -1, 2, false);
        *(&f.num_lines1 + idx) = cnt_all;
        double total_len_all = 0.0;
        for (const auto& l : lines_all) {
            total_len_all += std::hypot(l[2] - l[0], l[3] - l[1]);
        }
        *(&f.total_line_length1 + idx) = total_len_all;
    }

    return f;
}

std::vector<double> featureVector(const ImageFeatures& f) {
    return {
        f.edge_density,
        f.num_lines1, f.num_lines2, f.num_lines3, f.num_lines4,
        f.total_line_length1, f.total_line_length2, f.total_line_length3, f.total_line_length4,
        f.num_lines_edge1, f.num_lines_edge2, f.num_lines_edge3, f.num_lines_edge4,
        f.total_line_length_edge1, f.total_line_length_edge2, f.total_line_length_edge3, f.total_line_length_edge4,
        f.num_lines_edge_half1, f.num_lines_edge_half2, f.num_lines_edge_half3, f.num_lines_edge_half4,
        f.total_line_length_edge_half1, f.total_line_length_edge_half2, f.total_line_length_edge_half3, f.total_line_length_edge_half4,
        f.color_grad_mean, f.color_grad_std, f.color_edge_density
    };
}

std::pair<std::vector<double>, std::vector<double>> computeMeanStd(const std::vector<std::vector<double>>& vectors) {
    if (vectors.empty()) return {};
    size_t dim = vectors[0].size();
    std::vector<double> mean(dim, 0.0), stddev(dim, 0.0);
    for (const auto& v : vectors) {
        for (size_t i = 0; i < dim; ++i) {
            mean[i] += v[i];
        }
    }
    for (size_t i = 0; i < dim; ++i) mean[i] /= vectors.size();

    for (const auto& v : vectors) {
        for (size_t i = 0; i < dim; ++i) {
            double diff = v[i] - mean[i];
            stddev[i] += diff * diff;
        }
    }
    for (size_t i = 0; i < dim; ++i) stddev[i] = std::sqrt(stddev[i] / vectors.size());
    return { mean, stddev };
}

double computeZScore(const std::vector<double>& v,
    const std::vector<double>& mean,
    const std::vector<double>& std,
    double eps) {
    double score = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        double z = (v[i] - mean[i]) / (std[i] + eps);
        score += z * z;
    }
    return score;
}

double computeThreshold(const std::vector<double>& bg_scores, double thresh_k, double percentile) {
    if (bg_scores.empty()) return 0.0;
    std::vector<double> sorted = bg_scores;
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>(percentile / 100.0 * sorted.size());
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return sorted[idx] * thresh_k;
}

static std::vector<fs::path> getImageFiles(const fs::path& folder) {
    std::vector<fs::path> files;
    if (!fs::exists(folder) || !fs::is_directory(folder))
        return files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tiff")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

void classifyStereoImages(const std::string& bg_folder,
    const std::string& stereo_folder,
    const std::string& output_file,
    double thresh_k,
    double percentile) {
    auto start_total = std::chrono::high_resolution_clock::now();

    std::cout << u8"Извлечение признаков из фоновых изображений..." << std::endl;
    std::vector<fs::path> bg_paths = getImageFiles(bg_folder);
    if (bg_paths.empty()) {
        std::cerr << u8"Фоновые изображения не найдены в " << bg_folder << std::endl;
        return;
    }

    std::vector<std::vector<double>> bg_vectors;
    bg_vectors.reserve(bg_paths.size());

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)bg_paths.size(); ++i) {
        cv::Mat img = cv::imread(bg_paths[i].string());
        if (img.empty()) continue;
        cv::Mat preprocessed = preprocessImage(img);
        ImageFeatures feats = extractFeatures(preprocessed);
        std::vector<double> vec = featureVector(feats);
#pragma omp critical
        {
            bg_vectors.push_back(std::move(vec));
        }
    }

    if (bg_vectors.empty()) {
        std::cerr << u8"Нет допустимых фоновых изображений." << std::endl;
        return;
    }

    auto [mean_raw, std_raw] = computeMeanStd(bg_vectors);
    const double eps = 1e-6;

    std::vector<std::vector<double>> bg_norm;
    bg_norm.reserve(bg_vectors.size());
    for (const auto& v : bg_vectors) {
        std::vector<double> norm(v.size());
        for (size_t j = 0; j < v.size(); ++j)
            norm[j] = std::abs(v[j] - mean_raw[j]) / (std_raw[j] + eps);
        bg_norm.push_back(norm);
    }
    auto [mean_norm, std_norm] = computeMeanStd(bg_norm);

    std::vector<double> bg_scores;
    bg_scores.reserve(bg_norm.size());
    for (const auto& v : bg_norm) {
        bg_scores.push_back(computeZScore(v, mean_norm, std_norm, eps));
    }
    double threshold = computeThreshold(bg_scores, thresh_k, percentile);
    std::cout << u8"Порог = " << threshold << std::endl;

    std::vector<fs::path> all_stereo = getImageFiles(stereo_folder);
    if (all_stereo.empty()) {
        std::cerr << u8"Стереоизображения не найдены в " << stereo_folder << std::endl;
        return;
    }

    std::vector<std::pair<int, fs::path>> left_images;
    std::regex pattern(R"(frame_(\d+)_0\.bmp)", std::regex::icase);
    for (const auto& p : all_stereo) {
        std::string fname = p.filename().string();
        std::smatch match;
        if (std::regex_match(fname, match, pattern)) {
            int frame = std::stoi(match[1].str());
            left_images.emplace_back(frame, p);
        }
    }

    if (left_images.empty()) {
        std::cerr << u8"Cтереоизображения (frame_*_0.bmp) не найдены в " << stereo_folder << std::endl;
        return;
    }

    std::cout << u8"Обработка " << left_images.size() << u8" стереоизображений." << std::endl;

    std::vector<int> object_frames;

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)left_images.size(); ++i) {
        const auto& [frame, path] = left_images[i];
        cv::Mat img = cv::imread(path.string());
        if (img.empty()) continue;

        cv::Mat preprocessed = preprocessImage(img);
        ImageFeatures feats = extractFeatures(preprocessed);
        std::vector<double> vec = featureVector(feats);

        std::vector<double> norm(vec.size());
        for (size_t j = 0; j < vec.size(); ++j)
            norm[j] = std::abs(vec[j] - mean_raw[j]) / (std_raw[j] + eps);

        double score = computeZScore(norm, mean_norm, std_norm, eps);
        if (score > threshold) {
#pragma omp critical
            {
                object_frames.push_back(frame);
            }
        }
    }

    std::sort(object_frames.begin(), object_frames.end());
    std::ofstream out(stereo_folder + "\\" + output_file);
    if (!out.is_open()) {
        std::cerr << u8"Не удалось открыть выходной файл: " << output_file << std::endl;
        return;
    }
    for (int f : object_frames) {
        out << f << "\n";
    }
    out.close();

    std::cout << u8"Найдено " << object_frames.size() << u8" стереоизображений, содержащих объекты." << std::endl;
    std::cout << u8"Последовательность записана в " << output_file << std::endl;

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_total - start_total;
    std::cout << u8"\nОбщее время обработки: " << std::fixed << std::setprecision(2) << elapsed.count() << u8" секунд" << std::endl;
}