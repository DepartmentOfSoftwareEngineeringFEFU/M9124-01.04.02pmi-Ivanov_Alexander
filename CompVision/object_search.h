#pragma once

#include <opencv2/opencv.hpp>   // cv::Mat, cv::Vec4i, etc.
#include <vector>
#include <string>
#include <utility>              // std::pair
#include <regex>  // add if not already included

// ----------------------------------------------------------------------
//  Feature struct (same layout as original)
// ----------------------------------------------------------------------
struct ImageFeatures {
    double edge_density = 0.0;
    double num_lines1 = 0.0, num_lines2 = 0.0, num_lines3 = 0.0, num_lines4 = 0.0;
    double total_line_length1 = 0.0, total_line_length2 = 0.0, total_line_length3 = 0.0, total_line_length4 = 0.0;
    double num_lines_edge1 = 0.0, num_lines_edge2 = 0.0, num_lines_edge3 = 0.0, num_lines_edge4 = 0.0;
    double total_line_length_edge1 = 0.0, total_line_length_edge2 = 0.0, total_line_length_edge3 = 0.0, total_line_length_edge4 = 0.0;
    double num_lines_edge_half1 = 0.0, num_lines_edge_half2 = 0.0, num_lines_edge_half3 = 0.0, num_lines_edge_half4 = 0.0;
    double total_line_length_edge_half1 = 0.0, total_line_length_edge_half2 = 0.0, total_line_length_edge_half3 = 0.0, total_line_length_edge_half4 = 0.0;
    double color_grad_mean = 0.0;
    double color_grad_std = 0.0;
    double color_edge_density = 0.0;
};

// ----------------------------------------------------------------------
//  Public image processing functions
// ----------------------------------------------------------------------
cv::Mat preprocessImage(const cv::Mat& img);
ImageFeatures extractFeatures(const cv::Mat& img);
std::vector<double> featureVector(const ImageFeatures& f);

// ----------------------------------------------------------------------
//  Z‑score anomaly detection
// ----------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>> computeMeanStd(
    const std::vector<std::vector<double>>& vectors);

double computeZScore(const std::vector<double>& v,
    const std::vector<double>& mean,
    const std::vector<double>& std,
    double eps = 1e-6);

double computeThreshold(const std::vector<double>& bg_scores,
    double thresh_k = 1.0,
    double percentile = 99.0);

// ----------------------------------------------------------------------
//  Main classification function (zscore only)
// ----------------------------------------------------------------------
void classifyStereoImages(const std::string& bg_folder,
    const std::string& stereo_folder,
    const std::string& output_file = "sequence.txt",
    double thresh_k = 1.0,
    double percentile = 99.0);