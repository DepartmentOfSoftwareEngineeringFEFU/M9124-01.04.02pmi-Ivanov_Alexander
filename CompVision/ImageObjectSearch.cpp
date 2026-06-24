Mat src, src_gray;
Mat dst, detected_edges;
int lowThreshold = 0;
const int max_lowThreshold = 100;
const int rat = 3;
const int kernel_size = 3;
const int blur_size = 10;
const char* window_name = "Edge Map";
static void CannyThreshold(int, void*)
{
    blur(src_gray, detected_edges, Size(blur_size, blur_size));
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * rat, kernel_size);
    dst = Scalar::all(0);
    src.copyTo(dst, detected_edges);
    imshow(window_name, dst);
}
// https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
// https://duckduckgo.com/?q=scientific+supervisor
// https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrmeanshiftfiltering

void WindowWithSlidersRemoveBackground(Mat& src, vector<vector<int>>& out)
{
    const char* window_name = "Remove Background";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    createTrackbar("blur:", window_name, 0, 30);
    createTrackbar("hue:", window_name, 0, 100);
    createTrackbar("saturation:", window_name, 0, 100);
    createTrackbar("value:", window_name, 0, 100);
    createTrackbar("consume_hue:", window_name, 0, 40);
    createTrackbar("consume_saturation:", window_name, 0, 40);
    createTrackbar("consume_value:", window_name, 0, 40);
    createTrackbar("saturation_cutoff:", window_name, 0, 50);

    Mat src_copy, preview;
    int key = 0;
    while (key != 27)
    {
        key = waitKey(1) & 0xFF;
        src.copyTo(src_copy);
        int blur_size = getTrackbarPos("blur:", window_name);
        int hue = getTrackbarPos("hue:", window_name);
        int saturation = getTrackbarPos("saturation:", window_name);
        int value = getTrackbarPos("value:", window_name);
        int consume_hue = getTrackbarPos("consume_hue:", window_name);
        int consume_saturation = getTrackbarPos("consume_saturation:", window_name);
        int consume_value = getTrackbarPos("consume_value:", window_name);
        int saturation_cutoff = getTrackbarPos("saturation_cutoff:", window_name);
        blur(src_copy, src_copy, Size(blur_size + 1, blur_size + 1));
        out = removeBackgroundNoDis(src_copy, preview, hue, saturation, value, consume_hue, consume_saturation, consume_value, saturation_cutoff);
        imshow(window_name, preview);
    }
}

void WindowWithSlidersCannyThreshold(Mat& src, Mat& out)
{
    const char* window_name = "Canny Threshold";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    createTrackbar("Blur:", window_name, 0, 30);
    createTrackbar("Threshold:", window_name, 0, 100);
    createTrackbar("Kernel size:", window_name, 0, 30);
    createTrackbar("rat:", window_name, 0, 15);

    Mat src_copy, detected_edges;
    int key = 0;
    while (key != 27)
    {
        src.copyTo(src_copy);
        key = waitKey(1) & 0xFF;
        int blur_size = getTrackbarPos("Blur:", window_name);
        int thresh = getTrackbarPos("Threshold:", window_name);
        int kernel_size = getTrackbarPos("Kernel size:", window_name);
        int rat = getTrackbarPos("rat:", window_name);
        blur(src_copy, detected_edges, Size(blur_size + 1, blur_size + 1));
        Canny(detected_edges, detected_edges, thresh, thresh * rat, 3);
        dst = Scalar::all(0);
        src.copyTo(dst, detected_edges);
        imshow(window_name, dst);
    }
    detected_edges.copyTo(out);
}

void WindowWithSlidersHoughLines(Mat& src, vector<Vec4i>& linesP)
{
    const char* window_name = "Hough Lines";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    createTrackbar("Min threshold:", window_name, 0, 200);
    createTrackbar("Min length:", window_name, 0, 50);
    createTrackbar("Max gap:", window_name, 0, 15);

    int key = 0;
    while (key != 27)
    {
        Mat src_copy;
        src.copyTo(src_copy);
        key = waitKey(1) & 0xFF;
        int thresh = getTrackbarPos("Min threshold:", window_name);
        int len = getTrackbarPos("Min length:", window_name);
        int gap = getTrackbarPos("Max gap:", window_name);
        HoughLinesP(detected_edges, linesP, 1, CV_PI / 180, thresh, len, gap); // runs the actual detection
        // Draw the lines
        for (size_t i = 0; i < linesP.size(); i++)
        {
            Vec4i l = linesP[i];
            line(src_copy, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);

            int mask_size = 3, dist = 2;
            Vec2f p1, p2, l1, l2, pm;
            l1 = Vec2f(l[0], l[1]);
            l2 = Vec2f(l[2], l[3]);
            pm = (l1 + l2) / 2;
            p1[0] = pm[0] - (l1[1] - pm[1]) * dist / norm(l1 - pm);
            p1[1] = pm[1] + (l1[0] - pm[0]) * dist / norm(l1 - pm);
            p2[0] = pm[0] + (l1[1] - pm[1]) * dist / norm(l1 - pm);
            p2[1] = pm[1] - (l1[0] - pm[0]) * dist / norm(l1 - pm);

            line(src_copy, Point(p1[0], p1[1]), Point(p2[0], p2[1]), Scalar(255, 0, 0), 2, LINE_AA);
        }
        imshow(window_name, src_copy);
    }
}

// !!!!!!!!!!!!!
void WindowSubtractBackground(Mat& src, int algo)
{
    //create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    if (algo)
        pBackSub = createBackgroundSubtractorMOG2();
    else
        pBackSub = createBackgroundSubtractorKNN();

    if (src.empty())
        return;

    Mat fgMask;
    //update the background model
    pBackSub->apply(src, fgMask);

    //show the current frame and the fg masks
    imshow("Frame", src);
    imshow("FG Mask", fgMask);

    return;
}

Mat crop(Mat img, int x1, int x2, int y1, int y2) {
    return img(Rect(x1, x2, y1, y2));
}


std::tuple<Vec2f, Vec2f> getCrossPoints(Vec2f p1, Vec2f p2, int cross_len, bool round_result) {
    Vec2f c1, c2, pm;
    pm = (p1 + p2) / 2;
    //x1 / y1 = A
    //y2 / x2 = A
    //A = tg a
    //x2 = cos a = x1 * dist / sqrt(x1*y1)
    //y2 = sin a = y1 * dist / sqrt(x1*y1)
    //xp1 = x1 * dist / sqrt(x1*y1) + xm
    //yp1 = y1 * dist / sqrt(x1*y1) + xm
    c1[0] = pm[0] - (p1[1] - pm[1]) * cross_len / norm(p1 - pm);
    c1[1] = pm[1] + (p1[0] - pm[0]) * cross_len / norm(p1 - pm);
    c2[0] = pm[0] + (p1[1] - pm[1]) * cross_len / norm(p1 - pm);
    c2[1] = pm[1] - (p1[0] - pm[0]) * cross_len / norm(p1 - pm);
    if (round_result) {
        c1[0] = round(c1[0]);
        c1[1] = round(c1[1]);
        c2[0] = round(c2[0]);
        c2[1] = round(c2[1]);
    }
    return { c1, c2 };
}

void drawLineWithCrossingEdge(Vec4i l, Mat img, int cross_len) {
    line(img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
    Vec2f p1, p2, c1, c2, pm;
    p1 = Vec2f(l[0], l[1]);
    p2 = Vec2f(l[2], l[3]);
    pm = (p1 + p2) / 2;
    tie(c1, c2) = getCrossPoints(p1, p2, cross_len, true);
    line(img, Point(c1[0], c1[1]), Point(c2[0], c2[1]), Scalar(0, 255, 0), 1, LINE_AA);
    tie(c1, c2) = getCrossPoints(p1, pm, cross_len, true);
    line(img, Point(c1[0], c1[1]), Point(c2[0], c2[1]), Scalar(0, 255, 0), 1, LINE_AA);
    tie(c1, c2) = getCrossPoints(pm, p2, cross_len, true);
    line(img, Point(c1[0], c1[1]), Point(c2[0], c2[1]), Scalar(0, 255, 0), 1, LINE_AA);
}

int checkPointPairDifference(Vec2f p1, Vec2f p2, vector<vector<int>> segments) {
    int n = segments.size(), m = segments[0].size();
    if (p1[0] >= 0 and p1[1] >= 0 and p2[0] >= 0 and p2[1] >= 0 and
        p1[0] < m and p1[1] < n and p2[0] < m and p2[1] < n) {
        int color1 = segments[p1[1]][p1[0]], color2 = segments[p2[1]][p2[0]];
        if (segments[p1[1]][p1[0]] != segments[p2[1]][p2[0]]) {
            return segments[p1[1]][p1[0]] - segments[p2[1]][p2[0]];
        }
    }
    return 0;
}

bool checkEdge(Vec4i l, vector<vector<int>> segments, int cross_len) {
    if (cross_len == -1) {
        return true;
    }

    Vec2f c1, c2, p1, p2, pm, pm1, pm2;
    p1 = Vec2f(l[0], l[1]);
    p2 = Vec2f(l[2], l[3]);
    pm = (p1 + p2) / 2;
    pm1 = (p1 + pm) / 2;
    pm2 = (pm + p2) / 2;

    vector<vector<Vec2f>> line_parts = {
        {p1, p2},
        {p1, pm},
        {pm, p2},
        {p1, pm1},
        {pm1, pm},
        {pm, pm2},
        {pm2, p2},
    };
    int diff_direction = 0;
    for (vector<Vec2f> line_part : line_parts) {
        auto [c1, c2] = getCrossPoints(line_part[0], line_part[1], cross_len, true);
        int diff = checkPointPairDifference(c1, c2, segments);
        if (diff_direction == 0) {
            diff_direction = diff;
        }
        if (diff == 0 or diff * diff_direction <= 0) {
            return false;
        }
    }
    return true;
}

int detectLines(Mat canny_edges, Mat img, vector<vector<int>> segment_map, vector<Vec4i>& lines, vector<Vec4i>& lines_edge, int len, int cross_len, bool visualization) {
    int thresh = 50;
    int gap = len / 5;
    //visualization = 1;
    lines.clear();
    lines_edge.clear();

    int count = 0;
    HoughLinesP(canny_edges, lines, 1, CV_PI / 180, thresh, len, gap); // runs the actual detection
    // Draw the lines
    Mat cdst = img.clone();
    Mat cdstP = img.clone();
    cvtColor(cdst, cdst, COLOR_GRAY2BGR);
    cvtColor(cdstP, cdstP, COLOR_GRAY2BGR);
    for (size_t i = 0; i < lines.size(); i++) {
        if (visualization)
            drawLineWithCrossingEdge(lines[i], cdst, cross_len);
        if (checkEdge(lines[i], segment_map, cross_len)) {
            if (visualization)
                drawLineWithCrossingEdge(lines[i], cdstP, cross_len);
            count++;
            lines_edge.push_back(lines[i]);
        }
    }
    if (visualization) {
        imshow("Hough Lines", cdst);
        imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
        waitKey();
    }

    return count;
}

int checkImageForObjects(Mat img, bool visualization) {

    if (img.empty()) {
        std::cout << "Could not open or find the image!\n" << std::endl;
        return 0;
    }

    Mat img_resized;
    resize(img, img_resized, Size(256.f * img.cols / img.rows, 256)); //resize image

    //denoising
    Mat img_denoised;
    //fastNlMeansDenoisingColored(img_resized, img_denoised);
    //imshow("Denoised", img_denoised);
    //medianBlur(img_resized, img_denoised, 3);
    //imshow("medianDenoised3", img_denoised);
    medianBlur(img_resized, img_denoised, 5);
    if (visualization)
        imshow("medianDenoised5", img_denoised);
    //fastNlMeansDenoisingColored(img_resized, img_denoised);
    //medianBlur(img_denoised, img_denoised, 5);
    //imshow("medianDenoisedDpuble", img_denoised);


    //// Initiate ORB detector
    //Ptr<FeatureDetector> orb_detector = ORB::create();
    //Ptr<DescriptorExtractor> orb_descriptor = ORB::create();
    //vector<KeyPoint> keypoints;
    //orb_detector->detect(img_resized, keypoints);
    //Mat descriptors;
    //orb_descriptor->compute(img_resized, keypoints, descriptors);
    //Mat img_keypoints;
    //drawKeypoints(img_resized, keypoints, img_keypoints, (0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //imshow("Keypoints", img_keypoints);


    //colorSegmentation(base, base, 3);
    //WindowSubtractBackground(base, 1);
    //vector<vector<int>> nobg1 = removeBackgroundNoDis(base, 1);

    if (visualization)
        imshow("Source resized", img_resized);
    //waitKey();

    Mat img_gauss_blur;
    GaussianBlur(img_denoised, img_gauss_blur, Size(0, 0), 10);
    Mat img_normalised_lighting = (img_denoised - img_gauss_blur + 100);
    if (visualization)
        imshow("Normalised lighting", img_normalised_lighting);
    //waitKey();

    vector<vector<int>> segment_map;
    //WindowWithSlidersRemoveBackground(img_normalised_lighting, segment_map);
    Mat img_segmented, img_blurred;
    int blur_size_1 = 3;
    blur(img_normalised_lighting, img_blurred, Size(blur_size_1 + 1, blur_size_1 + 1));
    /*Mat img_laplacian;
    Laplacian(img_denoised, img_laplacian, CV_16S, 31, 0.0000000000000001, 0.0, BORDER_DEFAULT);
    convertScaleAbs(img_laplacian, img_laplacian);
    imshow("laplacian", img_laplacian);
    waitKey();*/

    /*int hue = 20;
    int saturation = 20;
    int value = 3;
    int consume_hue = 4;
    int consume_saturation = 4;
    int consume_value = 4;
    int saturation_cutoff = 1;
    segment_map = removeBackgroundNoDis(img_blurred, img_segmented, hue, saturation, value, consume_hue, consume_saturation, consume_value, saturation_cutoff);*/

    Mat img_gray;
    cvtColor(img_blurred, img_gray, COLOR_BGR2GRAY);
    segment_map = colorSegmentationThresh(img_gray, img_segmented, -5, 5);
    if (visualization)
        imshow("Remove Background", img_segmented);
    //waitKey();


    Mat img_grayscale;
    cvtColor(img_resized, img_grayscale, COLOR_BGR2GRAY);
    //WindowWithSlidersCannyThreshold(src_gray, detected_edges);
    Mat detected_edges;
    int blur_size_2 = 3;
    int canny_thresh = 15;
    int kernel_size = 3;
    int rat = 3;
    blur(img_grayscale, img_blurred, Size(blur_size_2 + 1, blur_size_2 + 1));
    Canny(img_blurred, detected_edges, canny_thresh, canny_thresh * rat, kernel_size);
    dst = Scalar::all(0);
    img_resized.copyTo(dst, detected_edges);
    if (visualization)
        imshow("Canny Threshold", dst);
    //waitKey();


    // Declare the output variables
    Mat cdst, cdstP;
    // Edge detection
    //Canny(src, dst, 0, 0, 3);

    // Copy edges to the images that will display the results in BGR
    cvtColor(detected_edges, cdst, COLOR_GRAY2BGR);
    cdstP = cdst.clone();

    vector<Vec2f> lines; // will hold the results of the detection

    //// Standard Hough Line Transform
    //HoughLines(detected_edges, lines, 1, CV_PI / 180, 150, 50, 5); // runs the actual detection
    //// Draw the lines
    //for (size_t i = 0; i < lines.size(); i++)
    //{
    //    float rho = lines[i][0], theta = lines[i][1];
    //    Point pt1, pt2;
    //    double a = cos(theta), b = sin(theta);
    //    double x0 = a * rho, y0 = b * rho;
    //    pt1.x = cvRound(x0 + 1000 * (-b));
    //    pt1.y = cvRound(y0 + 1000 * (a));
    //    pt2.x = cvRound(x0 - 1000 * (-b));
    //    pt2.y = cvRound(y0 - 1000 * (a));
    //    line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    //}

    // Probabilistic Line Transform
    vector<Vec4i> linesP, lines_edge; // will hold the results of the detection
    //WindowWithSlidersHoughLines(cdstP, linesP);
    int thresh = 50;
    int len = 15;
    int gap = 3;
    int cross_len = 2;
    detectLines(detected_edges, img_segmented, segment_map, linesP, lines_edge, 15, 2, visualization);
    //waitKey();
    int criteria_small = detectLines(detected_edges, img_segmented, segment_map, linesP, lines_edge, 30, 4, visualization);
    //waitKey();
    int criteria_big = detectLines(detected_edges, img_segmented, segment_map, linesP, lines_edge, 60, 8, visualization);
    //waitKey();
    detectLines(detected_edges, img_segmented, segment_map, linesP, lines_edge, 120, 16, visualization);
    //waitKey();
    if (criteria_big > 0) {
        return 2;
    }
    else if (criteria_big + criteria_small > 0) {
        return 1;
    }
    else {
        return 0;
    }
}

void cropImagesInFolder(string path_in, string path_out, int x1, int x2, int y1, int y2) {
    for (const auto& entry : filesystem::directory_iterator(path_in)) {
        std::cout << entry.path() << std::endl;
        Mat img = imread(entry.path().string());
        img = crop(img, x1, x2, y1, y2);
        imwrite(path_out + "\\" + entry.path().filename().string(), img);
    }
}



#include <cmath>

// ----------------------------------------------------------------------
// Expanded feature structure
// ----------------------------------------------------------------------
struct ImageFeatures {
    // Original RGB statistics
    //double r_mean, g_mean, b_mean;
    //double r_std, g_std, b_std;
    double edge_density;             // Canny edge density
    int    num_lines1;
    int    num_lines2;
    int    num_lines3;
    int    num_lines4;
    double total_line_length1;
    double total_line_length2;
    double total_line_length3;
    double total_line_length4;
    int    num_lines_edge1;
    int    num_lines_edge2;
    int    num_lines_edge3;
    int    num_lines_edge4;
    double total_line_length_edge1;
    double total_line_length_edge2;
    double total_line_length_edge3;
    double total_line_length_edge4;

    // New colour statistics
    //double r_kurt, g_kurt, b_kurt;   // per‑channel kurtosis
    //double L_mean, L_std;            // lightness (Lab L* channel)
    double color_grad_mean, color_grad_std;  // colour gradient magnitude stats
    double color_edge_density;       // density of strong colour edges
};

// ----------------------------------------------------------------------
// Feature extraction for a single image
// ----------------------------------------------------------------------
ImageFeatures extractFeatures(const cv::Mat& img) {
    ImageFeatures f;
    CV_Assert(!img.empty());

    //// ---- 1. Original RGB per‑channel mean / std ----
    //cv::Scalar mean, stddev;
    //cv::meanStdDev(img, mean, stddev);
    //f.r_mean = mean[2];   // BGR order
    //f.g_mean = mean[1];
    //f.b_mean = mean[0];
    //f.r_std = stddev[2];
    //f.g_std = stddev[1];
    //f.b_std = stddev[0];

    //// ---- 2. Kurtosis per channel (excess kurtosis) ----
    //// We iterate over the image to compute the 4th moment.
    //// Using the previously computed mean and std.
    //double n_pixels = img.rows * img.cols;
    //double r_m4 = 0.0, g_m4 = 0.0, b_m4 = 0.0;
    //for (int y = 0; y < img.rows; ++y) {
    //    const cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
    //    for (int x = 0; x < img.cols; ++x) {
    //        double r_diff = row[x][2] - f.r_mean;
    //        double g_diff = row[x][1] - f.g_mean;
    //        double b_diff = row[x][0] - f.b_mean;
    //        r_m4 += r_diff * r_diff * r_diff * r_diff;
    //        g_m4 += g_diff * g_diff * g_diff * g_diff;
    //        b_m4 += b_diff * b_diff * b_diff * b_diff;
    //    }
    //}
    //r_m4 /= n_pixels; g_m4 /= n_pixels; b_m4 /= n_pixels;
    //auto kurt = [](double m4, double std) {
    //    if (std < 1e-6) return 0.0;       // avoid division by zero
    //    return m4 / (std * std * std * std) - 3.0;
    //    };
    //f.r_kurt = kurt(r_m4, f.r_std);
    //f.g_kurt = kurt(g_m4, f.g_std);
    //f.b_kurt = kurt(b_m4, f.b_std);

    //// ---- 3. Lightness (Lab L* channel) statistics ----
    //cv::Mat lab;
    //cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
    //std::vector<cv::Mat> lab_planes;
    //cv::split(lab, lab_planes);       // 0:L, 1:a, 2:b
    //cv::Scalar L_mean_s, L_std_s;
    //cv::meanStdDev(lab_planes[0], L_mean_s, L_std_s);
    //f.L_mean = L_mean_s[0];
    //f.L_std = L_std_s[0];

    // ---- 4. Colour gradient magnitude (vector norm over R,G,B) ----
    // Split into BGR channels
    std::vector<cv::Mat> bgr;
    cv::split(img, bgr);
    cv::Mat grad_x, grad_y, mag2_sum(img.size(), CV_32F, cv::Scalar(0));
    for (int c = 0; c < 3; ++c) {
        cv::Mat ch = bgr[c];
        cv::Mat gx, gy, mag;
        cv::Sobel(ch, gx, CV_32F, 1, 0);
        cv::Sobel(ch, gy, CV_32F, 0, 1);
        cv::magnitude(gx, gy, mag);           // per‑channel gradient magnitude
        cv::accumulateSquare(mag, mag2_sum);  // sum of squared magnitudes
    }
    cv::Mat color_grad;
    cv::sqrt(mag2_sum, color_grad);           // final colour gradient magnitude

    cv::Scalar cg_mean, cg_std;
    cv::meanStdDev(color_grad, cg_mean, cg_std);
    f.color_grad_mean = cg_mean[0];
    f.color_grad_std = cg_std[0];

    // ---- 5. Colour edge density (threshold on colour gradient) ----
    // Use a fixed threshold (tunable) – adjust according to your image range.
    const double edge_thresh = 50.0;
    cv::Mat color_edges = color_grad > edge_thresh;
    int edge_pix = cv::countNonZero(color_edges);
    f.color_edge_density = static_cast<double>(edge_pix) / (img.rows * img.cols);

    // ---- Original grey‑level edge density and line features ----
    Mat img_grayscale, img_blurred;
    cvtColor(img, img_grayscale, COLOR_BGR2GRAY);
    //WindowWithSlidersCannyThreshold(src_gray, detected_edges);
    Mat detected_edges;
    int blur_size_2 = 3;
    int canny_thresh = 35;
    int kernel_size = 3;
    int rat = 3;
    blur(img_grayscale, img_blurred, Size(blur_size_2 + 1, blur_size_2 + 1));
    Canny(img_blurred, detected_edges, canny_thresh, canny_thresh * rat, kernel_size);
    f.edge_density = static_cast<double>(cv::countNonZero(detected_edges)) / (img.rows * img.cols);
    /*imshow("Canny Threshold1", img_grayscale);
    imshow("Canny Threshold2", img_blurred);
    Mat vis_edges;
    img.copyTo(vis_edges, detected_edges);
    imshow("Canny Threshold3", vis_edges);
    waitKey();*/

    vector<vector<int>> segment_map;
    Mat img_segmented;
    segment_map = colorSegmentationThresh(img_grayscale, img_segmented, -5, 5);
    std::vector<cv::Vec4i> lines, lines_edge;
    int thresh = 50;
    int len = 15;
    int gap = 3;
    int cross_len = 2;
    detectLines(detected_edges, img_segmented, segment_map, lines, lines_edge, 15, 2, false);
    f.num_lines1 = lines.size();
    f.num_lines_edge1 = lines_edge.size();
    f.total_line_length1 = 0.0;
    f.total_line_length_edge1 = 0.0;
    for (const auto& l : lines) {
        double dx = l[2] - l[0];
        double dy = l[3] - l[1];
        f.total_line_length1 += std::sqrt(dx * dx + dy * dy);
    }
    for (const auto& l : lines_edge) {
        double dx = l[2] - l[0];
        double dy = l[3] - l[1];
        f.total_line_length_edge1 += std::sqrt(dx * dx + dy * dy);
    }
    detectLines(detected_edges, img_segmented, segment_map, lines, lines_edge, 30, 4, false);
    f.num_lines2 = lines.size();
    f.num_lines_edge2 = lines_edge.size();
    f.total_line_length2 = 0.0;
    f.total_line_length_edge2 = 0.0;
    for (const auto& l : lines) {
        double dx = l[2] - l[0];
        double dy = l[3] - l[1];
        f.total_line_length2 += std::sqrt(dx * dx + dy * dy);
    }
    for (const auto& l : lines_edge) {
        double dx = l[2] - l[0];
        double dy = l[3] - l[1];
        f.total_line_length_edge2 += std::sqrt(dx * dx + dy * dy);
    }
    detectLines(detected_edges, img_segmented, segment_map, lines, lines_edge, 60, 8, false);
    f.num_lines3 = lines.size();
    f.num_lines_edge3 = lines_edge.size();
    f.total_line_length3 = 0.0;
    f.total_line_length_edge3 = 0.0;
    for (const auto& l : lines) {
        double dx = l[2] - l[0];
        double dy = l[3] - l[1];
        f.total_line_length3 += std::sqrt(dx * dx + dy * dy);
    }
    for (const auto& l : lines_edge) {
        double dx = l[2] - l[0];
        double dy = l[3] - l[1];
        f.total_line_length_edge3 += std::sqrt(dx * dx + dy * dy);
    }
    detectLines(detected_edges, img_segmented, segment_map, lines, lines_edge, 120, 16, false);
    f.num_lines4 = lines.size();
    f.num_lines_edge4 = lines_edge.size();
    f.total_line_length4 = 0.0;
    f.total_line_length_edge4 = 0.0;
    for (const auto& l : lines) {
        double dx = l[2] - l[0];
        double dy = l[3] - l[1];
        f.total_line_length4 += std::sqrt(dx * dx + dy * dy);
    }
    for (const auto& l : lines_edge) {
        double dx = l[2] - l[0];
        double dy = l[3] - l[1];
        f.total_line_length_edge4 += std::sqrt(dx * dx + dy * dy);
    }

    return f;
}

// ----------------------------------------------------------------------
// Convert a feature set into a flat vector (order must match computation)
// ----------------------------------------------------------------------
std::vector<double> featureVector(const ImageFeatures& f) {
    return {
        // Original
        //f.r_mean, f.g_mean, f.b_mean,
        //f.r_std,  f.g_std,  f.b_std,
        f.edge_density,
        static_cast<double>(f.num_lines1),
        static_cast<double>(f.num_lines2),
        static_cast<double>(f.num_lines3),
        static_cast<double>(f.num_lines4),
        f.total_line_length1,
        f.total_line_length2,
        f.total_line_length3,
        f.total_line_length4,
        static_cast<double>(f.num_lines_edge1),
        static_cast<double>(f.num_lines_edge2),
        static_cast<double>(f.num_lines_edge3),
        static_cast<double>(f.num_lines_edge4),
        f.total_line_length_edge1,
        f.total_line_length_edge2,
        f.total_line_length_edge3,
        f.total_line_length_edge4,
        // New
        //f.r_kurt, f.g_kurt, f.b_kurt,
        //f.L_mean, f.L_std,
        f.color_grad_mean, f.color_grad_std,
        f.color_edge_density
    };
}

// ----------------------------------------------------------------------
// Compute mean and standard deviation of each feature over a set of vectors
// ----------------------------------------------------------------------
void computeStats(const std::vector<std::vector<double>>& samples,
    std::vector<double>& mean,
    std::vector<double>& stddev) {
    size_t N = samples.size();
    if (N == 0) return;
    size_t D = samples[0].size();
    mean.assign(D, 0.0);
    stddev.assign(D, 0.0);

    // Mean
    for (const auto& vec : samples)
        for (size_t i = 0; i < D; ++i)
            mean[i] += vec[i];
    for (size_t i = 0; i < D; ++i)
        mean[i] /= N;

    // Standard deviation (population)
    for (const auto& vec : samples)
        for (size_t i = 0; i < D; ++i)
            stddev[i] += (vec[i] - mean[i]) * (vec[i] - mean[i]);
    for (size_t i = 0; i < D; ++i)
        stddev[i] = std::sqrt(stddev[i] / N);
}

// ----------------------------------------------------------------------
// Anomaly score: sum of squared z‑scores
// ----------------------------------------------------------------------
double anomalyScore(const std::vector<double>& vec,
    const std::vector<double>& mean,
    const std::vector<double>& stddev,
    double eps = 1e-6) {
    double score = 0.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        double diff = vec[i] - mean[i];
        double denom = stddev[i] + eps;
        double z = diff / denom;
        score += z * z;
    }
    return score;
}

// ----------------------------------------------------------------------
// Helper: print a feature vector with a label
// ----------------------------------------------------------------------
void printFeatureVector(const std::string& label,
    const std::vector<double>& vec, const std::vector<double>& mean, const std::vector<double>& stddev) {
    std::cout << label << " [";
    double eps = 1e-6;
    for (size_t i = 0; i < vec.size(); ++i) {
        double diff = vec[i] - mean[i];
        double denom = stddev[i] + eps;
        double z = diff / denom;
        std::cout << z * z;
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}


// ----------------------------------------------------------------------
// Preprocessing: resize, blur, lighting normalisation
// ----------------------------------------------------------------------
cv::Mat preprocessImage(const cv::Mat& img) {
    // 1. Resize so height = 256, keep aspect ratio
    cv::Mat resized;
    int new_width = static_cast<int>(256.0 * img.cols / img.rows);
    cv::resize(img, resized, cv::Size(new_width, 256));

    
    // Convert to grayscale if needed
    cv::Mat gray;
    if (resized.channels() == 3)
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    else
        gray = resized.clone();

    // ----- Step 1: Enhance contrast with CLAHE -----
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);    // Limits over‑amplification of noise
    clahe->setTilesGridSize(cv::Size(8, 8));  // 8x8 tiles – adjust based on image size

    cv::Mat enhanced;
    clahe->apply(gray, enhanced);
    cv::cvtColor(enhanced, enhanced, cv::COLOR_GRAY2BGR);
    cv::waitKey(0);


    // 2. Strong Gaussian blur for illumination estimation

    // 3. Lighting normalisation: subtract illumination + constant offset
    cv::Mat normalized;
    enhanced.convertTo(normalized, CV_32F, 1.0 / 255);
    cv::Mat blurred;
    cv::GaussianBlur(normalized, blurred, cv::Size(0, 0), 50.0);
    //cv::subtract(enhanced, blurred, normalized);
    normalized = normalized - blurred + 100;   // add 100 to all channels;
    cv::normalize(normalized, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    //imshow("normalized", normalized);
    //waitKey();


    return normalized;
}

// ----------------------------------------------------------------------
// Print feature vector with label
// ----------------------------------------------------------------------
void printFeatureVector(const std::string& label, const std::vector<double>& vec) {
    std::cout << label << " [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// ----------------------------------------------------------------------
// Process all images in a folder, return feature vectors
// ----------------------------------------------------------------------
void processFolder(const std::string& folder,
    const std::string& label_prefix,
    std::vector<std::vector<double>>& out_vectors) {
    std::vector<cv::String> files;
    cv::glob(folder, files);
    for (const auto& file : files) {
        cv::Mat img = cv::imread(file);
        if (img.empty()) {
            std::cerr << "Warning: could not read " << file << std::endl;
            continue;
        }
        cv::Mat preprocessed = preprocessImage(img);
        ImageFeatures f = extractFeatures(preprocessed);
        std::vector<double> vec = featureVector(f);
        out_vectors.push_back(vec);
        printFeatureVector(label_prefix + " " + file, vec);
    }
}

// ----------------------------------------------------------------------
int findOnjectImages(String bg_folder, vector<string> test_folders, double thresh_k) {
    // ---------- 1. Read background images ----------
    std::vector<cv::String> bg_files;
    std::vector<std::vector<double>> bg_vectors;
    processFolder(bg_folder, "BG", bg_vectors);
    if (bg_vectors.empty()) {
        std::cerr << "No background images found in ./bg" << std::endl;
        return -1;
    }

    // ---------- Compute background statistics ----------
    std::vector<double> bg_mean, bg_std;
    computeStats(bg_vectors, bg_mean, bg_std);

    // ---------- Determine anomaly threshold from background ----------
    std::vector<double> bg_scores;
    for (const auto& v : bg_vectors)
        bg_scores.push_back(anomalyScore(v, bg_mean, bg_std));
    std::sort(bg_scores.begin(), bg_scores.end());
    double threshold = bg_scores[static_cast<size_t>(0.99 * bg_scores.size())];
    std::cout << "\nAnomaly threshold (99th percentile of background): "
        << threshold << "\n" << std::endl;

    // ---------- 4. Read test images ----------
    std::map<std::string, int> total_counts, anomaly_counts;

    for (const auto& folder : test_folders) {
        std::vector<cv::String> files;
        cv::glob(folder, files);
        total_counts[folder] = files.size();
        anomaly_counts[folder] = 0;

        for (const auto& file : files) {
            cv::Mat img = cv::imread(file);
            if (img.empty()) {
                std::cerr << "Warning: could not read " << file << std::endl;
                continue;
            }
            cv::Mat preprocessed = preprocessImage(img);
            ImageFeatures f = extractFeatures(preprocessed);
            std::vector<double> vec = featureVector(f);
            printFeatureVector("TEST " + file, vec);

            double score = anomalyScore(vec, bg_mean, bg_std);
            if (score > threshold) {
                std::cout << "  *** ANOMALY *** (score: " << score << ")\n";
                anomaly_counts[folder]++;
            }
        }
    }

    // ---------- 5. Summary statistics ----------
    std::cout << "\n========== RESULT ==========\n";
    for (const auto& folder : test_folders) {
        int total = total_counts[folder];
        int anom = anomaly_counts[folder];
        double perc = total > 0 ? 100.0 * anom / total : 0.0;
        std::cout << folder << ": " << anom << " / " << total
            << " flagged as anomaly (" << perc << "%)\n";
    }
    std::cout << "============================\n";

    return 0;
}













//struct ImageFeatures {
//    double edge_density;
//    int    num_lines1, num_lines2, num_lines3, num_lines4;
//    double total_line_length1, total_line_length2, total_line_length3, total_line_length4;
//    int    num_lines_edge1, num_lines_edge2, num_lines_edge3, num_lines_edge4;
//    double total_line_length_edge1, total_line_length_edge2,
//        total_line_length_edge3, total_line_length_edge4;
//    double color_grad_mean, color_grad_std;
//    double color_edge_density;
//};
//
//static tuple<Vec2f, Vec2f> getCrossPoints(Vec2f p1, Vec2f p2, int cross_len, bool round_result) {
//    Vec2f c1, c2, pm;
//    pm = (p1 + p2) / 2;
//    c1[0] = pm[0] - (p1[1] - pm[1]) * cross_len / norm(p1 - pm);
//    c1[1] = pm[1] + (p1[0] - pm[0]) * cross_len / norm(p1 - pm);
//    c2[0] = pm[0] + (p1[1] - pm[1]) * cross_len / norm(p1 - pm);
//    c2[1] = pm[1] - (p1[0] - pm[0]) * cross_len / norm(p1 - pm);
//    if (round_result) {
//        c1[0] = round(c1[0]); c1[1] = round(c1[1]);
//        c2[0] = round(c2[0]); c2[1] = round(c2[1]);
//    }
//    return { c1, c2 };
//}
//
//static int checkPointPairDifference(Vec2f p1, Vec2f p2, const vector<vector<int>>& segments) {
//    int n = (int)segments.size(), m = (int)segments[0].size();
//    if (p1[0] >= 0 && p1[1] >= 0 && p2[0] >= 0 && p2[1] >= 0 &&
//        p1[0] < m && p1[1] < n && p2[0] < m && p2[1] < n) {
//        int color1 = segments[p1[1]][p1[0]];
//        int color2 = segments[p2[1]][p2[0]];
//        if (color1 != color2)
//            return color1 - color2;
//    }
//    return 0;
//}
//
//static bool checkEdge(Vec4i l, const vector<vector<int>>& segments, int cross_len) {
//    if (cross_len == -1)
//        return true;
//
//    Vec2f p1(l[0], l[1]), p2(l[2], l[3]);
//    Vec2f pm = (p1 + p2) / 2;
//    Vec2f pm1 = (p1 + pm) / 2;
//    Vec2f pm2 = (pm + p2) / 2;
//
//    vector<vector<Vec2f>> line_parts = {
//        {p1, p2}, {p1, pm}, {pm, p2},
//        {p1, pm1}, {pm1, pm}, {pm, pm2}, {pm2, p2}
//    };
//    int diff_direction = 0;
//    for (auto& line_part : line_parts) {
//        auto [c1, c2] = getCrossPoints(line_part[0], line_part[1], cross_len, true);
//        int diff = checkPointPairDifference(c1, c2, segments);
//        if (diff_direction == 0)
//            diff_direction = diff;
//        if (diff == 0 || diff * diff_direction <= 0)
//            return false;
//    }
//    return true;
//}
//
//static int detectLines(const Mat& canny_edges,
//    const Mat& /*img*/,
//    const vector<vector<int>>& segment_map,
//    vector<Vec4i>& lines,
//    vector<Vec4i>& lines_edge,
//    int len, int cross_len) {
//    int thresh = 50;
//    int gap = len / 5;
//    int count = 0;
//    lines.clear();
//    HoughLinesP(canny_edges, lines, 1, CV_PI / 180, thresh, len, gap);
//    for (size_t i = 0; i < lines.size(); i++) {
//        if (checkEdge(lines[i], segment_map, cross_len)) {
//            count++;
//            lines_edge.push_back(lines[i]);
//        }
//    }
//    return count;
//}
//
//Mat preprocessImage(const Mat& img) {
//    Mat resized;
//    int new_width = static_cast<int>(256.0 * img.cols / img.rows);
//    resize(img, resized, Size(new_width, 256));
//
//    Mat blurred;
//    GaussianBlur(resized, blurred, Size(0, 0), 10.0);
//
//    Mat normalized;
//    subtract(resized, blurred, normalized);
//    normalized += Scalar(100, 100, 100);
//    return normalized;
//}
//
//ImageFeatures extractFeatures(const Mat& img) {
//    ImageFeatures f;
//    CV_Assert(!img.empty());
//
//    vector<Mat> bgr;
//    split(img, bgr);
//    Mat mag2_sum(img.size(), CV_32F, Scalar(0));
//    for (int c = 0; c < 3; ++c) {
//        Mat gx, gy, mag;
//        Sobel(bgr[c], gx, CV_32F, 1, 0);
//        Sobel(bgr[c], gy, CV_32F, 0, 1);
//        magnitude(gx, gy, mag);
//        accumulateSquare(mag, mag2_sum);
//    }
//    Mat color_grad;
//    sqrt(mag2_sum, color_grad);
//    Scalar cg_mean, cg_std;
//    meanStdDev(color_grad, cg_mean, cg_std);
//    f.color_grad_mean = cg_mean[0];
//    f.color_grad_std = cg_std[0];
//
//    const double edge_thresh = 50.0;
//    Mat color_edges = color_grad > edge_thresh;
//    f.color_edge_density = static_cast<double>(countNonZero(color_edges)) / (img.rows * img.cols);
//
//    Mat gray, blur_img, edges;
//    cvtColor(img, gray, COLOR_BGR2GRAY);
//    int blur_kernel = 3;
//    blur(gray, blur_img, Size(blur_kernel + 1, blur_kernel + 1));
//    int canny_thresh = 15, kernel_size = 3, rat = 3;
//    Canny(blur_img, edges, canny_thresh, canny_thresh * rat, kernel_size);
//    f.edge_density = static_cast<double>(countNonZero(edges)) / (img.rows * img.cols);
//
//    vector<vector<int>> segment_map;
//    Mat segmented;
//    segment_map = colorSegmentationThresh(gray, segmented, -5, 5);
//
//    vector<Vec4i> lines, lines_edge;
//    // scale 1: len=15, cross=2
//    detectLines(edges, segmented, segment_map, lines, lines_edge, 15, 2);
//    f.num_lines1 = (int)lines.size() + 1;
//    f.num_lines_edge1 = (int)lines_edge.size() + 1;
//    f.total_line_length1 = 1; f.total_line_length_edge1 = 1;
//    for (auto& l : lines) {
//        double dx = l[2] - l[0], dy = l[3] - l[1];
//        f.total_line_length1 += sqrt(dx * dx + dy * dy);
//    }
//    for (auto& l : lines_edge) {
//        double dx = l[2] - l[0], dy = l[3] - l[1];
//        f.total_line_length_edge1 += sqrt(dx * dx + dy * dy);
//    }
//    // scale 2: len=30, cross=4
//    detectLines(edges, segmented, segment_map, lines, lines_edge, 30, 4);
//    f.num_lines2 = (int)lines.size() + 1;
//    f.num_lines_edge2 = (int)lines_edge.size() + 1;
//    f.total_line_length2 = 1; f.total_line_length_edge2 = 1;
//    for (auto& l : lines) {
//        double dx = l[2] - l[0], dy = l[3] - l[1];
//        f.total_line_length2 += sqrt(dx * dx + dy * dy);
//    }
//    for (auto& l : lines_edge) {
//        double dx = l[2] - l[0], dy = l[3] - l[1];
//        f.total_line_length_edge2 += sqrt(dx * dx + dy * dy);
//    }
//    // scale 3: len=60, cross=8
//    detectLines(edges, segmented, segment_map, lines, lines_edge, 60, 8);
//    f.num_lines3 = (int)lines.size() + 1;
//    f.num_lines_edge3 = (int)lines_edge.size() + 1;
//    f.total_line_length3 = 1; f.total_line_length_edge3 = 1;
//    for (auto& l : lines) {
//        double dx = l[2] - l[0], dy = l[3] - l[1];
//        f.total_line_length3 += sqrt(dx * dx + dy * dy);
//    }
//    for (auto& l : lines_edge) {
//        double dx = l[2] - l[0], dy = l[3] - l[1];
//        f.total_line_length_edge3 += sqrt(dx * dx + dy * dy);
//    }
//    // scale 4: len=120, cross=16
//    detectLines(edges, segmented, segment_map, lines, lines_edge, 120, 16);
//    f.num_lines4 = (int)lines.size() + 1;
//    f.num_lines_edge4 = (int)lines_edge.size() + 1;
//    f.total_line_length4 = 1; f.total_line_length_edge4 = 1;
//    for (auto& l : lines) {
//        double dx = l[2] - l[0], dy = l[3] - l[1];
//        f.total_line_length4 += sqrt(dx * dx + dy * dy);
//    }
//    for (auto& l : lines_edge) {
//        double dx = l[2] - l[0], dy = l[3] - l[1];
//        f.total_line_length_edge4 += sqrt(dx * dx + dy * dy);
//    }
//
//    return f;
//}
//
//vector<double> featureVector(const ImageFeatures& f) {
//    return {
//        f.edge_density,
//        static_cast<double>(f.num_lines1),
//        static_cast<double>(f.num_lines2),
//        static_cast<double>(f.num_lines3),
//        static_cast<double>(f.num_lines4),
//        f.total_line_length1,
//        f.total_line_length2,
//        f.total_line_length3,
//        f.total_line_length4,
//        static_cast<double>(f.num_lines_edge1),
//        static_cast<double>(f.num_lines_edge2),
//        static_cast<double>(f.num_lines_edge3),
//        static_cast<double>(f.num_lines_edge4),
//        f.total_line_length_edge1,
//        f.total_line_length_edge2,
//        f.total_line_length_edge3,
//        f.total_line_length_edge4,
//        f.color_grad_mean,
//        f.color_grad_std,
//        f.color_edge_density
//    };
//}
//
//void computeStats(const vector<vector<double>>& samples,
//    vector<double>& mean,
//    vector<double>& stddev) {
//    size_t N = samples.size();
//    if (N == 0) return;
//    size_t D = samples[0].size();
//    mean.assign(D, 0.0);
//    stddev.assign(D, 0.0);
//
//    for (auto& vec : samples)
//        for (size_t i = 0; i < D; ++i)
//            mean[i] += vec[i];
//    for (size_t i = 0; i < D; ++i)
//        mean[i] /= N;
//
//    for (auto& vec : samples)
//        for (size_t i = 0; i < D; ++i)
//            stddev[i] += (vec[i] - mean[i]) * (vec[i] - mean[i]);
//    for (size_t i = 0; i < D; ++i)
//        stddev[i] = sqrt(stddev[i] / N);
//}
//
//double anomalyScore(const vector<double>& vec,
//    const vector<double>& mean,
//    const vector<double>& stddev,
//    double eps = 1e-6) {
//    double score = 0.0;
//    for (size_t i = 0; i < vec.size(); ++i) {
//        double diff = vec[i] - mean[i];
//        double denom = stddev[i] + eps;
//        double z = diff / denom;
//        score += z * z;
//    }
//    return score;
//}
//
//void printFeatureVector(const string& label, const vector<double>& vec) {
//    cout << label << " [";
//    for (size_t i = 0; i < vec.size(); ++i) {
//        cout << vec[i];
//        if (i < vec.size() - 1) cout << ", ";
//    }
//    cout << "]" << endl;
//}
//
//void processFolder(const string& folder,
//    const string& label_prefix,
//    vector<vector<double>>& out_vectors) {
//    vector<String> files;
//    glob(folder, files);
//    for (auto& file : files) {
//        Mat img = imread(file);
//        if (img.empty()) {
//            cerr << "Warning: could not read " << file << endl;
//            continue;
//        }
//        Mat preprocessed = preprocessImage(img);
//        ImageFeatures f = extractFeatures(preprocessed);
//        vector<double> vec = featureVector(f);
//        out_vectors.push_back(vec);
//        printFeatureVector(label_prefix + " " + file, vec);
//    }
//}
//
//int findOnjectImages(String bg_folder, vector<string> test_folders, double thresh_k) {
//    vector<vector<double>> bg_vectors;
//    processFolder(bg_folder, "BG", bg_vectors);
//
//    vector<double> bg_mean, bg_std;
//    computeStats(bg_vectors, bg_mean, bg_std);
//
//    vector<double> bg_scores;
//    for (auto& v : bg_vectors)
//        bg_scores.push_back(anomalyScore(v, bg_mean, bg_std));
//    sort(bg_scores.begin(), bg_scores.end());
//    double threshold = bg_scores[static_cast<size_t>(0.99 * bg_scores.size())];
//    cout << "threshold: " << threshold << "\n" << endl;
//
//    map<string, int> total_counts, anomaly_counts;
//
//    for (auto& folder : test_folders) {
//        vector<String> files;
//        glob(folder, files);
//        total_counts[folder] = (int)files.size();
//        anomaly_counts[folder] = 0;
//
//        for (auto& file : files) {
//            Mat img = imread(file);
//            if (img.empty()) {
//                continue;
//            }
//            Mat preprocessed = preprocessImage(img);
//            ImageFeatures f = extractFeatures(preprocessed);
//            vector<double> vec = featureVector(f);
//            printFeatureVector(file, vec);
//
//            double score = anomalyScore(vec, bg_mean, bg_std);
//            if (score > threshold * thresh_k) {
//                cout << "  *** Object *** (score: " << score << ")\n";
//                anomaly_counts[folder]++;
//            }
//        }
//    }
//
//    cout << "\n===========================\n";
//    for (auto& folder : test_folders) {
//        int total = total_counts[folder];
//        int anom = anomaly_counts[folder];
//        double perc = total > 0 ? 100.0 * anom / total : 0.0;
//        cout << folder << ": " << anom << " / " << total
//            << " objects " << perc << "%)\n";
//    }
//    cout << "============================\n";
//
//    return 0;
//}


