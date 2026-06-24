using namespace cv;

Mat blend_soft_light(const Mat& img1, const Mat& img2) {
    Mat3f img1f, img2f;
    img1.convertTo(img1f, CV_32F, 1. / 255.);
    img2.convertTo(img2f, CV_32F, 1. / 255.);

    Mat3f result(img1f.rows, img1f.cols, Vec3f(0., 0., 0.));

    for (int r = 0; r < img1f.rows; ++r) {
        for (int c = 0; c < img2f.cols; ++c) {
            const Vec3f& v1 = img1f(r, c);
            const Vec3f& v2 = img2f(r, c);
            Vec3f& out = result(r, c);

            for (int k = 0; k < 3; k++) {
                if (v2[k] < 0.5)
                    out[k] = 2 * v1[k] * v2[k] + pow(v1[k], 2) * (1 - 2 * v2[k]);
                else
                    out[k] = 2 * v1[k] * (1 - v2[k]) + sqrt(v1[k]) * (2 * v2[k] - 1);
            }
        }
    }

    Mat3b result3b;
    result.convertTo(result3b, CV_8U, 255.);

    return result3b;
}

Mat blend_multiply(const Mat& img1, const Mat& img2, bool convert = true) {
    Mat3f img1f, img2f;
    if (convert) {
        img1.convertTo(img1f, CV_32F, 1. / 255.);
        img2.convertTo(img2f, CV_32F, 1. / 255.);
    }
    else {
        img1.convertTo(img1f, CV_32F, 1. / 255.);
        img2.convertTo(img2f, CV_32F, 2. / 255);
    }

    Mat3f result(img1f.rows, img1f.cols, Vec3f(0., 0., 0.));

    for (int r = 0; r < img1f.rows; ++r) {
        for (int c = 0; c < img2f.cols; ++c) {
            const Vec3f& v1 = img1f(r, c);
            const Vec3f& v2 = img2f(r, c);
            Vec3f& out = result(r, c);

            for (int k = 0; k < 3; k++) {
                out[k] = max(0.f, min(v1[k] * v2[k], 1.f));
            }
        }
    }

    Mat3b result3b;
    result.convertTo(result3b, CV_8UC3, 255.);

    return result3b;
}

Mat underwaterFilter(Mat base, Mat mask, Vec3d hue = Vec3d(1, 68, 36), int noise_mean = 128, int noise_stddev = 30) {
    cv::Mat hue_layer(base.rows, base.cols, CV_8UC3, cv::Scalar(hue));
    Mat result = blend_soft_light(base, hue_layer);

    Mat noise_layer = Mat::zeros(Size(base.cols, base.rows), CV_8U);
    randn(noise_layer, noise_mean, noise_stddev);
    cvtColor(noise_layer, noise_layer, COLOR_GRAY2RGB);
    result = blend_soft_light(result, noise_layer);

    result = blend_multiply(result, mask);

    //imshow("Display window", result);
    //int k = waitKey(0);

    return result;
}
