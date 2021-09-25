/*
*   GHANDOURI Féras - 11601442
*   M2 ID3D - 2021/2022
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <chrono>
#include <iostream>
#include <algorithm>

struct WinProps
{
    WinProps(const std::string& name, const cv::Size& size)
        : m_Name(name), m_Size(size) 
    {}
    const std::string& name() { return m_Name;  }
    const cv::Size& size() { return m_Size;  }
    std::string m_Name;
    cv::Size m_Size;
};

namespace ImIO
{
    cv::Mat readImg(const std::string& filename)
    {
        cv::Mat img = cv::imread(filename);

        if (img.empty())
        {
            std::cerr << "Could not read image: " << filename << std::endl;
        }

        return img;
    }
}

namespace ImOp
{
    double  conv(const cv::Mat& image, const cv::Mat& filter, const cv::Point& pixel)
    {
        int half = floor(filter.rows / 2);
        double value = 0.0;
        for (int x = -half; x <= half; ++x)
        {
            for (int y = -half; y <= half; ++y)
            {
                value += image.at<double>(pixel.x + x, pixel.y + y) * filter.at<double>(x + half, y + half);
            }
        }

        //return value;
        return std::max(0.0, std::min(value, 255.0));
        // Normalize value to be -255 - 255 range
        // Normalize value to be 0 - 255 range if image
        // amplitude
        // orientation/pente du gradient = theta arctan(grad(y)/grad(x))
    }
    cv::Mat conv2d(cv::Mat& srcImg, const cv::Mat& filter)
    {
        cv::Mat outImg(srcImg.rows, srcImg.cols, CV_64F);

        if (srcImg.channels() > 1)
        {
            cv::cvtColor(srcImg, srcImg, cv::COLOR_BGR2GRAY);
        }

        if (srcImg.type() != CV_64F)
        {
            srcImg.convertTo(srcImg, CV_64F);
        }

        // Actual convolution
        const int filterHalf = floor(filter.cols / 2);

        // Perform convolution and measure time
        auto start = std::chrono::high_resolution_clock::now();
        for (int x = filterHalf; x < srcImg.rows - filterHalf; ++x)
        {
            for (int y = filterHalf; y < srcImg.cols - filterHalf; ++y)
            {
                outImg.at<double>(x, y) = conv(srcImg, filter, cv::Point(x, y));
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);

        std::cout << "Image dimensions : " << srcImg.cols << "x" << srcImg.rows << " \n";
        std::cout << "Convolution duration : " << duration.count() << " ms. \n";

        srcImg.convertTo(srcImg, CV_8UC1);
        outImg.convertTo(outImg, CV_8UC1);

        return outImg;
    }
}

int main()
{
    // Initialize window
    WinProps resultWindow("Result", {800, 600});
    cv::namedWindow(resultWindow.name());

    // Read source image
    cv::Mat img = ImIO::readImg("../../data/img/pantheon.jpg");

    // 3x3 directional filters
    cv::Mat prewitt = (cv::Mat_<double>(3,3) << /* R0 */ -1.0, 0.0, 1.0,   /* R1 */ -1.0, 0.0, 1.0,  /* R2 */-1.0, 0.0, 1.0)   * 1.0 / 3.0;
    cv::Mat sobel   = (cv::Mat_<double>(3,3) << /* R0 */ -1.0,  0.0,  1.0, /* R1 */ -2.0, 0.0, 2.0,  /* R2 */ -1.0, 0.0, 1.0)  * 1.0 / 4.0;
    cv::Mat kirsch4 = (cv::Mat_<double>(3,3) << /* R0 */ -3.0, -3.0, -3.0, /* R1 */ 5.0,  0.0, -3.0, /* R2 */ 5.0,  5.0, -3.0) * 1.0 / 15.0;

    // Apply convolution
    cv::Mat out = ImOp::conv2d(img, kirsch4);
    
    // Display result
    imshow(resultWindow.name(), out);
    
    // Wait for a keystroke before terminating
    int key = cv::waitKey(0); 

    return 0;
}