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

        assert( !img.empty(), ("Could not read image" ) );

        return img;
    }
}

namespace ImOp
{
    double conv(const cv::Mat& image, const cv::Mat& filter, const cv::Point& pixel)
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

    template<typename T>
    T amplitude(const T& x,  const T& y) { return sqrt( (x*x) + (y*y) ); }

    template<typename T>
    T direction(const T& x,  const T& y) { return cvFastArctan(y, x); }

    cv::Mat conv2d(cv::Mat& srcImg, const cv::Mat& filter, bool OutputIsImage)
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
                if (OutputIsImage)
                {
                    outImg.at<double>(x, y) = conv(srcImg, filter, cv::Point(x, y)); // value is in [0;255] range
                }
                else
                {
                    outImg.at<double>(x, y) = conv(srcImg, filter, cv::Point(x, y)) / 255; // normalize value to be in [0;1] range
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);

        std::cout << "Image dimensions : " << srcImg.cols << "x" << srcImg.rows << " \n";
        std::cout << "Convolution duration : " << duration.count() << " ms. \n";

        if(OutputIsImage)
        {
            outImg.convertTo(outImg, CV_8UC1);
        }

        return outImg;
    }

    cv::Mat computeDirection(cv::Mat& Gx, cv::Mat& Gy)
    {
        cv::Mat out(Gx.rows, Gx.cols, CV_64F);
        Gx.convertTo(Gx, CV_64F);
        Gy.convertTo(Gy, CV_64F);

        for(int x = 0; x < Gx.rows; ++x)
        {
            for(int y = 0; y < Gx.cols; ++y)
            {
                out.at<double>(x,y) = direction(Gx.at<double>(x,y), Gy.at<double>(x,y));
            }
        }

        return out;
    }

     cv::Mat computeAmplitude(cv::Mat& Gx, cv::Mat& Gy)
     {
         cv::Mat out(Gx.rows, Gx.cols, CV_64F);
         Gx.convertTo(Gx, CV_64F);
         Gy.convertTo(Gy, CV_64F);

         for (int x = 0; x < Gx.rows; ++x)
         {
             for (int y = 0; y < Gx.cols; ++y)
             {
                 out.at<double>(x,y) = amplitude(Gx.at<double>(x,y), Gy.at<double>(x,y));
             }
         }

         return out;
     }
    
    cv::Mat globalThreshold(cv::Mat& imAmplitude, double threshold)
    {
        cv::Mat outImg(imAmplitude.rows, imAmplitude.cols, CV_8UC1);

        for(int x = 0; x < imAmplitude.rows; ++x)
        {
            for(int y = 0; y < imAmplitude.cols; ++y)
            {
                double& value = imAmplitude.at<double>(x, y);
                unsigned char &grayValue = outImg.at<uchar>(x, y);
                if(value < threshold)
                {
                    grayValue = 0;
                }
                else 
                {
                    grayValue = 255;
                }
            }

        }

        return outImg;
    }

}

int main()
{
    // Initialize window
    WinProps resultWindow("Result", {800, 600});
    cv::namedWindow(resultWindow.name());

    // Read source image
    cv::Mat img = ImIO::readImg("../../data/img/lena_std.tif");

    // 3x3 directional filters

    cv::Mat gradientX    = (cv::Mat_<double>(3,3) << /* R0 */ -1.0, 0.0, 1.0,   /* R1 */ -1.0, 0.0, 1.0,  /* R2 */-1.0, 0.0, 1.0)  * 1.0 / 3.0; // Prewitt filter
    cv::Mat gradientY    = (cv::Mat_<double>(3,3) << /* R0 */-1.0,-1.0,-1.0,   /* R1 */ 0.0, 0.0, 0.0,  /* R2 */  1.0,  1.0,  1.0) * 1.0 / 3.0;
    cv::Mat gradientDpos = (cv::Mat_<double>(3,3) << /* R0 */ 1.0, 1.0, 0.0,   /* R1 */ 1.0, 0.0, -1.0,  /* R2 */ 0.0, -1.0, -1.0) * 1.0 / 3.0;
    cv::Mat gradientDneg = (cv::Mat_<double>(3,3) << /* R0 */ 0.0, 1.0, 1.0,   /* R1 */-1.0, 0.0, 1.0,  /* R2 */ -1.0, -1.0, 0.0)  * 1.0 / 3.0;

    cv::Mat sobel        = (cv::Mat_<double>(3,3) << /* R0 */ -1.0,  0.0,  1.0, /* R1 */ -2.0, 0.0, 2.0,  /* R2 */ -1.0, 0.0, 1.0)  * 1.0 / 4.0;
    cv::Mat kirsch4      = (cv::Mat_<double>(3,3) << /* R0 */ -3.0, -3.0, -3.0, /* R1 */ 5.0,  0.0, -3.0, /* R2 */ 5.0,  5.0, -3.0) * 1.0 / 15.0;

    // Apply convolution
    cv::Mat outGradientX       = ImOp::conv2d(img, gradientX, false);
    cv::Mat outGradientY       = ImOp::conv2d(img, gradientY, false);
    cv::Mat outGradientDpos    = ImOp::conv2d(img, gradientDpos, false);
    cv::Mat outGradientDneg    = ImOp::conv2d(img, gradientDneg, false);

    cv::Mat outSobel           = ImOp::conv2d(img, sobel, true);
    cv::Mat outKirsch4         = ImOp::conv2d(img, kirsch4, false);


    // Threshold
    cv::Mat amplitudeBiDirectional = ImOp::computeAmplitude(outGradientX, outGradientY);

    cv::Mat outGlobalThreshold     = ImOp::globalThreshold(amplitudeBiDirectional, 0.1);

    // Optional : concatenate results into a single mat
    std::vector<cv::Mat> outMats      {outGradientX, outSobel, outKirsch4, outGradientY };
    std::vector<cv::Mat> outGradients {outGradientX, outGradientY, outGradientDpos, outGradientDneg};

    cv::Mat outGradientsConcat;                            
    cv::hconcat(outGradients, outGradientsConcat); 

    //ImOp::computeDirection(outGradientX, outGradientY);
    // Display result
    imshow(resultWindow.name(), outSobel);
    
    // Wait for a keystroke before terminating
    int key = cv::waitKey(0); 

    return 0;
}



// Links
// https://fr.wikipedia.org/wiki/Filtre_de_Prewitt