/*
*   GHANDOURI Féras - 11601442
*   M2 ID3D - 2021/2022
* 
* 
*   Implemented :
*   Convolution
*   Bi-directional gradient (Amplitude + direction)
*   Multi-directional gradient (Amplitude + direction)
* 
*   Global threshold
*   Local threshold
*   Hysteresis threshold
*   Non-max suppression
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <algorithm>

#include <stdio.h>

#include <stdarg.h>

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

        return value;

        // amplitude
        // orientation/pente du gradient = theta arctan(grad(y)/grad(x))
    }

    template<typename T>
    T amplitude(const T& x,  const T& y) { return sqrt( (x*x) + (y*y) ); }

    template<typename T>
    T direction(const T& x,  const T& y) { return cvFastArctan(y, x); }

    /// <summary>
    /// Apply a convolution filter of an input image
    /// </summary>
    /// <param name="srcImg"></param>
    /// <param name="filter"></param>
    /// <param name="OutputIsImage"></param>
    /// <returns></returns>
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

    cv::Mat computeAmplitudeGradient(cv::Mat& G, char dir)
    {
        // Single direction
        cv::Mat out(G.rows, G.cols, CV_64F);
        G.convertTo(G, CV_64F);

        assert((dir == 'X') || (dir == 'Y'));
        for (int x = 0; x < G.rows; ++x)
        {
            for (int y = 0; y < G.cols; ++y)
            {
                if (dir == 'X')
                {
                    out.at<double>(x, y) = amplitude(G.at<double>(x, y), 0.0);
                }
                else 
                {
                    out.at<double>(x, y) = amplitude(0.0, G.at<double>(x, y));
                }
            }
        }

        return out;
    }

    cv::Mat computeGradientDirection(cv::Mat& Gx, cv::Mat& Gy)
    {
        cv::Mat out(Gx.rows, Gx.cols, CV_64F);
        Gx.convertTo(Gx, CV_64F);
        Gy.convertTo(Gy, CV_64F);
        for (int x = 0; x < Gx.rows; ++x)
        {
            for (int y = 0; y < Gx.cols; ++y)
            {
                if (Gx.at<double>(x, y) < 0)
                {
                    out.at<double>(x, y) = -direction(Gx.at<double>(x, y), Gy.at<double>(x, y));
                }
                else
                {
                    out.at<double>(x, y) = direction(Gx.at<double>(x, y), Gy.at<double>(x, y));
                }
            }
        }

        return out;
    }

    cv::Mat computeGradientDirectionMD(cv::Mat& Gx, cv::Mat& Gy, cv::Mat& G_NW, cv::Mat& G_NE)
    {
        cv::Mat out(Gx.rows, Gx.cols, CV_64F);
        Gx.convertTo(Gx, CV_64F);
        Gy.convertTo(Gy, CV_64F);
        for (int x = 0; x < Gx.rows; ++x)
        {
            for (int y = 0; y < Gx.cols; ++y)
            {
                out.at<double>(x, y) = std::max({ Gx.at<double>(x, y), Gy.at<double>(x, y), G_NW.at<double>(x, y), G_NE.at<double>(x, y) });
            }
        }

        return out;
    }

    cv::Mat computeAmplitudeGBD(cv::Mat& Gx, cv::Mat& Gy)
    {
        cv::Mat out(Gx.rows, Gx.cols, CV_64F);
        Gx.convertTo(Gx, CV_64F);
        Gy.convertTo(Gy, CV_64F);
        for(int x = 0; x < Gx.rows; ++x)
        {
            for(int y = 0; y < Gx.cols; ++y)
            {
                out.at<double>(x, y) = amplitude(Gx.at<double>(x, y), Gy.at<double>(x, y));
            }
        }

        return out;
    }

    cv::Mat computeAmplitudeGMD(cv::Mat& Gx, cv::Mat& Gy, cv::Mat& G_NW, cv::Mat& G_NE)
    {
        // Module of multi directional gradient
        // G_NW : gradient in the north-west direction, 3pi/4
        // G_NE : gradient in the north-east direction, pi/4

        cv::Mat out(Gx.rows, Gx.cols, CV_64F);
        Gx.convertTo(Gx, CV_64F);
        Gy.convertTo(Gy, CV_64F);
        G_NW.convertTo(Gx, CV_64F);
        G_NE.convertTo(Gy, CV_64F);


        for (int x = 0; x < Gx.rows; ++x)
        {
            for (int y = 0; y < Gx.cols; ++y)
            {
                out.at<double>(x, y) = std::max({ Gx.at<double>(x, y), Gy.at<double>(x, y), 
                                                  G_NW.at<double>(x, y), G_NE.at<double>(x, y)});
            }
        }

        return out;
    }
    
    /// <summary>
    /// Applies a global threshold to an image based on the value of the gradient at each pixel
    /// </summary>
    /// <param name="imAmplitude"> Matrix containing the amplitude of a the gradient for each pixel </param>
    /// <param name="threshold"> Arbitrary value </param>
    /// <returns> Image with a local threshold applied </returns>
    cv::Mat globalThreshold(const cv::Mat& imAmplitude, double threshold)
    {
        // Return an image
        cv::Mat outImg(imAmplitude.rows, imAmplitude.cols, CV_8UC1);

        for(int x = 0; x < imAmplitude.rows; ++x)
        {
            for(int y = 0; y < imAmplitude.cols; ++y)
            {
                const double& value = imAmplitude.at<double>(x, y);
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
    
    /// <summary>
    /// Applies a local threshold to an image based on the mean of the gradients in the neighborhood of a pixel
    /// </summary>
    /// <param name="imAmplitude"> Matrix containing the amplitude of a the gradient for each pixel </param>
    /// <param name="filterDimension"> Size of the neighborhood to compute the local threshold </param>
    /// <returns> Image with a global threshold applied </returns>
    cv::Mat localThreshold(const cv::Mat& gradientAmp, const cv::Size& filterDimension)
    {
        // Filter dimensions define the extent of the neighborhood 
        // to compute a local threshold
        
        cv::Mat outImg(gradientAmp.rows, gradientAmp.cols, CV_8UC1);
        cv::Mat meanFilter(filterDimension, CV_64F);
        meanFilter = 1.0 / (filterDimension.height * filterDimension.width) ;

        const int filterHalf = floor(filterDimension.width / 2);

        for (int x = filterHalf; x < gradientAmp.rows - filterHalf; ++x)
        {
            for (int y = filterHalf; y < gradientAmp.cols - filterHalf; ++y)
            {
                const double& value      = gradientAmp.at<double>(x, y);
                double localMean         = conv(gradientAmp, meanFilter, cv::Point(x, y));
                unsigned char& grayValue = outImg.at<uchar>(x, y);

                if (value < localMean)
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

    
    cv::Mat hysteresisThreshold(const cv::Mat& gradientAmp, cv::Size neighbordhoodSize, double sh, double factor)
    {
        assert(factor <= 1.0, ("Low threshold is higher than high threshold"));
        // Relation between low and high threshold
        double sb = factor * sh;


        cv::Mat outImg = cv::Mat(gradientAmp.size(), CV_8UC1);
        cv::Mat neighborhood(neighbordhoodSize, CV_64F);
        int half = floor(neighbordhoodSize.width / 2);
        for (int x = half; x < outImg.rows - half; ++x)
        {
            for (int y = half; y < outImg.cols - half; ++y)
            {
                if (gradientAmp.at<double>(x, y) < sb)
                {
                    outImg.at<uchar>(x, y) = 0;
                }
                else if (gradientAmp.at<double>(x, y)  > sh)
                {
                    outImg.at<uchar>(x, y) = 255;
                }
                else
                {
                    // Check neighbors of pixels having an amplitude between sb and sh
                    bool notfound = true;
                    for (int fx = -half; fx <= half; ++fx)
                    {
                        for (int fy = -half; fy <= half; ++fy)
                        {
                            if (gradientAmp.at<double>(x + fx, y + fy)  > sh)
                            {
                                outImg.at<uchar>(x + fx, y + fy) = 255;
                                notfound = false;
                            }
                            else
                            {
                                outImg.at<uchar>(x, y) = 0;
                            }
                            if (!notfound) break;
                        }
                        if (!notfound) break;
                    }
                }
            }
        }

        return outImg;
    }

    
    // Non-max suppression
    cv::Mat affinage(const cv::Mat& gradientAmp, const cv::Mat& gradientDir)
    {
        cv::Mat outImg = cv::Mat(gradientAmp.size(), CV_64F);
        
        for (int x = 1; x < outImg.rows - 1; ++x)
        {
            for (int y = 1; y < outImg.cols - 1; ++y)
            {
                double angle = gradientDir.at<double>(x, y);
                if (angle < 0) angle += 180.0;
                double q = 1.0;
                double r = 1.0;

                // select the neighbors in the direction of gradient
                if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180))
                {
                    q = gradientAmp.at<double>(x, y + 1);
                    r = gradientAmp.at<double>(x, y - 1);
                }
                else if (22.5 <= angle && angle < 67.5)
                {
                    q = gradientAmp.at<double>(x + 1, y - 1);
                    r = gradientAmp.at<double>(x - 1, y + 1);
                }
                else if (67.5 <= angle && angle < 112.5)
                {
                    q = gradientAmp.at<double>(x + 1, y);
                    r = gradientAmp.at<double>(x - 1, y);
                }
                else if (112.5 <= angle && angle < 157.5)
                {
                    q = gradientAmp.at<double>(x - 1, y - 1);
                    r = gradientAmp.at<double>(x + 1, y + 1);
                }


                // check if central pixel is local max
                double centralVal = gradientAmp.at<double>(x, y)  ;
                if (centralVal >= q && centralVal >= r)
                {
                    outImg.at<double>(x, y) = centralVal ;
                }
                else
                {
                    outImg.at<double>(x, y) = 0.0;
                }

            }

        }


        return outImg;
    }


}

int main(int argc, char* argv[])
{
    // Read file
    if (argc < 2)
    {
        std::cerr << "Usage : " << argv[0] << " g | t <image.format> ." << std::endl;
    }
    else
    {
        // Initialize window
        WinProps resultWindow("Result", { 800, 600 });
        cv::namedWindow(resultWindow.name());

        // Read source image
        cv::Mat img = ImIO::readImg(argv[1]);
        cv::Mat orig;
        img.copyTo(orig);
        // 3x3 directional filters

        cv::Mat gradientX = (cv::Mat_<double>(3, 3) << /* R0 */ -1.0, 0.0, 1.0,   /* R1 */ -1.0, 0.0, 1.0,  /* R2 */-1.0, 0.0, 1.0) * 1.0 / 3.0;
        cv::Mat gradientY = (cv::Mat_<double>(3, 3) << /* R0 */-1.0, -1.0, -1.0,   /* R1 */ 0.0, 0.0, 0.0,  /* R2 */  1.0, 1.0, 1.0) * 1.0 / 3.0;
        cv::Mat gradientNE = (cv::Mat_<double>(3, 3) << /* R0 */ 1.0, 1.0, 0.0,   /* R1 */ 1.0, 0.0, -1.0,  /* R2 */ 0.0, -1.0, -1.0) * 1.0 / 3.0;
        cv::Mat gradientNW = (cv::Mat_<double>(3, 3) << /* R0 */ 0.0, 1.0, 1.0,   /* R1 */-1.0, 0.0, 1.0,  /* R2 */ -1.0, -1.0, 0.0) * 1.0 / 3.0;

        cv::Mat sobel = (cv::Mat_<double>(3, 3) << /* R0 */ -1.0, 0.0, 1.0, /* R1 */ -2.0, 0.0, 2.0,  /* R2 */ -1.0, 0.0, 1.0) * 1.0 / 4.0;
        cv::Mat kirsch4 = (cv::Mat_<double>(3, 3) << /* R0 */ -3.0, -3.0, -3.0, /* R1 */ 5.0, 0.0, -3.0, /* R2 */ 5.0, 5.0, -3.0) * 1.0 / 15.0;

        // Apply convolution
        cv::Mat outGradientX  = ImOp::conv2d(img, gradientX, false);
        cv::Mat outGradientY  = ImOp::conv2d(img, gradientY, false);
        cv::Mat outGradientNW = ImOp::conv2d(img, gradientNW, false); // north west direction, pi/4
        cv::Mat outGradientNE = ImOp::conv2d(img, gradientNE, false); // north west direction, 3pi/4

        cv::Mat outSobel   = ImOp::conv2d(img, sobel, true);
        cv::Mat outKirsch4 = ImOp::conv2d(img, kirsch4, false);


        // Threshold
        cv::Mat amplitudeGradientX        = ImOp::computeAmplitudeGradient(outGradientX, 'X');
        cv::Mat amplitudeGradientY        = ImOp::computeAmplitudeGradient(outGradientY, 'Y');
        cv::Mat amplitudeBiDirectional    = ImOp::computeAmplitudeGBD(outGradientX, outGradientY);
        cv::Mat amplitudeMultiDirectional = ImOp::computeAmplitudeGMD(outGradientX, outGradientY, outGradientNW, outGradientNE);


        cv::Mat outGlobalThreshold = ImOp::globalThreshold(amplitudeBiDirectional, 0.1);
        cv::Mat outLocalThreshold  = ImOp::localThreshold(amplitudeBiDirectional, cv::Size(11, 11));
        cv::Mat outHysteresis      = ImOp::hysteresisThreshold(amplitudeBiDirectional, cv::Size(3, 3), 0.13, 0.5);

        // Optional : concatenate results into a single mat
        // Filters
        std::vector<cv::Mat> outMats{ outGradientX, outSobel, outKirsch4, outGradientY };
        // Gradient amplitude values
        std::vector<cv::Mat> outGradients{ amplitudeGradientX, amplitudeGradientY, amplitudeBiDirectional, amplitudeMultiDirectional };

        // Affinage
        cv::Mat directionBiDirectional = ImOp::computeGradientDirection(outGradientX, outGradientY);
        cv::Mat affin = ImOp::affinage(amplitudeBiDirectional, directionBiDirectional);
        cv::Mat affinThresh = ImOp::hysteresisThreshold(affin, cv::Size(3, 3), 0.13, 0.5);

        // Thresholding
        std::vector<cv::Mat> outThresholds{ outGlobalThreshold, outLocalThreshold, outHysteresis, affinThresh };

        cv::Mat outGradientsConcat;
        cv::hconcat(outGradients, outGradientsConcat);

        cv::Mat outThresholdsConcat;
        cv::hconcat(outThresholds, outThresholdsConcat);

        // Display original image
        imshow("Source Image", orig);

        // Display result

        imshow("Amplitude Gradient X", amplitudeGradientX);
        imshow("Amplitude Gradient Y", amplitudeGradientY);
        imshow("Amplitude Bidirectional", amplitudeBiDirectional);
        imshow("Amplitude Multi-directional", amplitudeMultiDirectional);

        imshow("Global thresholding", outGlobalThreshold);
        imshow("Local Thresholding", outLocalThreshold);
        imshow("Hysteresis Thresholding", outHysteresis);
        imshow("Non-max suppression + Hysteresis", affinThresh);

        //imshow("Thresholds [Global ; Local; Hysteresis ; Non-Max Suppression + Hysteresis]", outThresholdsConcat);

        // Wait for a keystroke before terminating
        int key = cv::waitKey(0);
    }



    return 0;
}



// Links
// https://fr.wikipedia.org/wiki/Filtre_de_Prewitt