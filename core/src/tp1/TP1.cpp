#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <algorithm>

/*
*   GHANDOURI Féras 11601442
*   M2 ID3D
*/

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

// Convolution
template<typename T> 
struct Filter
{
    Filter(int rows, int cols)
        : m_Rows(rows), m_Cols(cols), m_Type(type) {}

    int m_Type;
    int m_Rows;
    int m_Cols;
};

void convolution(cv::Mat& image, const cv::Mat& filter, int* f)
{
    int filterHalfRows = filter.rows / 2;
    int filterHalfCols = filter.cols / 2;
    
	for (int x = filterHalfCols; x < image.cols - filterHalfCols; ++x)
	{

		for (int y = filterHalfRows; y < image.rows - filterHalfRows; ++y)
		{
            double sum = 0;
            // Loop through pixels in image
            for (int startX = x - filterHalfCols; startX < x + filterHalfCols; ++startX)
            {
                for (int startY = y - filterHalfRows; startY < y + filterHalfRows; ++startY)
                {
                   // sum = sum + image.at<uchar>(startY, startX) * f[curr];


                }
                

                
            }


            // Normalize value to be -255 - 255 range
            // Normalize value to be 0 - 255 range if image
            image.at<uchar>(y, x) = std::min(std::max(0, value), 255);
		}
	}

    // amplitude
    // orientation/pente du gradient = theta arctan(grad(y)/grad(x))
}

int main()
{
    // Lire filtre à partir d'un fichier
    // Lire image  à partir d'un fichier
    std::string image_path = "../../data/img/example.jpg";
    WinProps props("Default", {800, 600});

    // 3x3 matrix of double

    int f[9] = { -1, 0, 1,
                 -1, 0, 1,
                 -1, 0, 1 };
    cv::Mat smooth = (cv::Mat_<int>(3, 3) <<
         0, 0, 0,
         0, 1, 0,
         0, 0, 0
        );

    //smooth = smooth * 1.0 / 16.0
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    convolution(img, smooth, f);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    cv::namedWindow (props.name(), cv::WINDOW_NORMAL);
    cv::resizeWindow(props.name(), props.size());
    
    imshow(props.name(), img);

    // Wait for a keystroke in the 
    int key = cv::waitKey(0); 

    return 0;
}