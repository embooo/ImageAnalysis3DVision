# TP-AnalyseImage
Image source            |  Suppression des maxima locaux + seuillage par hysteresis
:-------------------------:|:-------------------------:
![alt text](https://github.com/embooo/ImageAnalysis3DVision/blob/main/gallery/gallery_orig.jpg?raw=true) |  ![alt text](https://github.com/embooo/ImageAnalysis3DVision/blob/main/gallery/gallery_hysteresis.png?raw=true)
## Running the program
Executable for TP1 is located in bin/Debug-windows-x86_64/tp1 <br>
To run the program, either  drag and drop an image file into the executable "tp1.exe" or use a command prompt and specify a file e.g tp1.exe <image.format>. <br>
The program is displaying the images of the gradient amplitudes, and different thresholds in each window.

## Features implemented
Source code located in : core/src/TP1

*   Convolution
*   Bi-directional gradient (Amplitude + direction)
*   Multi-directional gradient (Amplitude + direction)
*   Global threshold
*   Local threshold
*   Hysteresis threshold
*   Non-max suppression

## Build project
-------
The project is already built for Visual Studio 2019. <br>
If you need to generate the project files for another IDE : <br>

On Windows, execute buildVS2019.bat to generate the project files for Visual Studio 2019 with premake.
You can modify this script to generate for another IDE : https://premake.github.io/docs/Using-Premake/

Once built, the executable is located in the bin folder at the root of the archive. <br>
If an DLL error happens when executing the program, copy the .dll files located "core/opencv/build/bin/dll/Debug" to the folder containing the executable
