workspace "M2AnalyseImage"
    architecture "x64"
    startproject "tp1"

    configurations
    {
        "Debug",
        "Release"
    }

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

project "tp1"
    location "projects/%{prj.name}"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++11"

    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

    files
    {
        "core/src/%{prj.name}/**.h",
        "core/src/%{prj.name}/**.cpp"
    }

    includedirs
    {
        
        "core/opencv/sources",
        "core/opencv/sources/include",
        "core/opencv/build/include",
        "core/opencv/sources/modules/core/include",
        "core/opencv/sources/modules/highgui/include",
        "core/opencv/sources/modules/imgcodecs/include",
        "core/opencv/sources/modules/videoio/include",
        "core/opencv/sources/modules/imgproc/include"
    }

    libdirs 
    { 
        "core/opencv/build/bin/lib/Debug" 
    }

    links 
	{ 
			"opencv_core343d",
            "opencv_highgui343d",
            "opencv_imgcodecs343d",
            "opencv_imgproc343d",
            "opencv_videoio343d"
    }
	
	
    filter "configurations:Debug"
        symbols "on"