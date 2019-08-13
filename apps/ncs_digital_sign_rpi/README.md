# ncs_sign

**NCS Smart Digital Sign**

**Details**

    Target OS: Raspian OS
    Programming Language: C++
    Time to Complete: 45 min
    
**Introduction**

    This smart digital signage application is one of a series of reference 
    implementations for Computer Vision (CV) using the Intel Distribution
    of OpenVINO toolkit. This application is designed for a kiosk type display
    that will output relevent advertisements based on the user's demographics. 
    It is intended to provide real-time dynamic advertisement output for 
    kiosk type use cases. 

**Requirements**

    Hardware:
    Raspberry Pi 3
    USB Camera
    
    Software:
    Raspian OS
    Intel Distribution of OpenVINO toolkit 2019 R1 release
    NCS Signage Feature Source Code (patch-2 directory)
    RapidJson Library (patch-2/RapidJson directory)
    
**Setup**

    Install OpenVino Toolkit:
    
    Refer to (https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html) 
    for more information about how to install and setup the Intel Distribution
    of OpenVINO toolkit.
    
    Setup NCS USB Rules:
    
    1) cd /opt/intel/openvino/install_prerequisites
    2) ./install_NCS_udev_rules.sh
    
    Once Intel OpenVINO toolkit is installed and the demos have been verified,
    it is time to setup the interactive_face_detection_demo. 
    
    Building interactive_face_detection_demo:
    
    Navigate to a directory you have write access and create a samples build
    directory
        1) mkdir
        2) cd build
        3) cmake-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=armv7-a" 
        /opt/intel/openvino/deployment_tools/inference_engine/samples DTHREADING=SEQ
        4) make interactive_face_detection 
        
        To run the interactive_face_detection_demo you need to get the pre-trained 
        models that are used in the sample
        Get the pre-trained models from: 
        http://docs.openvinotoolkit.org/latest/_inference_engine_samples_interactive_face_detection_demo_README.html
        
        The interactive_face_detection_demo consists of four pre-trained 
        models: 
        
            1) face-detection-adas-0001, which is a primary detection network 
            for finding faces (REQUIRED)
            2) age-gender-recognition-retail-0013, which is executed on top of 
            the results of the first model and reports estimated age and gender 
            for each detected face (OPTIONAL)
            3) head-pose-estimation-adas-0001, which is executed on top of the 
            results of the first model and reports estimated head pose in 
            Tait-Bryan angles (OPTIONAL)
            4) emotions-recognition-retail-0003, which is executed on top of the 
            results of the first model and reports an emotion for each detected 
            face (OPTIONAL)
            5) facial-landmarks-35-adas-0002, which is executed on top of the 
            results of the first model and reports normed coordinates of estimated 
            facial landmarks (OPTIONAL)
            
    Find the build directory that was created earlier, go into that build folder
        Go into: cd /arm71/Release
        This is the environment where the interactive_face_detection_demo will
        run.
        Run the following command to run the interactive_face_detection_demo
        
        ./interactive_face_detection_demo -i <path_to_video> -m 
        <path_to_model>/face-detection-adas-0001.xml -m_ag <path_to_model>/age-gender
        -recognition-retail-0013.xml -m_em <path_to_model>/emotions-recognition-
        retail-0003.xml
        
**Editing interactive_face_detection_demo**
    
    Go into directory that has Intel OpenVINO interactive_face_detection_demo
    code:cd /opt/intel/openvino/inference_engine/samples/interactive_face_detection_demo
    
    Replace the following files inside that directory with Gitlab files(patch-2 directory):
    
        1) detectors.cpp
        2) visualizer.hpp
        3) visualizer.cpp
        4) main.cpp
        
    Change paths to advertisement images inside of config1.json for "maleMdvertisement"
    and "femaleAdvertisement" (they are marked with <INSERT AD PATH HERE>)
    
    Sample advertisements are located in patch-2/Advertisement_Images directory.
        
    Go into include directory within OpenVINO: cd /opt/intel/openvino/inference_engine
    /include
    
    Paste RapidJSON folder (with this Gitlab) into /include directory
    
    Navigate to build directory that was created earlier, within the build directory
    run the following commands: 
        1) cmake-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=armv7-a" 
        /opt/intel/openvino/deployment_tools/inference_engine/samples DTHREADING=SEQ
        2) make interactive_face_detection
    
    Go into: cd /arm71/Release
        Run the following command to run the interactive_face_detection_demo:
        
        ./interactive_face_detection_demo -i <path_to_video> -m 
        <path_to_model>/face-detection-adas-0001.xml -m_ag <path_to_model>/age-gender
        -recognition-retail-0013.xml -m_em <path_to_model>/emotions-recognition-
        retail-0003.xml
    
    You now have your Digital Signage Demo!
    
    
        
    
    
    
    
    