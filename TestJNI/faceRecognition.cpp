#include "faceRecognition.h"
#include <string.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


// Native Implementation
Mat cropImage(Mat bigImage, Rect rect) {
    Mat smallerImage = Mat(bigImage, rect).clone();
    return smallerImage;
}

Mat faceDetection(const char* filename, const char* xmlPath, bool rsize) {
    cout << "Face File Name: " << filename << endl;
    
    Mat image;
    image = imread( filename, 0 );
    
    CascadeClassifier faceDector;
    
    faceDector.load(xmlPath);
    
    vector<Rect> faces;
    faceDector.detectMultiScale(image, faces);
    
    Rect cropRect = Rect(faces[0].x, faces[0].y, faces[0].width, faces[0].height);
    Mat headImage = cropImage(image, cropRect);
    
    if (rsize) {
        Size size(50, 50);
        resize(headImage, headImage, size);
    }
    
    return headImage;
}

void saveImage(Mat image, char* filename) {
    imwrite(filename, image);
}

// JNI Interface

JNIEXPORT void JNICALL Java_edu_carleton_comp4601_finalproject_core_OpenCV_faceDetection
(JNIEnv * env, jobject obj, jstring xmlPath, jstring inPath, jstring outPath) {
    const char *xmlFilePath = env->GetStringUTFChars(xmlPath, 0);
    const char *inputImagePath = env->GetStringUTFChars(inPath, 0);
    const char *outputImagePath = env->GetStringUTFChars(outPath, 0);

    printf("XML Path: %s\n", xmlFilePath);
    printf("Input Image Path: %s\n", inputImagePath);
    printf("Output Image Path: %s\n", outputImagePath);
    
    Mat face = faceDetection(inputImagePath, xmlFilePath, false);
    imwrite(outputImagePath, face);

    
    env->ReleaseStringUTFChars(xmlPath, xmlFilePath);
    env->ReleaseStringUTFChars(inPath, inputImagePath);
    env->ReleaseStringUTFChars(outPath, outputImagePath);
}

JNIEXPORT jint JNICALL Java_edu_carleton_comp4601_finalproject_core_OpenCV_faceRecognization
(JNIEnv * env, jclass obj, jstring xmlPath, jobjectArray trainingImagePaths, jstring testImagePath, jint minConfidence) {
    int len = env->GetArrayLength(trainingImagePaths);
    
    vector<string> trainingImageNames;
    
    for (int i=0;i<len;i++) {
        jstring body = (jstring)env->GetObjectArrayElement(trainingImagePaths, i);
        trainingImageNames.push_back(env->GetStringUTFChars(body, 0));
    }
    
    const char *xmlFilePath = env->GetStringUTFChars(xmlPath, 0);
    const char *testImageName = env->GetStringUTFChars(testImagePath, 0);
    
    vector<Mat> trainImages;
    vector<int> trainLabels;
    
    for (int i=0;i<trainingImageNames.size();i++) {
        trainImages.push_back(faceDetection(trainingImageNames[i].c_str(), xmlFilePath, true));
        trainLabels.push_back(i);
    }
    
    Mat testImage = faceDetection(testImageName, xmlFilePath, true);
    
    Ptr<FaceRecognizer> model;
    
    // For smaller dataset, we use fisher face algorithm
    // For larger dataset, we use eigen face algorithm
    if (trainingImageNames.size() < 7 && trainingImageNames.size() > 2) {
        model = createFisherFaceRecognizer();
    } else {
        model = createEigenFaceRecognizer();
    }
    model->train(trainImages, trainLabels);
    
    int predictedLabel = -1;
    double confidence = 0.0;
    model->predict(testImage, predictedLabel, confidence);
    
    string result_message = format("Predicted class = %d, %f", predictedLabel, confidence);
    cout << result_message << endl;

    
    env->ReleaseStringUTFChars(xmlPath, xmlFilePath);
    env->ReleaseStringUTFChars(testImagePath, testImageName);
    
    // If the minConfidence is -1, then we do not care about confidence value
    if (minConfidence < 0 || confidence < minConfidence)
        return predictedLabel;
    else
        return -1;
}


int main() {
    return 0;
}

