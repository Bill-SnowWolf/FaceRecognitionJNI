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
    Mat image;
    image = imread( filename, 0 );
    
    CascadeClassifier faceDector;
    
    faceDector.load(xmlPath);
    
    vector<Rect> faces;
    faceDector.detectMultiScale(image, faces);
    
    //    cout << "Detected " << faces.size() << " faces\n";
    
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

void showImage(Mat image) {
    if (image.data) {
        namedWindow("Dected Face", WINDOW_AUTOSIZE);
        imshow("Face", image);
        waitKey(0);
    } else {
        printf("No Face Dected\n");
        return;
    }
}

//void loadTrainImages(const vector<string> &imagePaths, vector<Mat> &trainImages, vector<int> &trainLabels) {
//    for (string path: imagePaths) {
//        trainImages.push_back(faceDetection(path, true));
//        trainLabels.push_back(atoi(path.c_str()));
//    }
//}


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
//    showImage(face);
    //
    //    if (argc == 3) {
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
//        cout << trainingImageNames[i] << endl;
    }
    
    const char *xmlFilePath = env->GetStringUTFChars(xmlPath, 0);
    const char *testImageName = env->GetStringUTFChars(testImagePath, 0);
    
    vector<Mat> trainImages;
    vector<int> trainLabels;
    
    for (string path: trainingImageNames) {
        trainImages.push_back(faceDetection(path.c_str(), xmlFilePath, true));
        trainLabels.push_back(atoi(path.c_str()));
    }
    printf("Done.");
    
    Mat testImage = faceDetection(testImageName, xmlFilePath, true);
    
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->train(trainImages, trainLabels);
    
    int predictedLabel = -1;
    double confidence = 0.0;
    model->predict(testImage, predictedLabel, confidence);
    
    string result_message = format("Predicted class = %d, %f", predictedLabel, confidence);
    cout << result_message << endl;

    
    env->ReleaseStringUTFChars(xmlPath, xmlFilePath);
    env->ReleaseStringUTFChars(testImagePath, testImageName);
    
    if (confidence < minConfidence)
        return predictedLabel;
    else
        return -1;
}

JNIEXPORT jint JNICALL Java_edu_carleton_comp4601_finalproject_core_OpenCV_testSingleFace
(JNIEnv * env, jclass obj, jobjectArray jni_trainingImagePaths, jint minConfidence) {
    
    int numImages = env->GetArrayLength(jni_trainingImagePaths);
    if (numImages < 3) {
        cout << "To test on a single face, provide at least three images or more." << endl;
        exit(1);
    }

    // Load array with specified images; use paths as labels
    vector<Mat> trainingImages;
    vector<int> trainingLabels;
    for (int i = 0; i < numImages; ++i) {
        jstring element = (jstring)env->GetObjectArrayElement(jni_trainingImagePaths, i);
        string imagePath = env->GetStringUTFChars(element, 0);
        
        // The AT&T Database of Faces are entirely grayscale and this must be specified.
        trainingImages.push_back(imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE));
        
        // We're testing with one person's face so the label is the same for each image.
        trainingLabels.push_back(0);
    }
    
    // Use last image and label for testing
    Mat testImage = trainingImages[trainingImages.size() - 1];
    int testLabel = trainingLabels[trainingLabels.size() - 1];
    trainingImages.pop_back();
    trainingLabels.pop_back();
    
    // Train recognizer with training data
    Ptr<FaceRecognizer> recognizer = createEigenFaceRecognizer();
    recognizer->train(trainingImages, trainingLabels);
    
    // Ask for prediction using test image
    int predictedLabel = -1;
    double confidence = 0.0;
    recognizer->predict(testImage, predictedLabel, confidence);
    
    string result_message = format("Predicted class = %d / Actual class = %d, %f", predictedLabel, testLabel, confidence);
    cout << result_message << endl;
    
    return confidence < minConfidence ? predictedLabel : -1;
}


int main() {
    return 0;
}

