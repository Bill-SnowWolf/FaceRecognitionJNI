/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class edu_carleton_comp4601_finalproject_core_OpenCV */

#ifndef _Included_edu_carleton_comp4601_finalproject_core_OpenCV
#define _Included_edu_carleton_comp4601_finalproject_core_OpenCV
#ifdef __cplusplus
extern "C" {
#endif
    /*
     * Class:     edu_carleton_comp4601_finalproject_core_OpenCV
     * Method:    faceDetection
     * Signature: (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V
     */
    JNIEXPORT void JNICALL Java_edu_carleton_comp4601_finalproject_core_OpenCV_faceDetection
    (JNIEnv *, jclass, jstring, jstring, jstring);
    
    /*
     * Class:     edu_carleton_comp4601_finalproject_core_OpenCV
     * Method:    faceRecognization
     * Signature: (Ljava/lang/String;[Ljava/lang/String;Ljava/lang/String;I)I
     */
    JNIEXPORT jint JNICALL Java_edu_carleton_comp4601_finalproject_core_OpenCV_faceRecognization
    (JNIEnv *, jclass, jstring, jobjectArray, jstring, jint);
    
    /*
     * Class:     edu_carleton_comp4601_finalproject_core_OpenCV
     * Method:    testSingleFace
     * Signature: ([Ljava/lang/String;I)I
     */
    JNIEXPORT jint JNICALL Java_edu_carleton_comp4601_finalproject_core_OpenCV_testSingleFace
    (JNIEnv *, jclass, jobjectArray, jint);
    
#ifdef __cplusplus
}
#endif
#endif
