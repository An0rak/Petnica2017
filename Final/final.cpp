#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>
#include <sstream>
#include <fstream>
#include "../Core/core.cpp"

using namespace std;
using namespace cv;


bool loadCameraCalibration(Mat& cameraMatrix, Mat& distortionCoeff)
{
    FileStorage fs("file.yml", FileStorage::READ);
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distortionCoeff"] >> distortionCoeff;
    cameraMatrix.convertTo(cameraMatrix, CV_32F);
    distortionCoeff.convertTo(distortionCoeff, CV_32F);
    fs.release();
}


vector<float> findPlane(Mat img1, Mat img2)
{
    Mat tmp1, tmp2;
    undistort(img1, tmp1, cameraMatrix, distortionCoeff);
    undistort(img2, tmp2, cameraMatrix, distortionCoeff);
    plane = getPlane(tmp1, tmp2, cameraMatrix, distortionCoeff,
                    correctMatchingPoints1, correctMatchingPoints2,
                    correctKeyPoints1, correctKeyPoints2,allCorrectMatches,points3dfirst, points3dsecond);
    return plane;
}

float getDistance(Point3f point1, Point3f point2)
{
	float distance = sqrt( (point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y) + (point1.z - point2.z) * (point1.z - point2.z) );
	return distance;
}

vector<Point3f> getArrow(Point3f point, vector<float> plane)
{
	vector<Point3f> arrow;
	float x = plane[0] / 10000;
	float y = plane[1] / 10000;
	float z = plane[2] / 10000;
	for(int i = 0; i < 10000 ; i++)
	{
		Point3f arrowdot;
		arrowdot.x = point.x + i*x;
		arrowdot.y = point.y + i*y;
		arrowdot.z = point.z + i*z;
		arrow.push_back(arrowdot);
	}
	return arrow;
}

int main(int argc, char** argv)
{
	FileStorage fw("Results.yml", FileStorage::WRITE);
	Mat loadimg1 = imread("Images//img1-0.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat loadimg2 = imread("Images//img1-1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat loadimg3 = imread("Images//img1-2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	loadCameraCalibration(cameraMatrix, distortionCoeff);
	vector<float> ravan  = findPlane(loadimg2, loadimg3);
	Point3f rootOfArrow = points3dfirst[points3dfirst.size()/2];
	cout << rootOfArrow;
	vector<Point3f> arrow = getArrow(rootOfArrow, ravan);
	//fw << "Plane" << ravan;
	//fw << "Object Points" << points3dfirst;
	//fw << "Arrow" << arrow;
	Vec3f rvec(0,0,0);
	Vec3f tvec(0,0,0);
	vector<Point2f> reprojectedArrow;
	projectPoints(arrow,rvec,tvec,cameraMatrix,Mat(),reprojectedArrow);
	Mat tmp1, tmp2;
	undistort(loadimg2, tmp1, cameraMatrix, distortionCoeff);
    undistort(loadimg3, tmp2, cameraMatrix, distortionCoeff);
	//drawPoints(tmp1, reprojectedArrow);
    Mat img_kp1;
    Mat img_kp2;
    //drawMatches(tmp1, correctKeyPoints1, tmp2, correctKeyPoints2, correctMatches, res, Scalar::all(-1),CV_RGB(255,255,255),Mat(),2);
    drawKeypoints(tmp1, correctKeyPoints1, img_kp1);
    drawKeypoints(tmp2, correctKeyPoints2, img_kp2);
    fw << "KP1" << correctKeyPoints1;
    fw << "KP2" << correctKeyPoints2;
    imwrite("Image.jpg", tmp1);
    imshow("keypoints1", img_kp1);
    imshow("keypoints2", img_kp2);
    waitKey(0);

    waitKey();

	return 0;
}
