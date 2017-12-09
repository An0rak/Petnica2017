#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>
#include <sstream>
#include <fstream>
#include "../Final/final.cpp"

using namespace std;
using namespace cv;


const float calibrationSquareDimension = 1.0;
const Size chessBoardDimension = Size(9, 6);
// Neophodne velicine
Mat cameraMatrix(3,3,CV_32F);
Mat distortionCoeff(5,1,CV_32F);
vector<float> plane;
vector<Point2f> correctMatchingPoints1, correctMatchingPoints2;
vector<KeyPoint> correctKeyPoints1, correctKeyPoints2;
vector<DMatch> allCorrectMatches;
vector<Point3f> points3d;
Mat rvec(3,1,CV_32F);
Mat tvec(3,1,CV_32F);
Mat rmat(3,1,CV_32F);
vector<Point3f> chessboardPoints;
vector<float> chessboardplane;


Mat rodrigez(Mat rvec)
{

    float theta = sqrt(rvec.at<float>(0,0)*rvec.at<float>(0,0) + rvec.at<float>(1,0)*rvec.at<float>(1,0) + rvec.at<float>(2,0)*rvec.at<float>(2,0));
    Mat vec(3,1,CV_32F);
    vec.at<float>(0,0) = rvec.at<float>(0,0) / theta;
    vec.at<float>(1,0) = rvec.at<float>(1,0) / theta;
    vec.at<float>(2,0) = rvec.at<float>(2,0) / theta;
    float ux = vec.at<float>(0,0);
    float uy = vec.at<float>(1,0);
    float uz = vec.at<float>(2,0);
    Mat rmat(3,3,CV_32F);
    rmat.at<float>(0,0) = cos(theta) + ux*ux*(1-cos(theta));
    rmat.at<float>(0,1) = ux*uy*(1-cos(theta)) - uz * sin(theta);
    rmat.at<float>(0,2) = ux*uz*(1-cos(theta)) + uy * sin(theta);
    rmat.at<float>(1,0) = uy*ux*(1-cos(theta)) + uz * sin(theta);
    rmat.at<float>(1,1) = cos(theta) + uy*uy*(1-cos(theta));
    rmat.at<float>(1,2) = uy*uz*(1-cos(theta)) - ux * sin(theta);
    rmat.at<float>(2,0) = uz*ux*(1-cos(theta)) - uy * sin(theta);
    rmat.at<float>(2,1) = uz*uy*(1-cos(theta)) + ux * sin(theta);
    rmat.at<float>(2,2) = cos(theta) + uz*uz*(1-cos(theta));
    return rmat;
}

bool loadCameraCalibration(Mat& cameraMatrix, Mat& distortionCoeff)
{
    FileStorage fs("file.yml", FileStorage::READ);
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distortionCoeff"] >> distortionCoeff;
    cameraMatrix.convertTo(cameraMatrix, CV_32F);
    distortionCoeff.convertTo(distortionCoeff, CV_32F);
    fs.release();
}

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
    for(int i = 0; i<boardSize.height; i++)
    {
        for(int j = 0 ; j<boardSize.width; j++)
        {
            corners.push_back(Point3f((float)(j * squareEdgeLength), (float)(i * squareEdgeLength), 0.0));
        }
    }
}

void getChessBoardCorners(Mat image, vector<Point2f>& allFoundCorners, bool showResult = false)
{
        bool found = findChessboardCorners(image, Size(9,6), allFoundCorners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

        if(!found)
        {
            printf("Chess board missing!\n");
        }

}

void getRelativeCameraLocation(Mat image, Mat& tvec, Mat& rvec)
{

    vector<Point2f>  checkerboardImageSpacePoints;
    getChessBoardCorners(image, checkerboardImageSpacePoints, false);
    vector<Point2f> checkerboardImageSpacePointsinNewCS;

    vector<Point3f> worldSpaceCornerPoints;
    createKnownBoardPosition(chessBoardDimension, calibrationSquareDimension, worldSpaceCornerPoints);

    vector<int> inliers;
    solvePnPRansac(worldSpaceCornerPoints, checkerboardImageSpacePoints, cameraMatrix, distortionCoeff, rvec, tvec, false, 100, 1, worldSpaceCornerPoints.size()/2, inliers, CV_ITERATIVE);
    //solvePnP(worldSpaceCornerPoints, checkerboardImageSpacePoints, cameraMatrix, distortionCoeff, rvec, tvec, false, CV_ITERATIVE);
}


void getCameraPosition(Mat img, Mat& tvec, Mat& rmat)
{
    Mat tmp;
    undistort(img, tmp, cameraMatrix, distortionCoeff);
    getRelativeCameraLocation(tmp, tvec, rvec);
    tvec.convertTo(tvec, CV_32F);
    rvec.convertTo(rvec, CV_32F);
    rmat = rodrigez(rvec);
    FileStorage fs("Vektori.yml", FileStorage::WRITE);
    fs << "RMAT" << rmat;
    fs << "RVEC" << rvec;
    fs << "TVEC" << tvec;
    fs.release();
}
vector<float> findPlane(Mat img1, Mat img2)
{
    Mat tmp1, tmp2;
    undistort(img1, tmp1, cameraMatrix, distortionCoeff);
    undistort(img2, tmp2, cameraMatrix, distortionCoeff);
    plane = getPlane(tmp1, tmp2, cameraMatrix, distortionCoeff,
                    correctMatchingPoints1, correctMatchingPoints2,
                    correctKeyPoints1, correctKeyPoints2,allCorrectMatches,points3d);
    FileStorage fs("Plane.yml", FileStorage::WRITE);
    fs << "Plane" << plane;
    fs << "Plane Points" << points3d;
    fs.release();
    return plane;
}

void registerImages()
{
    vector<float> chessboardmodel {0,0,1,0};
    vector<float> chessboard = transformPlane(rmat, tvec, chessboardmodel, false);
    vector<Point3f> chessboardPointsmodel;
    createKnownBoardPosition(chessBoardDimension, calibrationSquareDimension, chessboardPointsmodel);
    chessboardPoints = transformPoints(rmat, tvec, chessboardPointsmodel, false);
    FileStorage fs("Rezultati.yml", FileStorage::WRITE);
    fs << "ChessBoard Plane" << chessboard;
    fs << "ChessBoard Points" << chessboardPoints;
    fs << "Object Plane" << plane;
    fs << "Object Points" << points3d;
    fs << "ChessBoard Model" << chessboardPointsmodel;
}


int main(int argc, char** argv)
{
    Mat frame;
    Mat frameCorners;

    loadCameraCalibration(cameraMatrix, distortionCoeff);

    /*VideoCapture vid(1);

    if(!vid.isOpened())
    {
        return 0;
    }

    int framesPerSecond = 30;
    namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

    Mat img;

    int imgs = 1;
    int count = 0;

    Mat loadimg1, loadimg2, loadimg3;
    while(true)
    {
         if(!vid.read(frame))
         {
             break;
         }
         vector<Vec2f> foundPoints;
         bool found = findChessboardCorners(frame, chessBoardDimension, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
         frame.copyTo(frameCorners);
         drawChessboardCorners(frameCorners, chessBoardDimension, foundPoints, found);

         if(found)
          {
             imshow("Webcam", frameCorners);
         }
         else
         {
             imshow("Webcam", frame);
         }

         char character = waitKey(100 / framesPerSecond);
         switch(character)
         {
             case ' ':
                img = frame;
                imwrite("Images//img" + to_string(imgs) + "-" + to_string(count) + ".jpg",img);
                count++;
                if(count == 1)
                {
                    getCameraPosition(img, rmat, tvec);

                }
                count++;
                if(count == 3)
                {
                    Mat loadimg1 = imread("Images//img" + to_string(imgs) + "-2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
                    Mat loadimg2 = imread("Images//img" + to_string(imgs) + "-3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
                    findPlane(loadimg1, loadimg2);
                    count = 0;
                    imgs++;
                }
                break;
             case 27:
                //exit
                return 0;
                break;
              case 'q':
                registerImages();
                break;
         }
    }*/
    FileStorage fs("Sinteticki.yml", FileStorage::WRITE);
    namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
    Mat loadimg1 = imread("Images//img1-1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat loadimg2 = imread("Images//img1-2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat loadimg3 = imread("Images//img1-3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    imshow("Webcam", loadimg1);
    waitKey(0);
    //getCameraPosition(loadimg1, tvec, rmat);
    findPlane(loadimg2, loadimg3);
    fs << "Points" << points3d;
    //registerImages();
    cout << "Finish";
    return 0;
}
