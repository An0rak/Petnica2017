#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <random>
#include <time.h>

using namespace std;
using namespace cv;


//Drawing of points_____________________________________________________________________________________________________________________________________________________________________________________________
void drawPoints(Mat& img, vector<Point2f> points)
{
		for(size_t i = 0; i < points.size(); i++)
		{
				circle(img, points[i], 5, CV_RGB(100,200,100), 2, 8, 0);
		}
}



//Drawing of epipolar line_____________________________________________________________________________________________________________________________________________________________________________________

//* \brief Compute and draw the epipolar lines in two images
 //*      associated to each other by a fundamental matrix
 //*
 //* \param F         Fundamental matrix
 //* \param img1      First image
 //* \param img2      Second image
 //* \param points1   Set of points in the first image
 //* \param points2   Set of points in the second image matching to the first set
 //* \param inlierDistance      Points with a high distance to the epipolar lines are
 //*                not displayed. If it is negative, all points are displayed


//Draw Epipolar Lines_______________________________________________________________________________________________________________________________________________________________________________
float distancePointLine(const Point2f point, const Vec<float,3>& line)
{
  //Line is given as a*x + b*y + c = 0
  return fabsf(line(0)*point.x + line(1)*point.y + line(2))
      / std::sqrt(line(0)*line(0)+line(1)*line(1));
}

void cv::computeCorrespondEpilines( InputArray _points, int whichImage,
                                    InputArray _Fmat, OutputArray _lines )
{
    Mat points = _points.getMat(), F = _Fmat.getMat();
    int npoints = points.checkVector(2);
    if( npoints < 0 )
        npoints = points.checkVector(3);
    CV_Assert( npoints >= 0 && (points.depth() == CV_32F || points.depth() == CV_32S));

    _lines.create(npoints, 1, CV_32FC3, -1, true);
    CvMat c_points = points, c_lines = _lines.getMat(), c_F = F;
    cvComputeCorrespondEpilines(&c_points, whichImage, &c_F, &c_lines);
}


void drawEpipolarLines(cv::Mat& F,
                cv::Mat& img1, cv::Mat& img2,
                std::vector<cv::Point2f> points1,
                std::vector<cv::Point2f> points2,
                float inlierDistance = -1)
{
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);

  // * Allow color drawing

  if (img1.type() == CV_8U)
  {
    cv::cvtColor(img1, outImg(rect1), CV_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), CV_GRAY2BGR);
  }
  else
  {
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }
  std::vector<cv::Vec<float,3> > epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);

  CV_Assert(points1.size() == points2.size() &&
        points2.size() == epilines1.size() &&
        epilines1.size() == epilines2.size());

  cv::RNG rng(0);
  for(size_t i=0; i<points1.size(); i++)
  {
    if(inlierDistance > 0)
    {
      if(distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
        distancePointLine(points2[i], epilines1[i]) > inlierDistance)
      {
        //The point match is no inlier
        continue;
      }
    }

    // * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa

    cv::Scalar color(rng(256),rng(256),rng(256));

    Mat tmp2 = outImg(rect2); //non-connst lvalue refernce to type cv::Mat cannot bind to a temporary of type cv::Mat
    Mat tmp1 = outImg(rect1);

    cv::line(tmp2,
      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
      cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
      color);
    cv::circle(tmp1, points1[i], 3, color, -1, CV_AA);

    cv::line(tmp1,
      cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
      cv::Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
      color);
    cv::circle(tmp2, points2[i], 3, color, -1, CV_AA);
  }
  cv::imshow("Epl", outImg);
  cv::waitKey(1);
}

//RANSAC___________________________________________________________________________________________________________________________________________________________________________________________________
float distancePointPlane(const Point3f& point, const vector<float>& plane)
{
      return fabsf(plane[0]*point.x + plane[1]*point.y + plane[2]*point.z + plane[3])
      / sqrt(plane[0]*plane[0] + plane[1]*plane[1] + plane[2] * plane[2]) ;
}

vector<float> getPlaneParams(vector<Point3f>& samplePoints)
{
    vector<float> plane;
    float a = ((samplePoints[0].y - samplePoints[1].y)*(samplePoints[1].z - samplePoints[2].z) - (samplePoints[1].y - samplePoints[2].y)*(samplePoints[0].z - samplePoints[1].z))
                 / ((samplePoints[0].x - samplePoints[1].x)*(samplePoints[1].z - samplePoints[2].z) - (samplePoints[1].x - samplePoints[2].x)*(samplePoints[0].z - samplePoints[1].z));
    plane.push_back(a);
    plane.push_back(-1);
    float c = a*(samplePoints[1].x - samplePoints[0].x) / (samplePoints[0].z - samplePoints[1].z);
    plane.push_back(c);
    plane.push_back(samplePoints[0].y - a*samplePoints[0].x - c*samplePoints[0].z);
    return plane;
}



float calculateError(vector<Point3f>& inliners, vector<float>& model)
{
    float tmp;
    for(size_t i = 0; i <inliners.size(); i++)
    {
        float tmp2 = distancePointPlane(inliners[i], model);
        tmp += tmp2 * tmp2;
    }
    tmp = sqrt(tmp);
    return tmp;
}

vector<Point3f> fisherYatesShuffle(vector<Point3f> data)
{
    vector<Point3f> tmp = data;
    for(int i=0; i < 3; i++)
    {
        int j = rand() % (tmp.size() - i);
        swap(tmp[tmp.size() - 1 - i], data[j]);
    }
    vector<Point3f> result;
    result.push_back(tmp[tmp.size() - 1]);
    result.push_back(tmp[tmp.size() - 2]);
    result.push_back(tmp[tmp.size() - 3]);
    return result;
}

vector<float> ransac(vector<Point3f>& data, int& k, int t, int d)
{
    vector<float> bestfit;
    float besterror = 1000000;
    while(bestfit.size() == 0)
    {
        for(int iterations = 0; iterations < k; iterations++)
        {
            vector<Point3f> maybeinliners = fisherYatesShuffle(data);
            vector<float> maybemodel = getPlaneParams(maybeinliners);
            vector<Point3f> alsoinliners;
            for(size_t i = 0; i < data.size() ; i++)
            {
                if(distancePointPlane(data[i], maybemodel) < t)
                {
                    alsoinliners.push_back(data[i]);
                }
            }
            if(alsoinliners.size() > d)
            {
                float thiserror = calculateError(alsoinliners, maybemodel);
                if(thiserror < besterror)
                {
                    bestfit = maybemodel;
                    besterror = thiserror;
                }
            }
        }
    }
    return bestfit;
}


//Convert From Homogenous_________________________________________________________________________________________________________________________________________________________________________________
vector<Point3f> convertFromHomogenous(Mat points4d)
{
	points4d.convertTo(points4d,CV_32F);
    vector<Point3f> points3d;
    Point3f tmp;
    for(size_t i = 0; i < points4d.cols; i++)
    {
        tmp.x = points4d.at<float>(0,i) / points4d.at<float>(3,i);
        tmp.y = points4d.at<float>(1,i) / points4d.at<float>(3,i);
        tmp.z = points4d.at<float>(2,i) / points4d.at<float>(3,i);
        points3d.push_back(tmp);
    }
    return points3d;
}

bool DecomposeEtoRandT(const Mat_<float>& E,Mat_<float>& R1,Mat_<float>& R2,Mat_<float>& t1,Mat_<float>& t2)
{
	//Using HZ E decomposition
	FileStorage fw("SVD.yml", FileStorage::WRITE);
	fw << "E" << E;
   	SVD svd(E,SVD::MODIFY_A);
	Mat u = svd.u;
	Mat vt = svd.vt;
	u.convertTo(u, CV_32F);
	vt.convertTo(vt, CV_32F);
	fw << "U" << svd.u;
	fw << "VT" <<svd.vt;


   //check if first and second singular values are the same (as they should be)
   	float singular_values_ratio = fabsf(svd.w.at<float>(0) / svd.w.at<float>(1));
   	if(singular_values_ratio > 1) singular_values_ratio = 1 / singular_values_ratio; // flip ratio to keep it [0,1]
   	if (singular_values_ratio < 0.7)
	{
	   	cerr << "singular values of essential matrix are too far apart\n";
	   	return false;
   	}

   	Matx33f W(0,-1,0,   //HZ 9.13
			 1,0,0,
			 0,0,1);
   	Matx33f Wt(0,1,0,
			  -1,0,0,
			  0,0,1);
   	R1 = u * Mat(W) * vt; //HZ 9.19
   	R2 = u * Mat(Wt) * vt; //HZ 9.19
   	t1 = u.col(2); //u3
   	t2 = -u.col(2); //u3
	return true;
}
/*//Using HZ E decomposition
SVD svd(E, SVD::MODIFY_A);
//check if first and second singular values are the same (as they should be)


Matx33f W(0,-1,0,   //HZ 9.13
		  1,0,0,
		  0,0,1);
Matx33f Wt(0,1,0,
		   -1,0,0,
		   0,0,1);
Matx33f Z(0,1,0,
		 -1,0,0,
		  0,0,0);
Mat_<float> U = svd.u.clone();
Mat_<float> Vt = svd.vt.clone();
U.convertTo(U, CV_32F);
Vt.convertTo(Vt, CV_32F);
R1 = U * Mat(W) * Vt; //HZ 9.19
R2 = U * Mat(Wt) * Vt; //HZ 9.19
//t1 = svd.u.col(2); //u3
//t2 = -svd.u.col(2); //u3
t1 = U * Mat(Z) * U.t();
t2 = -t1;
//t1.at<float>(0,0) = T1.at<float>(2,1);
//t1.at<float>(1,0) = T1.at<float>(0,2);
//t1.at<float>(2,0) = T1.at<float>(1,0);
//t2 = -t1;
return true;*/

template<typename T, typename V>
void keepVectorsByStatus(vector<T>& f1, vector<V>& f2, const vector<uchar>& status) {
    vector<T> oldf1 = f1;
    vector<V> oldf2 = f2;
    f1.clear();
    f2.clear();
    for (int i = 0; i < status.size(); ++i) {
        if(status[i])
        {
            f1.push_back(oldf1[i]);
            f2.push_back(oldf2[i]);
        }
    }
}

template<typename T, typename V, typename K>
void keepVectorsByStatus(vector<T>& f1, vector<V>& f2, vector<K>& f3, const vector<uchar>& status) {
    vector<T> oldf1 = f1;
    vector<V> oldf2 = f2;
    vector<K> oldf3 = f3;
    f1.clear();
    f2.clear();
    f3.clear();
    for (int i = 0; i < status.size(); ++i) {
        if(status[i])
        {
            f1.push_back(oldf1[i]);
            f2.push_back(oldf2[i]);
            f3.push_back(oldf3[i]);
        }
    }
}

vector<Point3f> trackedFeatures3D;


bool triangulateAndCheckReproj(const Mat& P, const Mat& P1,
								vector<Point2f> correctMatchingPoints1,vector<Point2f> correctMatchingPoints2,
								Mat cameraMatrix, vector<Point3f>& points3d)
{
	vector<Point2f> trackedFeatures = correctMatchingPoints1;
	vector<Point2f> bootstrap_kp = correctMatchingPoints2;

	//undistort
    FileStorage random("Random.yml", FileStorage::WRITE);
    Mat normalizedTrackedPts,normalizedBootstrapPts;
    undistortPoints(Mat(trackedFeatures), normalizedTrackedPts, cameraMatrix, Mat());
    undistortPoints(Mat(bootstrap_kp), normalizedBootstrapPts, cameraMatrix, Mat());
	random << "Prva" << normalizedTrackedPts;
	random << "Druga" << normalizedBootstrapPts;
    //triangulate
    Mat pt_3d_h(4,trackedFeatures.size(),CV_32F);
    triangulatePoints(P,P1,normalizedTrackedPts,normalizedBootstrapPts,pt_3d_h);
    random << "Aj" << pt_3d_h;
    //Mat pt_3d; convertPointsFromHomogeneous(Mat(pt_3d_h.t()).reshape(4, 1),pt_3d);
	vector<Point3f> points3dp;
	points3dp = convertFromHomogenous(pt_3d_h);
    //    cout << pt_3d.size() << endl;
    //    cout << pt_3d.rowRange(0,10) << endl;
    random << "Aj2" << points3dp;
    vector<uchar> status(points3dp.size(),0);
    for (int i=0; i<points3dp.size(); i++)
	{
        status[i] = (points3dp[i].z > 0) ? 1 : 0;
    }
    int count = countNonZero(status);

    float percentage = ((float)count / (float)points3dp.size());
	random << "percentage" << percentage;
    if(percentage < 0.85)
	{
    	return false; //less than 75% of the points are in front of the camera
	}
	else
	{
		points3d.clear();
    	points3d = convertFromHomogenous(pt_3d_h);
        return true;
	}
    //calculate reprojection
    /*cv::Mat_<float> R = P(cv::Rect(0,0,3,3));
    Vec3d rvec(0,0,0); //Rodrigues(R ,rvec);
    Vec3d tvec(0,0,0);
		//tvec = P.col(3);
    vector<Point2f> reprojected_pt_set1;
    projectPoints(points3dp,rvec,tvec,cameraMatrix,Mat(),reprojected_pt_set1);
    //cout << Mat(reprojected_pt_set1).rowRange(0,10) << endl;
    vector<Point2f> bootstrapPts_v = bootstrap_kp;
    Mat bootstrapPts = Mat(bootstrapPts_v);
	//cout << bootstrapPts.rowRange(0,10) << endl;
    float reprojErr = cv::norm(Mat(reprojected_pt_set1),bootstrapPts,NORM_L2)/(float)bootstrapPts_v.size();
    cout << "reprojection Error " << reprojErr;
    if(reprojErr < 100)
	{
        vector<uchar> status(bootstrapPts_v.size(),0);
        for (int i = 0;  i < bootstrapPts_v.size(); ++ i)
		{
            status[i] = (norm(bootstrapPts_v[i]-reprojected_pt_set1[i]) < 20.0);
    	}

        trackedFeatures3D.clear();
        trackedFeatures3D.resize(points3dp.size());
        trackedFeatures3D = points3dp;
		points3d.clear();
    	points3d = convertFromHomogenous(pt_3d_h);
        keepVectorsByStatus(trackedFeatures,trackedFeatures3D,status);
        return true;
    }
    return false;*/
}


vector<float> transformPlane(Mat R ,Mat T, vector<float> plane, bool inverse)
{
	R.convertTo(R, CV_32F);
	T.convertTo(T, CV_32F);
	Mat Rp(3,3,CV_32F);
	if(inverse)
	{
		Rp = R.inv();
	}
	else
	{
		Rp = R;
	}
	Mat n(3,1,CV_32F);
	Mat N(3,1,CV_32F);
	n.at<float>(0,0) = plane[0];
	n.at<float>(1,0) = plane[1];
	n.at<float>(2,0) = plane[2];
	N = Rp * n;
	vector<float> newPlane;
	newPlane.push_back(N.at<float>(0,0));
	newPlane.push_back(N.at<float>(1,0));
	newPlane.push_back(N.at<float>(2,0));
	float d = plane[3] - T.at<float>(0,0) * plane[0] - T.at<float>(1,0) * plane[1] - T.at<float>(2,0) * plane[2];
	newPlane.push_back(d);
	return newPlane;
}

vector<Point3f> transformPoints(Mat& R, Mat& T, vector<Point3f>& points, bool inverse)
{
	vector<Point3f> outpoints;
	R.convertTo(R, CV_32F);
	T.convertTo(T, CV_32F);
	Mat Rp(3,3,CV_32F);
	if(inverse)
	{
		Rp = R.inv();
	}
	else
	{
		Rp = R;
	}
	Mat tmp(3,1,CV_32F);
	Mat res1(3,1,CV_32F);
	for(int i=0; i < points.size(); i++)
	{
		tmp.at<float>(0,0)= points[i].x;
		tmp.at<float>(1,0)= points[i].y;
		tmp.at<float>(2,0)= points[i].z;

		res1 = Rp * tmp;
		res1.at<float>(0,0) += T.at<float>(0,0);
		res1.at<float>(1,0) += T.at<float>(1,0);
		res1.at<float>(2,0) += T.at<float>(2,0);

		Point3f tmpp;
		tmpp.x = res1.at<float>(0,0);
		tmpp.y = res1.at<float>(1,0);
		tmpp.z = res1.at<float>(2,0);
		outpoints.push_back(tmpp);
	}
	return outpoints;
}

void sintetickiPodaci(Mat cameraMatrix, float t, Mat_<float> wp, vector<Point2f>& points1, vector<Point2f>& points2)
{
  Mat_<float> tmp;
  Mat_<float> res;
  Mat_<float> wpn = wp.clone();
  for(int i=0; i< wp.rows; i++)
  {
    tmp = wp.row(i).t();
    res = cameraMatrix * tmp;
	FileStorage store("test.txt", FileStorage::WRITE);
    points1.push_back(Point2f(res.at<float>(0,0) / res.at<float>(2,0), res.at<float>(1,0) / res.at<float>(2,0)));
  }
  for(int i = 0; i < wpn.rows; i++)
  {
    wpn.at<float>(i,0) -= t;
  }
  for(int i=0; i< wpn.rows; i++)
  {
    tmp = wpn.row(i).t();
    res = cameraMatrix * tmp;
    points2.push_back(Point2f(res.at<float>(0,0) / res.at<float>(2,0), res.at<float>(1,0) / res.at<float>(2,0)));
  }
}




vector<float> getPlane(Mat img1, Mat img2, Mat cameraMatrixread, Mat distortionCoeffread, vector<Point2f>& correctMatchingPoints1, vector<Point2f>& correctMatchingPoints2,
														vector<KeyPoint>& correctKeyPoints1, vector<KeyPoint>& correctKeyPoints2,vector<DMatch>& correctMatches, vector<Point3f>& points3d )
{
	vector<float> plane;
	/*Mat cameraMatrix(3,3,CV_32F);
    Mat distortionCoeff(5,1,CV_32F);

	cameraMatrixread.convertTo(cameraMatrix, CV_32F);
    distortionCoeffread.convertTo(distortionCoeff, CV_32F);

	//Feature Matching______________________________________________________________________________________________________________________________________________________________________________

	    // detecting keypoints

	    Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector(new FastAdjuster(20,true), 0, 1000, 1000));
	    vector<KeyPoint> keypoints1;
	    detector->detect(img1, keypoints1);

	    vector<KeyPoint> keypoints2;
	    detector->detect(img2, keypoints2);



	    // computing descriptors
	    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("FREAK");
	    Mat descriptors1;
	    extractor->compute(img1, keypoints1, descriptors1);



	    Mat descriptors2;
	    extractor->compute(img2, keypoints2, descriptors2);


	    // matching descriptors
	    BFMatcher matcher(NORM_HAMMING, false);
	    vector<DMatch>  matches;
	    matcher.match(descriptors1, descriptors2, matches);


	//Triangulation____________________________________________________________________________________________________________________________________________________________________________

	    vector<Point2f> matchingPoints1;
	    vector<Point2f> matchingPoints2;
	    for(int i=0; i<matches.size(); i++)
	    {
	        matchingPoints1.push_back(keypoints1[matches[i].queryIdx].pt);
	        matchingPoints2.push_back(keypoints2[matches[i].trainIdx].pt);
	    }
*/

		Mat_<float> cameraMatrix = (Mat_<float>(3,3) <<
									4,0,2,
									0,4,2,
									0,0,1);

	    Mat_<float> wp = (Mat_<float>(16  ,3) <<
	                           5000, 1000, 40,
	                           6000, 1000, 20,
	                           7000, 1000, 20,
	                           8000, 1000, 20,
	                           5000, 2000, 20,
	                           6000, 2000, 20,
	                           7000, 2000, 20,
	                           8000, 2000, 20,
	                           5000, 3000, 20,
	                           6000, 3000, 20,
	                           7000, 3000, 20,
	                           8000, 3000, 20,
	                           5000, 4000, 20,
	                           6000, 4000, 20,
	                           7000, 4000, 20,
	                           8000, 4000, 20);

	    sintetickiPodaci(cameraMatrix, -300, wp, correctMatchingPoints1, correctMatchingPoints2);
	    Mat first(1000, 1000, CV_8UC3, Scalar(0, 0, 0));
	    Mat second(1000, 1000, CV_8UC3, Scalar(0, 0, 0));
	    drawPoints(first, correctMatchingPoints1);
	    drawPoints(second, correctMatchingPoints2);
		namedWindow("Window", CV_WINDOW_AUTOSIZE);
		imwrite("first.jpg", first);
		imwrite("second.jpg", second);

	    vector<uchar> status;
	    Mat_<float> fundamentalMatrix = findFundamentalMat(correctMatchingPoints1, correctMatchingPoints2, FM_RANSAC, 1, 0.99, status);

		/*

	    Mat_<float> tmp1(3,1);
	    Mat_<float> tmp2(3,1);
	    Mat_<float> formula;
	    float error = 0;
	    for(int i = 0; i < matches.size(); i++)
	    {
	        if(status[i] != 0)
	        {
	            correctMatchingPoints1.push_back(matchingPoints1[i]);
	            tmp1(0,0)=matchingPoints1[i].x;
	            tmp1(1,0)=matchingPoints1[i].y;
	            tmp1(2,0)=1.0;
	            correctMatchingPoints2.push_back(matchingPoints2[i]);
	            tmp2(0,0)=matchingPoints2[i].x;
	            tmp2(1,0)=matchingPoints2[i].y;
	            tmp2(2,0)=1.0;
	            formula = tmp2.t() * fundamentalMatrix * tmp1;
	            error += formula(0,0);
	        }
	    }


	    //drawEpipolarLines(fundamentalMatrix, img1, img2, correctMatchingPoints1, correctMatchingPoints2, -1);



	    error = error / correctMatchingPoints1.size();

	    printf("%f \n",error);
*/

	    printf("%d \n", correctMatchingPoints1.size());

		vector<KeyPoint> keypointsc1;
	    vector<KeyPoint> keypointsc2;
	    for( size_t i = 0; i < correctMatchingPoints1.size(); i++ )
	    {
	        keypointsc1.push_back(KeyPoint(correctMatchingPoints1[i], 1.f));
	        keypointsc2.push_back(KeyPoint(correctMatchingPoints2[i], 1.f));
	    }

	    /*for(size_t i = 0; i < matches.size(); i++)
	    {
	        if(status[i] != 0)
	        {
	            correctMatches.push_back(matches[i]);
	        }
	    }*/


		Mat_<float> E = cameraMatrix.t() * fundamentalMatrix * cameraMatrix;

	    Mat_<float> R1(3,3);
	    Mat_<float> R2(3,3);
	    Mat_<float> t1(3,3);
	    Mat_<float> t2(3,3);

	    DecomposeEtoRandT(E,R1,R2,t1,t2);

	    if(determinant(R1)+1.0 < 1e-09)
		{
	        E = -E;
	        DecomposeEtoRandT(E,R1,R2,t1,t2);
	    }
		if(fabsf(determinant(R2))-1.0 > 1e-07)
		{
            cerr << "det(R) != +-1.0, this is not a rotation matrix";
            return plane;
		}

		FileStorage fw("Stereo.yml", FileStorage::WRITE);
		fw << "Fundamental" << fundamentalMatrix;
		fw << "Esssential" << E;
		fw << "R1" << R1;
		fw << "R2" << R2;
		fw << "T1" << t1;
		fw << "T1" << t2;
		fw.release();



	    Mat points4d;


	    Mat_<float> P = Mat::eye(3,4,CV_32FC1);

		Mat_<float> P1 = (Mat_<float>(3,4) <<
	    R1.at<float>(0,0),   R1.at<float>(0,1),    R1.at<float>(0,2),    t1.at<float>(0),     //2,1
	    R1.at<float>(1,0),   R1.at<float>(1,1),    R1.at<float>(1,2),    t1.at<float>(1),		//0,2
	    R1.at<float>(2,0),   R1.at<float>(2,1),    R1.at<float>(2,2),    t1.at<float>(2));	//1,0
		if(!triangulateAndCheckReproj(P,P1,correctMatchingPoints1,correctMatchingPoints2, cameraMatrix, points3d))
		{
			P1 = (Mat_<float>(3,4) <<
	        R1.at<float>(0,0),   R1.at<float>(0,1),    R1.at<float>(0,2),    t2.at<float>(0),
	        R1.at<float>(1,0),   R1.at<float>(1,1),    R1.at<float>(1,2),    t2.at<float>(1),
	        R1.at<float>(2,0),   R1.at<float>(2,1),    R1.at<float>(2,2),    t2.at<float>(2));

			if(!triangulateAndCheckReproj(P,P1,correctMatchingPoints1,correctMatchingPoints2, cameraMatrix, points3d))
			{
		    	P1 = (Mat_<float>(3,4) <<
				R2.at<float>(0,0),   R2.at<float>(0,1),    R2.at<float>(0,2),    t2.at<float>(0),
		        R2.at<float>(1,0),   R2.at<float>(1,1),    R2.at<float>(1,2),    t2.at<float>(1),
		        R2.at<float>(2,0),   R2.at<float>(2,1),    R2.at<float>(2,2),    t2.at<float>(2));
				if(!triangulateAndCheckReproj(P,P1,correctMatchingPoints1,correctMatchingPoints2, cameraMatrix, points3d))
				{
		        	P1 = (Mat_<float>(3,4) <<
	                R2.at<float>(0,0),   R2.at<float>(0,1),    R2.at<float>(0,2),    t1.at<float>(0),
	                R2.at<float>(1,0),   R2.at<float>(1,1),    R2.at<float>(1,2),    t1.at<float>(1),
	                R2.at<float>(2,0),   R2.at<float>(2,1),    R2.at<float>(2,2),    t1.at<float>(2));
					if(!triangulateAndCheckReproj(P,P1,correctMatchingPoints1,correctMatchingPoints2, cameraMatrix, points3d))
					{
	                	printf("Nemoguca Traangulacija");
		            }
		        }
			}
		}




	    int datasetSize = correctMatchingPoints1.size();
	    int k = (int)(log(1 - 0.99) / (log(1 - 0.1))) + 1;

	    plane = ransac(points3d, k, 2, datasetSize/2);

	 	return plane;
}



//Main____________________________________________________________________________________________________________________________________________________________________________________________

int mdodosdaain(int argc, char** argv)
{
//Initialization________________________________________________________________________________________________________________________________________________________________________________
		Mat loadimg1 = imread("Images/sveska1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		Mat loadimg2 = imread("Images/sveska2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		Mat img1, img2;

		FileStorage fr("calibration_data.yml", FileStorage::READ);

		Mat cameraMatrix;
		Mat distortionCoeff;

		fr["cameraMatrix"] >> cameraMatrix;
		fr["distortionCoeff"] >> distortionCoeff;

		undistort(loadimg1, img1, cameraMatrix, distortionCoeff);
		undistort(loadimg2, img2, cameraMatrix, distortionCoeff);

		vector<Point2f> correctMatchingPoints1, correctMatchingPoints2;
		vector<KeyPoint> correctKeyPoints1, correctKeyPoints2;
		vector<DMatch> correctMatches;


		//vector<float> plane = getPlane(img1, img2, cameraMatrix, distortionCoeff, correctMatchingPoints1, correctMatchingPoints2, correctKeyPoints1, correctKeyPoints2, correctMatches, points3d);

		VideoCapture vid(1);;
		if(!vid.isOpened())
		{
				return 0;
		}
		namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
		int framesPerSecond = 60;
		Mat frame;
		Mat undistortFrame, finalFrame;
		Mat currentFrame, undistortCurrentFrame;
		Mat oldFrame;
		Mat initMat1, initMat2;
		vector<Point2f> prevPts, nextPts;
		vector<uchar> status;
		vector<float> err;
		int init = 0;
		TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
		Size winSize(31,31);
		vector<float> plane;
		Mat tvector, rvector;
		vector<uchar> inliers;
		vector<Point3f> points3d;

		bool ok = true;
		while(ok)
		{
				if(!vid.read(frame))
				{
						break;
				}
				undistort(frame, undistortFrame, cameraMatrix, distortionCoeff);
				cvtColor(frame, finalFrame, CV_RGB2GRAY);
				imshow("Webcam", frame);
				char character = waitKey(100 / framesPerSecond);
				switch(character)
				{
						case ' ':
								if(init == 0)
								{
										initMat1 = finalFrame.clone();
								}
								if(init == 1)
								{
										initMat2 = finalFrame.clone();
										oldFrame = initMat2.clone();
										prevPts = correctMatchingPoints2;
								}
								if(init == 2)
								{
										currentFrame = finalFrame.clone();
								}
								init++;
								break;
						case 'i':
								plane = getPlane(initMat1, initMat2, cameraMatrix, distortionCoeff, correctMatchingPoints1, correctMatchingPoints2, correctKeyPoints1, correctKeyPoints2, correctMatches, points3d);
								prevPts = correctMatchingPoints2;
								ok = false;
								break;
						case 's':
								if(plane.size() != 0)
								{
										//currentFrame = finalFrame.clone();
										//undistort(currentFrame, undistortCurrentFrame, cameraMatrix, distortionCoeff);
										//calcOpticalFlowPyrLK(oldFrame, undistortCurrentFrame, prevPts, nextPts, status, err, winSize, 3, termcrit, 0, 0.001);
										//solvePnPRansac(points3d, nextPts, cameraMatrix, distortionCoeff, rvector, tvector, true, 6, 5, nextPts.size()/2, inliers, CV_P3P);
								}
								break;
						case 27:
								return 0;
								break;
				}
		}

		undistort(currentFrame, undistortCurrentFrame, cameraMatrix, distortionCoeff);
		calcOpticalFlowPyrLK(oldFrame, undistortCurrentFrame, prevPts, nextPts, status, err, winSize, 3, termcrit, 0, 0.001);
		printf("\n %d \n %d", points3d.size(), nextPts.size());
		solvePnPRansac(points3d, nextPts, cameraMatrix, distortionCoeff, rvector, tvector, true, 6, 5.0, nextPts.size()/2, inliers, CV_P3P);

		FileStorage fw("Vectori", FileStorage::WRITE);
		Mat rMat;
		Rodrigues(rvector, rMat);
		fw << "rMat" << rMat;
		fw << "tvector" << tvector;


		namedWindow("keypoints1", CV_WINDOW_AUTOSIZE);
		namedWindow("keypoints2", CV_WINDOW_AUTOSIZE);
		namedWindow("OF points", CV_WINDOW_AUTOSIZE);
		imshow("keypoints1", initMat1);
		imshow("keypoints2", initMat2);

		/*calcOpticalFlowPyrLK(initMat1, initMat2, correctMatchingPoints1, nextPts, status, err, winSize, 3, termcrit, 0, 0.001);
		drawPoints(initMat1, correctMatchingPoints1);
		Mat tmp = initMat2.clone();
		drawPoints(initMat2, correctMatchingPoints2);
		drawPoints(tmp, nextPts);
		vector<float> plane = getPlane(initMat1, initMat2, cameraMatrix, distortionCoeff, correctMatchingPoints1, correctMatchingPoints2, correctKeyPoints1, correctKeyPoints2, correctMatches, points3d);
		*/
		imshow("keypoints1", initMat1);
		imshow("keypoints2", initMat2);
		//imshow("OF points", tmp);



//Results_________________________________________________________________________________________________________________________________________________________________________________________

		// drawing the results
		namedWindow("matches", 1);

		//drawKeypoints(img1, correctKeyPoints1, img_kp1);
		//drawKeypoints(img2, correctKeyPoints2, img_kp2);



		///drawMatches(img1, correctKeyPoints1, img2, correctKeyPoints2, correctMatches, img_matches);
		printf("pop sere");
		//imshow("keypoints1", img_kp1);
		//imshow("keypoints2", img_kp2);
		//imshow("matches", img_matches);
		waitKey(0);
		return 0;
}
