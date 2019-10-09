/**
 * @file      main.cpp
 * @brief     SfM starting point
 * @authors   Vaibhav Arcot
 * @date      2019
 * @copyright University of Pennsylvania
 */
 // Useful places to look
// https://github.com/LiangliangNan/Easy3D
// https://docs.opencv.org/master/d4/d18/tutorial_sfm_scene_reconstruction.html
// Inverse: https://github.com/md-akhi/Inverse-matrix/blob/master/Inverse-matrix.cpp


 // Steps:
 // SIFT
 // Feature matching (hard)
 // RANSAC
 // Estimate Fundamental matrix matrix 
 // Find epipolar lines
 // Estimate Pose
 // Visulize


#include <iostream>
#include <fstream>
#include <CudaSift/cudaSift.h>
#include <CudaSift/cudaImage.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>  
#include <cmath>
#include <iomanip>
#include <cusolver_common.h>
#include "SfM/kernals.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);
void showCorrespondence(SiftData &siftData1, SiftData &siftData2, cv::Mat limg_0, cv::Mat rimg_0);
double ScaleUp(CudaImage &res, CudaImage &src);

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	int devNum = 0, imgSet = 0;
	if (argc > 1)
		devNum = std::atoi(argv[1]);
	if (argc > 2)
		imgSet = std::atoi(argv[2]);

	// Read images using OpenCV
	cv::Mat limg, rimg;
	cv::imread("../data/dino/viff.000.ppm", 0).convertTo(limg, CV_32FC1);
	cv::imread("../data/dino/viff.001.ppm", 0).convertTo(rimg, CV_32FC1);
	//cv::flip(limg, rimg, -1);
	unsigned int w = limg.cols;
	unsigned int h = limg.rows;
	std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

	// Initial Cuda images and download images to device
	std::cout << "Initializing data..." << std::endl;
	InitCuda(devNum);
	CudaImage img1, img2;
	img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)limg.data);
	img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)rimg.data);
	img1.Download();
	img2.Download();

	// Extract Sift features from images
	SiftData siftData1, siftData2;
	float initBlur = 1.5f;
	float thresh = 1.0f;
	InitSiftData(siftData1, 32768, true, true);
	InitSiftData(siftData2, 32768, true, true);

	// A bit of benchmarking 
	//for (int thresh1=1.00f;thresh1<=4.01f;thresh1+=0.50f) {
	float *memoryTmp = AllocSiftTempMemory(w, h, 5, false);
	ExtractSift(siftData1, img1, 5, initBlur, thresh, 0.0f, false, memoryTmp);
	ExtractSift(siftData2, img2, 5, initBlur, thresh, 0.0f, false, memoryTmp);
	FreeSiftTempMemory(memoryTmp);

	// Match Sift features and find a homography
	MatchSiftData(siftData1, siftData2);
	/*float homography[9];
	int numMatches;
	FindHomography(siftData1, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0);
	int numFit = ImproveHomography(siftData1, homography, 5, 0.00f, 0.80f, 2.0);

	std::cout << "Number of original features: " << siftData1.numPts << " " << siftData2.numPts << std::endl;
	std::cout << "Number of matching features: " << numFit << " " << numMatches << " " << 100.0f*numFit / std::min(siftData1.numPts, siftData2.numPts) << "% " << initBlur << " " << thresh << std::endl;
	*/
	// Define kernal matrix
	float K[9] = {2360.0, 0, w/2.0, 
		         0, 2360, h/2.0,
		         0,0,1};
	float inv_K[9] = {1.0 / 2360, 0, -(w/2.0) / 2360,
		           0, 1.0 / 2360, -(h/2.0) / 2360,
					0, 0, 1};
	SfM::Image_pair sfm(K, inv_K, 2, siftData1.numPts);
	sfm.FillXU(siftData1.d_data);
	sfm.estimateE();
	sfm.testSVD();
	//MatchAll(siftData1, siftData2, homography);
	//showCorrespondence(siftData1, siftData2, limg, rimg);
	// Free Sift data from device
	FreeSiftData(siftData1);
	FreeSiftData(siftData2);
}

void estimate_E() {

}

void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography)
{
#ifdef MANAGEDMEM
	SiftPoint *sift1 = siftData1.d_data;
	SiftPoint *sift2 = siftData2.d_data;
#else
	SiftPoint *sift1 = siftData1.h_data;
	SiftPoint *sift2 = siftData2.h_data;
#endif
	int numPts1 = siftData1.numPts;
	int numPts2 = siftData2.numPts;
	int numFound = 0;
#if 0
	homography[0] = homography[4] = -1.0f;
	homography[1] = homography[3] = homography[6] = homography[7] = 0.0f;
	homography[2] = 1279.0f;
	homography[5] = 959.0f;
#endif
	for (int i = 0; i < numPts1; i++) {
		float *data1 = sift1[i].data;
		std::cout << i << ":" << sift1[i].scale << ":" << (int)sift1[i].orientation << " " << sift1[i].xpos << " " << sift1[i].ypos << std::endl;
		bool found = false;
		for (int j = 0; j < numPts2; j++) {
			float *data2 = sift2[j].data;
			float sum = 0.0f;
			for (int k = 0; k < 128; k++)
				sum += data1[k] * data2[k];
			float den = homography[6] * sift1[i].xpos + homography[7] * sift1[i].ypos + homography[8];
			float dx = (homography[0] * sift1[i].xpos + homography[1] * sift1[i].ypos + homography[2]) / den - sift2[j].xpos;
			float dy = (homography[3] * sift1[i].xpos + homography[4] * sift1[i].ypos + homography[5]) / den - sift2[j].ypos;
			float err = dx * dx + dy * dy;
			if (err < 100.0f) // 100.0
				found = true;
			if (err < 100.0f || j == sift1[i].match) { // 100.0
				if (j == sift1[i].match && err < 100.0f)
					std::cout << " *";
				else if (j == sift1[i].match)
					std::cout << " -";
				else if (err < 100.0f)
					std::cout << " +";
				else
					std::cout << "  ";
				std::cout << j << ":" << sum << ":" << (int)sqrt(err) << ":" << sift2[j].scale << ":" << (int)sift2[j].orientation << " " << sift2[j].xpos << " " << sift2[j].ypos << " " << (int)dx << " " << (int)dy << std::endl;
			}
}
		std::cout << std::endl;
		if (found)
			numFound++;
	}
	std::cout << "Number of finds: " << numFound << " / " << numPts1 << std::endl;
	std::cout << homography[0] << " " << homography[1] << " " << homography[2] << std::endl;//%%%
	std::cout << homography[3] << " " << homography[4] << " " << homography[5] << std::endl;//%%%
	std::cout << homography[6] << " " << homography[7] << " " << homography[8] << std::endl;//%%%
}

void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img)
{
	int numPts = siftData1.numPts;
#ifdef MANAGEDMEM
	SiftPoint *sift1 = siftData1.d_data;
	SiftPoint *sift2 = siftData2.d_data;
#else
	SiftPoint *sift1 = siftData1.h_data;
	SiftPoint *sift2 = siftData2.h_data;
#endif
	float *h_img = img.h_data;
	int w = img.width;
	int h = img.height;
	std::cout << std::setprecision(3);
	for (int j = 0; j < numPts; j++) {
		int k = sift1[j].match;
		if (sift1[j].match_error < 5) {
			float dx = sift2[k].xpos - sift1[j].xpos;
			float dy = sift2[k].ypos - sift1[j].ypos;
#if 0
			if (false && sift1[j].xpos > 550 && sift1[j].xpos < 600) {
				std::cout << "pos1=(" << (int)sift1[j].xpos << "," << (int)sift1[j].ypos << ") ";
				std::cout << j << ": " << "score=" << sift1[j].score << "  ambiguity=" << sift1[j].ambiguity << "  match=" << k << "  ";
				std::cout << "scale=" << sift1[j].scale << "  ";
				std::cout << "error=" << (int)sift1[j].match_error << "  ";
				std::cout << "orient=" << (int)sift1[j].orientation << "," << (int)sift2[k].orientation << "  ";
				std::cout << " delta=(" << (int)dx << "," << (int)dy << ")" << std::endl;
			}
#endif
#if 1
			int len = (int)(fabs(dx) > fabs(dy) ? fabs(dx) : fabs(dy));
			for (int l = 0; l < len; l++) {
				int x = (int)(sift1[j].xpos + dx * l / len);
				int y = (int)(sift1[j].ypos + dy * l / len);
				h_img[y*w + x] = 255.0f;
			}
#endif
		}
		int x = (int)(sift1[j].xpos + 0.5);
		int y = (int)(sift1[j].ypos + 0.5);
		int s = std::min(x, std::min(y, std::min(w - x - 2, std::min(h - y - 2, (int)(1.41*sift1[j].scale)))));
		int p = y * w + x;
		p += (w + 1);
		for (int k = 0; k < s; k++)
			h_img[p - k] = h_img[p + k] = h_img[p - k * w] = h_img[p + k * w] = 0.0f;
		p -= (w + 1);
		for (int k = 0; k < s; k++)
			h_img[p - k] = h_img[p + k] = h_img[p - k * w] = h_img[p + k * w] = 255.0f;
	}
	std::cout << std::setprecision(6);
}

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
#ifdef MANAGEDMEM
	SiftPoint *mpts = data.d_data;
#else
	if (data.h_data == NULL)
		return 0;
	SiftPoint *mpts = data.h_data;
#endif
	float limit = thresh * thresh;
	int numPts = data.numPts;
	cv::Mat M(8, 8, CV_64FC1);
	cv::Mat A(8, 1, CV_64FC1), X(8, 1, CV_64FC1);
	double Y[8];
	for (int i = 0; i < 8; i++)
		A.at<double>(i, 0) = homography[i] / homography[8];
	for (int loop = 0; loop < numLoops; loop++) {
		M = cv::Scalar(0.0);
		X = cv::Scalar(0.0);
		for (int i = 0; i < numPts; i++) {
			SiftPoint &pt = mpts[i];
			if (pt.score<minScore || pt.ambiguity>maxAmbiguity)
				continue;
			float den = A.at<double>(6)*pt.xpos + A.at<double>(7)*pt.ypos + 1.0f;
			float dx = (A.at<double>(0)*pt.xpos + A.at<double>(1)*pt.ypos + A.at<double>(2)) / den - pt.match_xpos;
			float dy = (A.at<double>(3)*pt.xpos + A.at<double>(4)*pt.ypos + A.at<double>(5)) / den - pt.match_ypos;
			float err = dx * dx + dy * dy;
			float wei = (err < limit ? 1.0f : 0.0f); //limit / (err + limit);
			Y[0] = pt.xpos;
			Y[1] = pt.ypos;
			Y[2] = 1.0;
			Y[3] = Y[4] = Y[5] = 0.0;
			Y[6] = -pt.xpos * pt.match_xpos;
			Y[7] = -pt.ypos * pt.match_xpos;
			for (int c = 0; c < 8; c++)
				for (int r = 0; r < 8; r++)
					M.at<double>(r, c) += (Y[c] * Y[r] * wei);
			X += (cv::Mat(8, 1, CV_64FC1, Y) * pt.match_xpos * wei);
			Y[0] = Y[1] = Y[2] = 0.0;
			Y[3] = pt.xpos;
			Y[4] = pt.ypos;
			Y[5] = 1.0;
			Y[6] = -pt.xpos * pt.match_ypos;
			Y[7] = -pt.ypos * pt.match_ypos;
			for (int c = 0; c < 8; c++)
				for (int r = 0; r < 8; r++)
					M.at<double>(r, c) += (Y[c] * Y[r] * wei);
			X += (cv::Mat(8, 1, CV_64FC1, Y) * pt.match_ypos * wei);
		}
		cv::solve(M, X, A, cv::DECOMP_CHOLESKY);
	}
	int numfit = 0;
	for (int i = 0; i < numPts; i++) {
		SiftPoint &pt = mpts[i];
		float den = A.at<double>(6)*pt.xpos + A.at<double>(7)*pt.ypos + 1.0;
		float dx = (A.at<double>(0)*pt.xpos + A.at<double>(1)*pt.ypos + A.at<double>(2)) / den - pt.match_xpos;
		float dy = (A.at<double>(3)*pt.xpos + A.at<double>(4)*pt.ypos + A.at<double>(5)) / den - pt.match_ypos;
		float err = dx * dx + dy * dy;
		if (err < limit)
			numfit++;
		pt.match_error = sqrt(err);
	}
	for (int i = 0; i < 8; i++)
		homography[i] = A.at<double>(i);
	homography[8] = 1.0f;
	return numfit;
}

void showCorrespondence(SiftData &siftData1, SiftData &siftData2, cv::Mat limg_0, cv::Mat rimg_0)
{
	int numPts = siftData1.numPts;
	SiftPoint *sift1 = siftData1.h_data;
	SiftPoint *sift2 = siftData2.h_data;

	int w = limg_0.size().width;
	int h = limg_0.size().height;

	cv::resize(rimg_0, rimg_0, cv::Size(w, h));

	cv::Mat img_m = cv::Mat::zeros(h, 2 * w, 0);
	limg_0.copyTo(img_m(cv::Rect(0, 0, w, h)));
	rimg_0.copyTo(img_m(cv::Rect(w, 0, w, h)));

	std::cout << sift1[1].xpos << ", " << sift1[1].ypos << std::endl;
	for (int j = 0; j < numPts; j++)
	{
		int k = sift1[j].match;
		if (sift1[j].match_error < 2)
		{
			cv::circle(img_m, cv::Point(sift1[j].xpos, sift1[j].ypos), 2, cv::Scalar(60, 20, 220), 2);
			cv::circle(img_m, cv::Point(sift1[j].match_xpos + w, sift1[j].match_ypos), 2, cv::Scalar(173, 216, 230), 2);
			cv::line(img_m, cv::Point(sift1[j].xpos, sift1[j].ypos), cv::Point(sift1[j].match_xpos + w, sift1[j].match_ypos), cv::Scalar(0, 255, 0), 1);
		}
	}

	cv::namedWindow("Result");
	cv::resizeWindow("Result", cv::Size(600, 300));
	cv::imshow("Result", img_m);
	cv::waitKey();
}
