#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui_c.h> 
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
using namespace cv;
using namespace std;
// 计算可倾斜外接矩形与圆轮廓
void  cacBoundRectRandomDirection(cv::Mat src)
{
	RNG rng(12345);
	Mat threMat;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//  二值化操作
	threshold(src, threMat, 20, 255, THRESH_BINARY);
	// 计算边界轮廓
	findContours(threMat, contours, hierarchy,
		CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	// 对每个找到的轮廓创建可倾斜的边界框和椭圆
	vector<RotatedRect> minRect(contours.size());
	vector<RotatedRect> minEllipse(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		// 计算最小面积矩形
		minRect[i] = minAreaRect(Mat(contours[i]));
		// 求外接轮廓的椭圆
		if (contours[i].size() > 5)
		{
			minEllipse[i] = fitEllipse(Mat(contours[i]));
		}
	}
	// 绘出轮廓及其可倾斜的边界框和边界椭圆
	Mat resultMat = Mat::zeros(threMat.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		// 随机生成颜色
		Scalar color = Scalar(rng.uniform(0, 255),
			rng.uniform(0, 255), rng.uniform(0, 255));
		// 边界轮廓绘制
		drawContours(resultMat, contours, i, color, 1, 8,
			vector<Vec4i>(), 0, Point());
		// 绘制边界椭圆
		ellipse(resultMat, minEllipse[i], color, 2, 8);
		// 绘制边界矩形
		Point2f rect_points[4]; minRect[i].points(rect_points);
		for (int j = 0; j < 4; j++)
			line(resultMat, rect_points[j], rect_points[(j + 1) %
				4], color, 1, 8);
	}
	imshow("resultMat", resultMat);
}

int main() {
	VideoCapture cap("test2.avi");
	if (!cap.isOpened()) {
		cout << "video read failed!" << endl;
		return -1;
	}
	Mat currentFrame;
	Mat lastFrame = imread("first.jpg");
	//medianBlur(lastFrame, lastFrame, 3);
	cvtColor(lastFrame, lastFrame, COLOR_BGR2GRAY);
	//imshow("first", lastFrame);
	//waitKey(10);
	namedWindow("switch", 1);
	while (1) {
		cap >> currentFrame;
		if (currentFrame.empty())break;
		/*medianBlur(currentFrame, currentFrame, 3);*/
		cvtColor(currentFrame, currentFrame, COLOR_BGR2GRAY);
		Mat diffFrame(currentFrame.rows, currentFrame.cols, CV_8UC1, Scalar(0));
		absdiff(currentFrame, lastFrame, diffFrame);
		Mat blurMat;
		blur(diffFrame, blurMat, Size(3, 3));
		cacBoundRectRandomDirection(blurMat);
		//threshold(diffFrame, diffFrame, 20, 255, THRESH_BINARY);
		//Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
		//Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
		//erode(diffFrame, diffFrame, element2);
		//dilate(diffFrame, diffFrame, element1);
		imshow("switch", diffFrame);
		//imshow("switch2", currentFrame);
		waitKey(20);
	}
	return 0;
}