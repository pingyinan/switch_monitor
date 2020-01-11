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
// �������б��Ӿ�����Բ����
void  cacBoundRectRandomDirection(cv::Mat src)
{
	RNG rng(12345);
	Mat threMat;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//  ��ֵ������
	threshold(src, threMat, 20, 255, THRESH_BINARY);
	// ����߽�����
	findContours(threMat, contours, hierarchy,
		CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	// ��ÿ���ҵ���������������б�ı߽�����Բ
	vector<RotatedRect> minRect(contours.size());
	vector<RotatedRect> minEllipse(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		// ������С�������
		minRect[i] = minAreaRect(Mat(contours[i]));
		// �������������Բ
		if (contours[i].size() > 5)
		{
			minEllipse[i] = fitEllipse(Mat(contours[i]));
		}
	}
	// ��������������б�ı߽��ͱ߽���Բ
	Mat resultMat = Mat::zeros(threMat.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		// ���������ɫ
		Scalar color = Scalar(rng.uniform(0, 255),
			rng.uniform(0, 255), rng.uniform(0, 255));
		// �߽���������
		drawContours(resultMat, contours, i, color, 1, 8,
			vector<Vec4i>(), 0, Point());
		// ���Ʊ߽���Բ
		ellipse(resultMat, minEllipse[i], color, 2, 8);
		// ���Ʊ߽����
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