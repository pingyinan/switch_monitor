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

vector<Rect> selectRegions;
void readData() {
	ifstream region_data("region.txt");
	int d[20];
	int i = 0;
	while (region_data >> d[i])
		i = i + 1;;
	region_data.close();
	for (int i = 0; i < 5; i++)
		selectRegions.push_back(Rect(d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]));
	cout << "read region successfully" << endl;
}
int main() {
	VideoCapture cap("test2.avi");
	if (!cap.isOpened()) {
		cout << "video read failed!" << endl;
		return -1;
	}
	Mat currentFrame;
	Mat background;
	Mat lastFrame;
	int Posflag = 0;
	readData();//读如选择的区域
	/*namedWindow("switch", 1);*/
	int flag = 0;
	while (1) {
		cap >> currentFrame;
		if (currentFrame.empty())break;
		if (flag == 0) {
			lastFrame = currentFrame.clone();
			background = currentFrame.clone();
			flag = 1;
			continue;
		}


		Mat diffFrame(currentFrame.rows, currentFrame.cols, CV_8UC1, Scalar(0));
		Mat graycurrent, graylast, graybackground;

		cvtColor(currentFrame, graycurrent, COLOR_BGR2GRAY);
		cvtColor(lastFrame, graylast, COLOR_BGR2GRAY);
		cvtColor(background, graybackground, COLOR_BGR2GRAY);
		//直方图均衡化
		//Mat P3 = graycurrent(selectRegions[3]);
		//imshow("P31", P3);
		//equalizeHist(P3, P3);
		//imshow("P3", P3);
		//滤波
		medianBlur(graylast, graylast, 3);
		medianBlur(graycurrent, graycurrent, 3);
		medianBlur(graybackground, graybackground, 3);

		absdiff(graycurrent, graylast, diffFrame);
		vector<vector<Point>> contours[5];
		vector<Vec4i> hierarchy[5];
		threshold(diffFrame, diffFrame, 20, 255, THRESH_BINARY);
		//如果使用腐蚀膨胀，可以将噪声经可能出去，那么baseline则为0
		Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
		Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
		erode(diffFrame, diffFrame, element1);
		dilate(diffFrame, diffFrame, element2);
		//
		Mat P[5];
		for (int i = 0; i < 5; i++) {
			P[i]= diffFrame(selectRegions[i]);
			findContours(P[i], contours[i], hierarchy[i], CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		}
		
		for (int i = 0; i < 5; i++) {
			rectangle(diffFrame, selectRegions[i], Scalar(255), 2, 3, 0);
			putText(diffFrame, "P" + to_string(i) + "contour number is:" + to_string(contours[i].size()), Point(selectRegions[i].x, selectRegions[i].y), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);

		}
		int threshold = 1;
		int filter = 3;
		static int N = 0;
		switch (Posflag) {
			case 0:
				putText(diffFrame, "geli", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
				if (contours[0].size() < threshold && contours[1].size() < threshold && contours[2].size() < threshold && contours[3].size() < threshold && contours[4].size() < threshold) 
				{
					N = 0;
				}
				else if (contours[0].size() > threshold && contours[1].size() > threshold && contours[2].size() > threshold)
				{
					N++;
					if (N == 5)
					{
						putText(diffFrame, "move to hezha", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
						N = 0;
						Posflag = 1;
					}
				}
				else if (contours[3].size() > threshold && contours[4].size() > threshold)
				{
					N++;
					if (N == 5)
					{
						putText(diffFrame, "move to jiedi", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
						N = 0;
						Posflag = 2;
					}
				}		
				break;
			case 1:
				putText(diffFrame, "move to hezha", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
				if (contours[0].size() > threshold && contours[1].size() > threshold && contours[2].size() > threshold)
				{				
					N = 0;
				}
				else if (contours[0].size() < threshold && contours[1].size() < threshold && contours[2].size() < threshold)
				{
					N++;
					if (N == 5)
					{
						putText(diffFrame, "hezha", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
						N = 0;
						Posflag = 3;
					}
				}
				break;
			case 2:
				putText(diffFrame, "move to jiedi", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
				if (contours[3].size() > threshold && contours[4].size() > threshold)
				{
					N = 0;
				}
				else if (contours[3].size() < threshold && contours[4].size() < threshold)
				{
					N++;
					if (N == 5) 
					{
						putText(diffFrame, "jiedi", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
						N = 0;
						Posflag = 4;
					}
				}
				break;
			case 3:
				putText(diffFrame, "hezha", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
				if (contours[0].size() < threshold && contours[1].size() < threshold && contours[2].size() < threshold)
				{
					N = 0;
				}
				else if (contours[0].size() > threshold && contours[1].size() > threshold && contours[2].size() > threshold)
				{
					N++;
					if (N == 5) 
					{
						putText(diffFrame, "move to geli", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
						N = 0;
						Posflag = 5;
					}
				}
				break;
			case 4:
				putText(diffFrame, "jiedi", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
				if (contours[3].size() < threshold && contours[4].size() < threshold)
				{
					N = 0;
				}
				else if (contours[3].size() > threshold && contours[4].size() > threshold)
				{
					N++;
					if (N == 5)
					{
						putText(diffFrame, "move to geli", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
						N = 0;
						Posflag = 5;
					}
				}
				break;
			case 5:
				putText(diffFrame, "move to geli", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
				if (contours[3].size() > threshold && contours[4].size() > threshold || contours[0].size() > threshold && contours[1].size() > threshold && contours[2].size() > threshold)
				{
					N = 0;
				}
				else if (contours[0].size() < threshold && contours[1].size() < threshold && contours[2].size() < threshold && contours[3].size() < threshold && contours[4].size() < threshold)
				{
					N++;
					if (N == 5) 
					{
						putText(diffFrame, "geli", Point(450, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255), 2, 8, 0);
						N = 0;
						Posflag = 0;
					}
				}
		}

		imshow("diff", diffFrame);	
		imshow("last", graycurrent);

		lastFrame = currentFrame.clone();
		waitKey(20);
	}
	return 0;
}