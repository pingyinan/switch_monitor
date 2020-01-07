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
#define R 2
#define G 1
#define B 0
#define H 0
#define S 1
#define V 2

bool downFlag = false;
bool upFlag = false;
bool getTargetFlag = false;
bool intialdone = false;

Mat current_frame;

Point startPoint;
Point endPoint;
Point movingPoint;
Rect roiRect;
vector<Rect> selectRegions;
int histSize = 10;
float range[] = { 0, 256 };
const float* histRange = { range };
bool uniform = true;
bool accumu = false;
vector<Mat> model_hist;
string name[3] = { "jiedi.jpg","geli.jpg","hezha.jpg" };
vector<Mat> frames;
#define RGBplane
//#define HSVplane
// #define Train
#define Running

void calculate_hist(Mat& image, Mat & hist) 
{
#ifdef RGBplane
	Mat b_hist, g_hist, r_hist;
	vector<Mat> bgr;
	split(image, bgr);
	calcHist(&bgr[R], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumu);
	calcHist(&bgr[G], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumu);
	calcHist(&bgr[B], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumu);
	// cout << r_hist.cols << r_hist.rows <<r_hist.type()<< hist.cols << hist.rows << endl;
	int area = image.cols*image.rows;
	r_hist = r_hist / area;
	g_hist = g_hist / area;
	b_hist = b_hist / area;
	r_hist.copyTo(hist.col(0));
	g_hist.copyTo(hist.col(1));
	b_hist.copyTo(hist.col(2));
	//cout << "rows:" << r_hist.rows << "cols:" << r_hist.cols << endl;
	// cout << hist << endl;
	// 设置直方图绘图参数
	int hist_w = 640; int hist_h = 512;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	// 分别归一化直方图到[ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	// 分别对每个通道进行绘图
	for (int i = 1; i <= histSize; i++)
	{
		line(histImage,
			Point(bin_w*(i - 0.5) - 5, hist_h),
			Point(bin_w*(i - 0.5) - 5, hist_h - cvRound(b_hist.at<float>(i - 1))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage,
			Point(bin_w*(i - 0.5), hist_h),
			Point(bin_w*(i - 0.5), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage,
			Point(bin_w*(i - 0.5) + 5, hist_h),
			Point(bin_w*(i - 0.5) + 5, hist_h - cvRound(r_hist.at<float>(i - 1))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
#endif
#ifdef HSVplane
	Mat h_hist, s_hist, v_hist;
	cvtColor(image, image, CV_BGR2HSV);
	vector<Mat> hsv;
	split(image, hsv);
	calcHist(&hsv[H], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumu);
	calcHist(&hsv[S], 1, 0, Mat(), s_hist, 1, &histSize, &histRange, uniform, accumu);
	calcHist(&hsv[V], 1, 0, Mat(), v_hist, 1, &histSize, &histRange, uniform, accumu);
	// cout << "rows:" << h_hist.rows << "cols:" << h_hist.cols << endl;
	// 设置直方图绘图参数
	int hist_w = 640; int hist_h = 512;
	int bin_w = cvRound((double)hist_w / histSize);

	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	// 分别归一化直方图到[ 0, histImage.rows ]
	normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(s_hist, s_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(v_hist, v_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	// 分别对每个通道进行绘图
	for (int i = 1; i <= histSize; i++)
	{
		line(histImage,
			Point(bin_w*(i - 0.5) - 5, hist_h),
			Point(bin_w*(i - 0.5) - 5, hist_h - cvRound(h_hist.at<float>(i - 1))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage,
			Point(bin_w*(i - 0.5), hist_h),
			Point(bin_w*(i - 0.5), hist_h - cvRound(s_hist.at<float>(i - 1))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage,
			Point(bin_w*(i - 0.5) + 5, hist_h),
			Point(bin_w*(i - 0.5) + 5, hist_h - cvRound(v_hist.at<float>(i - 1))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
#endif

	imshow("calcHist", histImage);
}
void on_mouse(int event, int x, int y, int , void* i)
{	
	Mat tempImage = current_frame.clone();
	int n = *(int *)i;
	//左键按下，取当前位置
	if (event == CV_EVENT_LBUTTONDOWN) {
		downFlag = true;
		getTargetFlag = false;
		//cout << "click" << endl;
		startPoint = Point(x, y);
	}
	//弹起，取当前位置作为终点
	if (event == CV_EVENT_LBUTTONUP) {
		upFlag = true;

		endPoint = Point(x, y);
		//终点最值限定
		if (endPoint.x > current_frame.cols)endPoint.x = current_frame.cols;
		if (endPoint.y > current_frame.cols)endPoint.y = current_frame.rows;
	}
	//鼠标现在选择的区域
	if (downFlag == true) {
		// cout << "moving" << endl;
		Point movingPoint = Point(x, y);
		if (movingPoint.x > current_frame.cols)movingPoint.x = current_frame.cols;
		if (movingPoint.y > current_frame.cols)movingPoint.y = current_frame.rows;
		Point tempPoint = startPoint;
		if (startPoint.x != movingPoint.x && movingPoint.y != endPoint.y) //如果鼠标移动
		{

			startPoint.x = min(startPoint.x, movingPoint.x);
			startPoint.y = min(startPoint.y, movingPoint.y);
			endPoint.x = max(tempPoint.x, movingPoint.x);
			endPoint.y = max(tempPoint.y, movingPoint.y);
			roiRect = Rect(startPoint.x, startPoint.y, endPoint.x - startPoint.x, endPoint.y - startPoint.y); //鼠标现在所选择的区域
			if (roiRect.area() > 0) {
				rectangle(tempImage, roiRect, Scalar(0, 0, 255), 2, 3, 0);
				putText(tempImage, "P" + to_string(n), Point(roiRect.x, roiRect.y), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);
			}
			if (selectRegions.size() > 0) {
				for (int j = 0; j < selectRegions.size(); j++) {
					rectangle(tempImage, selectRegions[j], Scalar(0, 0, 255), 2, 3, 0);
					putText(tempImage, "P" + to_string(j), Point(selectRegions[j].x, selectRegions[j].y), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);
				}
			}
			imshow("test", tempImage);
		}
	}
	//if (downflag == true && upflag == false) {
	//	point movingpoint = point(x,y);
	//	mat tempimage = current_frame.clone();
	//	rectangle(tempimage, startpoint, movingpoint, scalar(0, 0, 255), 2, 3, 0);
	//	imshow("test", tempimage);
	//}
	//按下选取完并弹起后
	if (downFlag == true && upFlag == true) {
		//起点和终点不相同时，才提取区域
		if (startPoint.x != endPoint.x && startPoint.y != endPoint.y) 
		{	
			// cout << roiRect << endl;
			Mat extractRegion = current_frame(roiRect).clone();
			// imshow("extractRegion", extractRegion);
		}
		downFlag = false;
		upFlag = false;
		getTargetFlag = true;

	}
}

void initial()
{	
	int i = 0;
	Mat tempImage;
	
	while (i < 5)
	{
		setMouseCallback("test", on_mouse, &i);
		cout << "using mouse to choose region P" << i << endl;
		Mat selectImage;
		Mat hist(histSize, 3, CV_32FC1, Scalar(0, 0, 0));
		current_frame = imread("first.jpg");

	find:
		tempImage = current_frame.clone();
		//显示
		if (roiRect.area() > 0) {
			rectangle(tempImage, roiRect, Scalar(0, 0, 255), 2, 3, 0);
			putText(tempImage, "P" + to_string(i), Point(roiRect.x, roiRect.y), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);
		}
		if (selectRegions.size() > 0) {
			for(int j = 0; j < selectRegions.size(); j++){
				rectangle(tempImage, selectRegions[j], Scalar(0, 0, 255), 2, 3, 0);
				putText(tempImage, "P" + to_string(j), Point(selectRegions[j].x, selectRegions[j].y), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);
			}
		}

		//cout << "columns:" << current_frame.cols << " rows:" << current_frame.rows << endl;
		imshow("test", tempImage);
		waitKey(10);
		if (getTargetFlag == true)
		{
			//cout << model_hist[0].at<float>(1, 2) << endl;
			cout << "press ENTER to store choose region" << endl;
			getTargetFlag = false; 
		}
		else
		{
			goto find;
		}
		if (waitKey(0) == 13)  //按ENTER确认选择区域，保存颜色直方图和区域信息
		{
			selectImage = current_frame(roiRect).clone();
			calculate_hist(selectImage, hist);
			model_hist.push_back(hist);
			selectRegions.push_back(roiRect);//保存选取的区域
			roiRect = Rect(0,0,0,0);
			// imshow("P" + to_string(i), selectImage);
			i = i + 1;
		}

	}
}

int main(){

	float maxdistance[5] = {0,0,0,0,0};
	float mindistance[5] = {3,3,3,3,3};

	VideoCapture cap("test2.avi");
	if (!cap.isOpened()) {
		cerr << " can not open a camera or file" << endl;
		return -1;
	}
	long totalFrameNumber=cap.get(CAP_PROP_FRAME_COUNT);
	cout<<"total frame number is :"<<totalFrameNumber<<endl;

	namedWindow("test", 1);

	#ifdef Train
	initial();//读入视频第一帧图像，选择5个检测区域
	#endif

	//已经训练完成的话直接读数据
	#ifdef Running
	ifstream region_data("region.txt");
    int d[20];
    int i = 0;
    while (region_data >> d[i])
        i=i+1;;
    region_data.close();
    for(int i = 0;i<5;i++)
    	selectRegions.push_back(Rect(d[i*4],d[i*4+1],d[i*4+2],d[i*4+3]));
    cout<<"read region successfully"<<endl;

    ifstream thread_data("thread.txt");
    float b[10];
    int j = 0;
    while(thread_data >> b[j])
    	j = j + 1;
    thread_data.close();
    for(int j = 0;j<5;j++){
    	mindistance[j] = b[j*2];
    	cout<<"minP"+to_string(j)+":"<<mindistance[j]<<endl;
    	maxdistance[j] = b[j*2+1];
    	cout<<"maxP"+to_string(j)+":"<<maxdistance[j]<<endl;
    }
    cout<<"read thread successfully"<<endl;

    ifstream model_hist_data("model_hist.txt");
    float h[5*3*histSize];
    int k = 0;
    while(model_hist_data >> h[k])
    	k = k + 1;
    model_hist_data.close();
    
    for (int k = 0;k < 5;k++){
    	Mat hist(histSize, 3, CV_32FC1, Scalar(0, 0, 0));
    	for (int i = 0;i < 3;i++)
    		for (int j = 0; j < histSize; j++)
    			hist.at<float>(j, i) = h[k * (3 * histSize) + i * histSize + j];
    	model_hist.push_back(hist);
    }
    cout<<"read model_hist successfully"<<endl;
    #endif

	//读入数据
	float distance[5];//5个检测区域偏离初始状态的距离，初始状态为隔离状态

	while(1){

		cap >> current_frame;

		if (current_frame.empty())

			break;

		Mat tempImage = current_frame.clone();

		Mat hist(histSize, 3, CV_32FC1, Scalar(0, 0, 0));

		for(int j = 0; j < selectRegions.size(); j++){
			Mat region = current_frame(selectRegions[j]);

			calculate_hist(region, hist);

			distance[j] = norm(model_hist[j], hist, CV_L2);

			#ifdef Train
			if(distance[j] < mindistance[j])
				mindistance[j] = distance[j];
			else if(distance[j] > maxdistance[j])
				maxdistance[j] = distance[j];
			#endif
			rectangle(tempImage, selectRegions[j], Scalar(0, 0, 255), 2, 3, 0);
			putText(tempImage, "P" + to_string(j) + "distance = " + to_string(distance[j]), Point(selectRegions[j].x, selectRegions[j].y), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);
		}

		#ifdef Train
		if(distance[0]>0.4&&distance[1]>0.4&&distance[2]>0.4)
			putText(tempImage, "hezha", Point(700,20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);
		else if(distance[3]>0.4&&distance[4]>0.4)
			putText(tempImage, "jiedi", Point(700,20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);
		else if(distance[0]<0.1&&distance[1]<0.1&&distance[2]<0.1&&distance[3]<0.1&&distance[4]<0.1)
			putText(tempImage, "geli", Point(700,20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);
		else
			putText(tempImage, "changing", Point(700,20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);
		#endif
		#ifdef Running
		float percent = 0.8;//
		cout<<distance[0]<<" "<<maxdistance[0]<<endl;
		if(distance[0]>maxdistance[0]*percent && distance[1]>maxdistance[1]*percent && distance[2]> maxdistance[2]*percent)
			putText(tempImage, "hezha2", Point(700,20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);
		else if(distance[3]>maxdistance[3]*percent && distance[4]>maxdistance[4]*percent)
			putText(tempImage, "jiedi", Point(700,20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);
		// else if(distance[0]<mindistance[0]+0.07 && distance[1]<mindistance[1]+0.07 && distance[2]<mindistance[2]+0.07 && distance[3]<mindistance[3]+0.07 && distance[4]<mindistance[4]+0.07)
		else if(distance[0]< 0.1+mindistance[0] && distance[1]<0.1+mindistance[1] && distance[2]<0.1+mindistance[2] && distance[3]<0.1+mindistance[3] && distance[4]<0.1+mindistance[4])
			putText(tempImage, "geli", Point(700,20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);
		else
			putText(tempImage, "changing", Point(700,20), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2, 8, 0);

		#endif


		imshow("test",tempImage);
		waitKey(30);
	}

	//保存选择区域以及各个区域内颜色直方图L2距离的极值
	#ifdef Train
	ofstream thread_file,region_file,model_hist_file;
	thread_file.open("./thread.txt");
	region_file.open("./region.txt");
	model_hist_file.open("./model_hist.txt");
	for (int i = 0;i < 5;i++){
		thread_file<<mindistance[i]<<" "<<maxdistance[i]<<endl;
		region_file<<selectRegions[i].x<<" "<<selectRegions[i].y<<" "<<selectRegions[i].width<<" "<<selectRegions[i].height<<endl;

	}
	for (int i = 0;i<5;i++){
		for(int j = 0;j<3;j++)//j = col;k = row,每个位置存一行，存完R再存G，再B
			for(int k = 0;k<histSize;k++){
				float n = model_hist[i].at<float>(k,j);
				model_hist_file<<n<<" ";
			}
		model_hist_file<<endl;	
	}
	model_hist_file.close();
	thread_file.close();
	region_file.close();
	cout<<"selected regions and threadshold have be saved "<<endl;
	#endif
	return 0;	
}

