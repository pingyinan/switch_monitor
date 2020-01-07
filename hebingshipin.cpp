#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui_c.h>
#include <iostream>
using namespace cv;
using namespace std;
int main(){
  VideoCapture cap("2.mp4");
  VideoCapture cap2("3.mp4");
  // 设置摄像头的拍摄属性为 分辨率640x480，帧率30fps
  if (!cap.isOpened() || !cap.isOpened()) {
    cerr << " can not open a camera or file" << endl;
    return -1;
  }

  VideoWriter writer("test2.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.get(CAP_PROP_FPS), Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)), true);
  Mat videoPlay;
  int capflag = 0;
  namedWindow("VideoPlay", WINDOW_NORMAL);
  while (1){

    if(capflag == 0)
      cap >> videoPlay;
    else 
      cap2 >> videoPlay;
    if(videoPlay.empty() && capflag == 0){
      capflag = 1;
      continue;
    }
    else if(videoPlay.empty() && capflag == 1)
      break;

    writer << videoPlay;
    imshow("VideoPlay", videoPlay);
    waitKey(1000 / 30);
  }
  // 释放相关对象
  writer.release();
  cap.release();
  destroyWindow("VideoPlay");
  return 0;
}

