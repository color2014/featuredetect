// Capture.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include <stdio.h>
/*#include <cv.h>
#include <cxcore.h>
#include <highgui.h>*/
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <time.h>
#include "string.h"

#define OTSU_THRESHOLD_MIN 15
#define OTSU_THRESHOLD_RET -15

/*****************动态生成二值化阈值******************/
int cvOSTU(IplImage *src)
{
	int deltaT = 0;
	uchar grayflag = 1;
	IplImage *gray = NULL;

	if(src->nChannels != 1)//注意这里将源图像变灰
	{
		gray = cvCreateImage(cvGetSize(src), 8, 1);
		cvCvtColor(src, gray, CV_BGR2GRAY);
		grayflag = 0;
	}
	else gray = src;

	uchar* ImgData = (uchar*)(gray->imageData);
	int thresholdValue = 1;  //阈值
	int ihist[256];   //图像直方图，256个点	
	int gmax = 255, gmin = 0;//最亮与最暗像素值
	int i, imgsize;   //循环变量与图像尺寸
	int n, n1, n2;    //n:非零像素个数，n1:前景像素个数, n2:背景像素个数
	double m1, m2, sum, csum, fmax, sb; //m1:前景灰度均值, m2:背景灰度均值。

	//对直方图置零
	memset(ihist, 0, sizeof(ihist));

	//生成直方图
	imgsize = (gray->widthStep)*(gray->height);  //图像数据总数
	for (i = 0; i < imgsize; i ++)
	{
		ihist[((int)(*ImgData)) & 255] ++;  //灰度统计
		ImgData ++;  //下一个像素

		if((int)(*ImgData) > gmax) 
			gmax = (int)(*ImgData);
		if((int)(*ImgData) < gmin)
			gmin = (int)(*ImgData);
	}

	//设置参数
	sum = csum = 0.0;
	n = 0;

	for (i = 0; i <= 255; i ++){
		sum += (double)i * (double)ihist[i];//x*f(x)质量矩
		n += ihist[i];   //f(x)质量，n：总灰度值
	}

	//加入光照调节参数
	deltaT = (int)( sum / imgsize );//背景像素个数，deltaT：光照调节参数

	deltaT = deltaT>>1;//pal值校正△T

	if (!n){
		//如果图像全黑，输出警告与结果
		printf ("NOT NORMAL thresholdValue=160\n");
		return (160);
	}

	// OTSU算法:
	fmax = -1.0;
	n1 = 0;

	for (i = 0; i < 255; i++)
	{
		n1 += ihist[i];  //前景像素个数
		if (!n1)	{continue;}

		n2 = n - n1;     //背景像素个数
		if (n2 == 0)	{break;}

		csum += (double)i * ihist[i];  //前景总灰度

		m1 = csum / n1;        //前景灰度均值
		m2 = (sum - csum)/n2;  //背景灰度均值

		sb = (double)n1 * (double)n2 * (m1 - m2) * (m1 - m2);  //类间方差

		/*这里是原理形式，可以被优化*/
		if ( sb > fmax )
		{
			fmax = sb;
			thresholdValue = i;//得到使类间方差最大的阈值T
		}

	}

	if(!grayflag)
		cvReleaseImage(&gray);
	if(thresholdValue < OTSU_THRESHOLD_MIN)
		return OTSU_THRESHOLD_MIN;
	return(thresholdValue - deltaT + OTSU_THRESHOLD_RET );//根据情况修改修正变量

}

/**************************基于运动模板的跟踪算法****************************/
void  cvUpdateMHI( IplImage* img, IplImage* dst)
{
	float MHI_DURATION = 0.5;
	const double MAX_TIME_DELTA = 0.25;
	const double MIN_TIME_DELTA = 0.05;
	char ID_T[50] = {0};

	double timestamp = clock()/1000.; // clock返回毫秒值，这里转换为秒为单位
	CvSize size=cvSize( img->width, img->height ); // get current frame size
	int i;
	IplImage* silh = NULL;
	IplImage* mhi = NULL;
	IplImage* orient = NULL;
	IplImage* segmask = NULL;
	IplImage* mask = NULL;
	CvSeq* seq;
	CvRect comp_rect;
	double count;
	double angle;
	CvPoint center;
	double magnitude;
	CvScalar color;
	CvFont font;
	CvMemStorage* storage = 0;

	// allocate images at the beginning or
	// reallocate them if the frame size is changed
	if( !mhi || mhi->width!=size.width || mhi->height!=size.height)
	{
		cvInitFont( &font, CV_FONT_HERSHEY_SCRIPT_COMPLEX, 1, 1, 0.0, 1, CV_AA);
		cvReleaseImage( &mhi);
		cvReleaseImage( &orient);
		cvReleaseImage( &segmask);
		cvReleaseImage( &mask);
		mhi = cvCreateImage( size, IPL_DEPTH_32F, 1);//IPL_DEPTH_32F - 单精度浮点数
		cvZero( mhi);// clear MHI at the beginning
		orient = cvCreateImage( size, IPL_DEPTH_32F, 1);
		segmask = cvCreateImage( size, IPL_DEPTH_32F, 1);
		mask = cvCreateImage( size, IPL_DEPTH_8U, 1);//IPL_DEPTH_8U - 无符号8位整型 
	}
	silh = img; // copy
	//****更新运动历史图像
	cvUpdateMotionHistory( silh, mhi, timestamp, MHI_DURATION);// update MHI

	//****cvCvtScale的第四个参数shift = (MHI_DURATION-timestamp)*255.0/MHI_DURATION，控制帧差的消失速率
	//使用线性变换转换数组
	cvConvertScale( mhi, mask, 255.0/ MHI_DURATION, (MHI_DURATION - timestamp)*255.0 / MHI_DURATION);

	cvZero( dst);
	cvCopy( mask, dst, 0);//B,G,R,O->dist:convert to GREEN image
	//计算运动的梯度方向以及正确的方向掩模mask，Filter size=3

	//****计算运动历史图像的梯度方向
	cvCalcMotionGradient( mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);

	if( !storage)
		storage = cvCreateMemStorage(0);
	else
		cvClearMemStorage(storage);

	//****运动分割:获得运动部件的连续序列
	// segmask is marked motion components map. It is not used further
	seq = cvSegmentMotion( mhi, segmask, storage, timestamp, MAX_TIME_DELTA);
	// iterate through the motion components,
	// One more iteration (i==-1) corresponds to the whole image (global motion)

	for( i = 0; i < seq->total; i ++)
	{
		if( i<0 )
		{
			// case of the whole image，对一幅图像做操作
			comp_rect = cvRect( 0, 0, size.width, size.height);
			color = CV_RGB(255, 255, 255);  //white color
			magnitude = 100;  //画线长度以及圆半径的大小控制
		}
		else{
			//i-th motion component
			comp_rect = ((CvConnectedComp*)cvGetSeqElem(seq, i))->rect;
			//去掉小的部分
			if(comp_rect.width + comp_rect.height < 20)
				continue;
			color = CV_RGB(128, 128, 128);          // red color
			magnitude = ((comp_rect.width + comp_rect.height) >> 1)/ MHI_DURATION;
			//if(seq->total>0) MessageBox(NULL,"Motion Detected",NULL,O);
		}

		// select component ROI
		cvSetImageROI( silh, comp_rect);
		cvSetImageROI( mhi, comp_rect);
		cvSetImageROI( orient, comp_rect);
		cvSetImageROI( mask, comp_rect);

		//****在选择的区域内，计算运动方向
		//计算某些选择区域的全局运动方向 
		/*函数 cvCalcGlobalOrientation 
		在选择的区域内计算整个运动方向，并且返回 0°到 360°之间的角度值。
		首先函数创建运动直方图，寻找基本方向做为直方图最大值的坐标。
		然后函数计算与基本方向的相对偏移量，做为所有方向向量的加权和：运行越近，权重越大。
		得到的角度是基本方向和偏移量的循环和。 */
		angle = cvCalcGlobalOrientation( orient, mask, mhi, timestamp, MHI_DURATION);

		angle = 360.0 - angle;  // adjust for images with top-left origin

		//****在轮廓内计算点数
		count = cvNorm( silh, 0, CV_L1, 0);

		// The function cvResetlmageROI releases image ROI
		cvResetImageROI( mhi);
		cvResetImageROI( orient);
		cvResetImageROI( mask);
		cvResetImageROI( silh);

		// check for the case of little motion
		if( count< comp_rect.width * comp_rect.height * 0.05)/////5%的像素
			continue;

		// draw a clock with arrow indicating the direction
		//画一个指示运动方向的时钟
		center = cvPoint( (comp_rect.x + comp_rect.width/2), (comp_rect.y + comp_rect.height/2));
		cvCircle( dst, center, cvRound(magnitude* 1.0), color, 1, CV_AA, 0);
		cvLine( dst,
			center, 
			cvPoint( cvRound( center.x + magnitude*cos(angle*CV_PI/180.0)),
			cvRound( center.y - magnitude * sin(angle*CV_PI/180.0))),
			color, 
			1, 
			CV_AA,
			0);
		sprintf_s(ID_T, "%d", i);//
		cvPutText(dst, ID_T, center, &font, color);
	}
}


/*******************框图跟踪目标,连通域检测******************/
void cvTrack(IplImage* src1, IplImage*src2, IplImage* dst)
{
	CvMemStorage* storage = 0;
	CvFont font;
	char charname[50];

	storage = cvCreateMemStorage(0);//开辟默认大小的空间
	CvSeq* contour=0;

	cvFindContours( src1, storage, &contour, sizeof(CvContour), 
		CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);//得到目标外边缘

	int num = 0;
	for (; contour != 0; contour = contour->h_next)
	{
		num ++;
		CvPoint ptCenter;
		CvRect rect;
		rect = cvBoundingRect(contour, 0);//得到目标外接矩形

		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5f, 0.5f, 0, 1, 8);//初始化字体

		_itoa_s(num, charname, 10);//将num的int类型转化为string类型

		if ((rect.height + rect.width)>=16)
		{
			//绘制目标外接矩形
			cvRectangle(dst, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height),
				CV_RGB(255, 255, 0), 1, 8);//前景图

			cvRectangle(src2, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height),
				CV_RGB(160, 160, 160), 1, 8);//二值图

			ptCenter = cvPoint( (rect.x + rect.width/2), (rect.y + rect.height/2));//质心坐标

			//绘制编号
			cvPutText(dst, charname, ptCenter, &font, cvScalar(255, 0, 0)); 
			cvPutText(src2, charname, ptCenter, &font, cvScalar(255, 0, 0)); 

			//绘制运动轨迹
			cvLine(dst, ptCenter, ptCenter, cvScalar(0, 0, 255), 2, 8, 0);

		}

	}
	cvClearMemStorage(storage);

}
int main( int argc, char** argv )
{
	//声明IplImage指针
	IplImage* pFrame = NULL; 
	IplImage* pFrImg = NULL;
	IplImage* pFrImg_2 = NULL;
	IplImage* pBkImg = NULL;
	IplImage* pImg2 = NULL;

	CvMat* pFrameMat = NULL;
	CvMat* pFrMat = NULL;
	CvMat* pBkMat = NULL;

	CvCapture* pCapture = NULL;

	/*CvMemStorage* pt_storage = 0;*/

	int nFrmNum = 0; 
	int key_sum = 1;
	int signal = 1;

	//创建窗口
	cvNamedWindow("video", 1);
	cvNamedWindow("background",1);
	cvNamedWindow("foreground",1);

	//使窗口有序排列
	cvMoveWindow("video", 30, 0);
	cvMoveWindow("background", 395, 0);
	cvMoveWindow("foreground", 760, 0);

 	//打开视频文件
	if(argc == 1)
		if( !(pCapture = cvCaptureFromFile("person.avi ")))
			{
				fprintf(stderr, "Can not open video file %s\n", argv[1]);
			    return -2;
		    }

	double m_FrameNum = cvGetCaptureProperty(pCapture, CV_CAP_PROP_FRAME_COUNT);

	while(1)
		{
			//逐帧读取视频
			pFrame = cvQueryFrame( pCapture );
			nFrmNum ++;

			/*************************保存BMP格式文件*************************/

			//char FileNum[256];
			//char Filename[256];
			//strcpy_s(Filename, "person_");
			//_itoa_s(nFrmNum, FileNum, 10);//把整型转化为字符串
			//strcat_s(Filename, FileNum);//从尾端添加字符串
			//strcat_s(Filename, ".BMP");
			//
			//pImg2 = cvCreateImage(cvGetSize(pFrame), pFrame->depth, pFrame->nChannels);
			//cvCopy(pFrame, pImg2);//复制当前帧图片
			//cvSaveImage((LPCSTR)Filename, pImg2);//把图像写入文件

			//如果是第一帧，需要申请内存，并初始化
			if(nFrmNum == 1)
			{
				pBkImg = cvCreateImage(cvSize(pFrame->width, pFrame->height),  IPL_DEPTH_8U, 1);
				pFrImg = cvCreateImage(cvSize(pFrame->width, pFrame->height),  IPL_DEPTH_8U, 1);

				pBkMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
				pFrMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
				pFrameMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);

				//转化成单通道图像再处理
				cvCvtColor(pFrame, pBkImg, CV_BGR2GRAY);
				cvCvtColor(pFrame, pFrImg, CV_BGR2GRAY);

				cvConvert(pFrImg, pFrameMat);
				cvConvert(pFrImg, pFrMat);
				cvConvert(pFrImg, pBkMat);	
			}
			else
			{
				if(nFrmNum == m_FrameNum)
				{
					pCapture = cvCaptureFromFile("person.avi"); //连续播放视频 2011.06.07
				}
				cvCvtColor(pFrame, pFrImg, CV_BGR2GRAY);
				cvConvert(pFrImg, pFrameMat);//将前面的帧转换为矩阵格式

				/****先高斯滤波，以平滑图像，去噪声****/
				cvSmooth(pFrameMat, pFrameMat, CV_GAUSSIAN, 3, 0, 0);//考虑其他滤波方法

				//当前帧跟背景图相减，差帧法；如果是第一帧，即为背景图
				cvAbsDiff(pFrameMat, pBkMat, pFrMat);

				//二值化前景图，采用OSTU动态赋予阈值
				cvThreshold(pFrMat, pFrImg, cvOSTU(pFrame), 255.0, CV_THRESH_BINARY);

				//输出当前选用的阈值
				printf("Threshold Value Is %d\n", cvOSTU(pFrame));

				//保存当前背景图像
				pFrImg_2 = cvCreateImage(cvGetSize(pFrImg), pFrImg->depth, pFrImg->nChannels);
				cvCopy(pFrImg, pFrImg_2);

				/****进行形态学滤波（开运算），去掉噪音****/ 
				cvErode(pFrImg, pFrImg, 0, 5);
				cvDilate(pFrImg, pFrImg, 0, 5);//改变膨胀次数

				//更新背景图，采用“累积差分动态生成”
				cvRunningAvg(pFrameMat, pBkMat, 0.003, 0);//函数 cvRunningAvg 计算输入图像 image 的加权和，以及累积器 acc 
				                                          //使得 acc 成为帧序列的一个 running average(滑动平均)；

				//将背景转化为图像格式，用以在background中显示
				cvConvert(pBkMat, pBkImg);

				/*******************框图跟踪目标,连通域检测******************/
				cvTrack(pFrImg, pFrImg_2, pFrame);
				//cvUpdateMHI(pFrImg, pFrImg);

				//显示图像
				cvShowImage("video", pFrame);
				cvShowImage("background", pBkImg);
				cvShowImage("foreground", pFrImg_2);

				//如果有按键事件，则跳出循环;此等待也为cvShowImage函数提供时间完成显示
				//等待时间可以根据CPU速度调整
								
			}

			/*********键盘响应**********/

			int key = cvWaitKey(2);

			//暂停状态
			if(signal == 0)
			{	
				if( (char)key == 27)//退出播放
					break;
				else
				{
					if( (char)key == 32)//继续播放
					{
						signal = 1;
						key_sum --;
					}
					else
					{
						if ((char)key > 0)//任意键单帧播放
						{
							pFrame = cvQueryFrame( pCapture );
							cvShowImage("video", pFrame);
						}
					}

				}

			}
			else//运行状态
			{
				if( (char)key == 27)//退出播放
					break;

				switch((char)key)
			   {
				   //空格，暂停播放
				   case 32:
					   key_sum++;
					   if(key_sum%2 == 0)
						   signal = 0;
					   break;
					
				   default:
					   ;
				}
			}

		}

		//销毁窗口
		cvDestroyWindow("video");
		cvDestroyWindow("background");
		cvDestroyWindow("foreground");

		cvReleaseCapture(&pCapture);

		cvReleaseMat(&pFrameMat);
		cvReleaseMat(&pFrMat);
		cvReleaseMat(&pBkMat);

		//释放图像和矩阵
		cvReleaseImage(&pFrame);
		cvReleaseImage(&pFrImg);
		cvReleaseImage(&pBkImg);
		cvReleaseImage(&pFrImg_2);
		cvReleaseImage(&pImg2);

		return 0;
}


