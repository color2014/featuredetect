// Capture.cpp : �������̨Ӧ�ó������ڵ㡣
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

/*****************��̬���ɶ�ֵ����ֵ******************/
int cvOSTU(IplImage *src)
{
	int deltaT = 0;
	uchar grayflag = 1;
	IplImage *gray = NULL;

	if(src->nChannels != 1)//ע�����ｫԴͼ����
	{
		gray = cvCreateImage(cvGetSize(src), 8, 1);
		cvCvtColor(src, gray, CV_BGR2GRAY);
		grayflag = 0;
	}
	else gray = src;

	uchar* ImgData = (uchar*)(gray->imageData);
	int thresholdValue = 1;  //��ֵ
	int ihist[256];   //ͼ��ֱ��ͼ��256����	
	int gmax = 255, gmin = 0;//�����������ֵ
	int i, imgsize;   //ѭ��������ͼ��ߴ�
	int n, n1, n2;    //n:�������ظ�����n1:ǰ�����ظ���, n2:�������ظ���
	double m1, m2, sum, csum, fmax, sb; //m1:ǰ���ҶȾ�ֵ, m2:�����ҶȾ�ֵ��

	//��ֱ��ͼ����
	memset(ihist, 0, sizeof(ihist));

	//����ֱ��ͼ
	imgsize = (gray->widthStep)*(gray->height);  //ͼ����������
	for (i = 0; i < imgsize; i ++)
	{
		ihist[((int)(*ImgData)) & 255] ++;  //�Ҷ�ͳ��
		ImgData ++;  //��һ������

		if((int)(*ImgData) > gmax) 
			gmax = (int)(*ImgData);
		if((int)(*ImgData) < gmin)
			gmin = (int)(*ImgData);
	}

	//���ò���
	sum = csum = 0.0;
	n = 0;

	for (i = 0; i <= 255; i ++){
		sum += (double)i * (double)ihist[i];//x*f(x)������
		n += ihist[i];   //f(x)������n���ܻҶ�ֵ
	}

	//������յ��ڲ���
	deltaT = (int)( sum / imgsize );//�������ظ�����deltaT�����յ��ڲ���

	deltaT = deltaT>>1;//palֵУ����T

	if (!n){
		//���ͼ��ȫ�ڣ������������
		printf ("NOT NORMAL thresholdValue=160\n");
		return (160);
	}

	// OTSU�㷨:
	fmax = -1.0;
	n1 = 0;

	for (i = 0; i < 255; i++)
	{
		n1 += ihist[i];  //ǰ�����ظ���
		if (!n1)	{continue;}

		n2 = n - n1;     //�������ظ���
		if (n2 == 0)	{break;}

		csum += (double)i * ihist[i];  //ǰ���ܻҶ�

		m1 = csum / n1;        //ǰ���ҶȾ�ֵ
		m2 = (sum - csum)/n2;  //�����ҶȾ�ֵ

		sb = (double)n1 * (double)n2 * (m1 - m2) * (m1 - m2);  //��䷽��

		/*������ԭ����ʽ�����Ա��Ż�*/
		if ( sb > fmax )
		{
			fmax = sb;
			thresholdValue = i;//�õ�ʹ��䷽��������ֵT
		}

	}

	if(!grayflag)
		cvReleaseImage(&gray);
	if(thresholdValue < OTSU_THRESHOLD_MIN)
		return OTSU_THRESHOLD_MIN;
	return(thresholdValue - deltaT + OTSU_THRESHOLD_RET );//��������޸���������

}

/**************************�����˶�ģ��ĸ����㷨****************************/
void  cvUpdateMHI( IplImage* img, IplImage* dst)
{
	float MHI_DURATION = 0.5;
	const double MAX_TIME_DELTA = 0.25;
	const double MIN_TIME_DELTA = 0.05;
	char ID_T[50] = {0};

	double timestamp = clock()/1000.; // clock���غ���ֵ������ת��Ϊ��Ϊ��λ
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
		mhi = cvCreateImage( size, IPL_DEPTH_32F, 1);//IPL_DEPTH_32F - �����ȸ�����
		cvZero( mhi);// clear MHI at the beginning
		orient = cvCreateImage( size, IPL_DEPTH_32F, 1);
		segmask = cvCreateImage( size, IPL_DEPTH_32F, 1);
		mask = cvCreateImage( size, IPL_DEPTH_8U, 1);//IPL_DEPTH_8U - �޷���8λ���� 
	}
	silh = img; // copy
	//****�����˶���ʷͼ��
	cvUpdateMotionHistory( silh, mhi, timestamp, MHI_DURATION);// update MHI

	//****cvCvtScale�ĵ��ĸ�����shift = (MHI_DURATION-timestamp)*255.0/MHI_DURATION������֡�����ʧ����
	//ʹ�����Ա任ת������
	cvConvertScale( mhi, mask, 255.0/ MHI_DURATION, (MHI_DURATION - timestamp)*255.0 / MHI_DURATION);

	cvZero( dst);
	cvCopy( mask, dst, 0);//B,G,R,O->dist:convert to GREEN image
	//�����˶����ݶȷ����Լ���ȷ�ķ�����ģmask��Filter size=3

	//****�����˶���ʷͼ����ݶȷ���
	cvCalcMotionGradient( mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);

	if( !storage)
		storage = cvCreateMemStorage(0);
	else
		cvClearMemStorage(storage);

	//****�˶��ָ�:����˶���������������
	// segmask is marked motion components map. It is not used further
	seq = cvSegmentMotion( mhi, segmask, storage, timestamp, MAX_TIME_DELTA);
	// iterate through the motion components,
	// One more iteration (i==-1) corresponds to the whole image (global motion)

	for( i = 0; i < seq->total; i ++)
	{
		if( i<0 )
		{
			// case of the whole image����һ��ͼ��������
			comp_rect = cvRect( 0, 0, size.width, size.height);
			color = CV_RGB(255, 255, 255);  //white color
			magnitude = 100;  //���߳����Լ�Բ�뾶�Ĵ�С����
		}
		else{
			//i-th motion component
			comp_rect = ((CvConnectedComp*)cvGetSeqElem(seq, i))->rect;
			//ȥ��С�Ĳ���
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

		//****��ѡ��������ڣ������˶�����
		//����ĳЩѡ�������ȫ���˶����� 
		/*���� cvCalcGlobalOrientation 
		��ѡ��������ڼ��������˶����򣬲��ҷ��� 0�㵽 360��֮��ĽǶ�ֵ��
		���Ⱥ��������˶�ֱ��ͼ��Ѱ�һ���������Ϊֱ��ͼ���ֵ�����ꡣ
		Ȼ���������������������ƫ��������Ϊ���з��������ļ�Ȩ�ͣ�����Խ����Ȩ��Խ��
		�õ��ĽǶ��ǻ��������ƫ������ѭ���͡� */
		angle = cvCalcGlobalOrientation( orient, mask, mhi, timestamp, MHI_DURATION);

		angle = 360.0 - angle;  // adjust for images with top-left origin

		//****�������ڼ������
		count = cvNorm( silh, 0, CV_L1, 0);

		// The function cvResetlmageROI releases image ROI
		cvResetImageROI( mhi);
		cvResetImageROI( orient);
		cvResetImageROI( mask);
		cvResetImageROI( silh);

		// check for the case of little motion
		if( count< comp_rect.width * comp_rect.height * 0.05)/////5%������
			continue;

		// draw a clock with arrow indicating the direction
		//��һ��ָʾ�˶������ʱ��
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


/*******************��ͼ����Ŀ��,��ͨ����******************/
void cvTrack(IplImage* src1, IplImage*src2, IplImage* dst)
{
	CvMemStorage* storage = 0;
	CvFont font;
	char charname[50];

	storage = cvCreateMemStorage(0);//����Ĭ�ϴ�С�Ŀռ�
	CvSeq* contour=0;

	cvFindContours( src1, storage, &contour, sizeof(CvContour), 
		CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);//�õ�Ŀ�����Ե

	int num = 0;
	for (; contour != 0; contour = contour->h_next)
	{
		num ++;
		CvPoint ptCenter;
		CvRect rect;
		rect = cvBoundingRect(contour, 0);//�õ�Ŀ����Ӿ���

		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5f, 0.5f, 0, 1, 8);//��ʼ������

		_itoa_s(num, charname, 10);//��num��int����ת��Ϊstring����

		if ((rect.height + rect.width)>=16)
		{
			//����Ŀ����Ӿ���
			cvRectangle(dst, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height),
				CV_RGB(255, 255, 0), 1, 8);//ǰ��ͼ

			cvRectangle(src2, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height),
				CV_RGB(160, 160, 160), 1, 8);//��ֵͼ

			ptCenter = cvPoint( (rect.x + rect.width/2), (rect.y + rect.height/2));//��������

			//���Ʊ��
			cvPutText(dst, charname, ptCenter, &font, cvScalar(255, 0, 0)); 
			cvPutText(src2, charname, ptCenter, &font, cvScalar(255, 0, 0)); 

			//�����˶��켣
			cvLine(dst, ptCenter, ptCenter, cvScalar(0, 0, 255), 2, 8, 0);

		}

	}
	cvClearMemStorage(storage);

}
int main( int argc, char** argv )
{
	//����IplImageָ��
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

	//��������
	cvNamedWindow("video", 1);
	cvNamedWindow("background",1);
	cvNamedWindow("foreground",1);

	//ʹ������������
	cvMoveWindow("video", 30, 0);
	cvMoveWindow("background", 395, 0);
	cvMoveWindow("foreground", 760, 0);

 	//����Ƶ�ļ�
	if(argc == 1)
		if( !(pCapture = cvCaptureFromFile("person.avi ")))
			{
				fprintf(stderr, "Can not open video file %s\n", argv[1]);
			    return -2;
		    }

	double m_FrameNum = cvGetCaptureProperty(pCapture, CV_CAP_PROP_FRAME_COUNT);

	while(1)
		{
			//��֡��ȡ��Ƶ
			pFrame = cvQueryFrame( pCapture );
			nFrmNum ++;

			/*************************����BMP��ʽ�ļ�*************************/

			//char FileNum[256];
			//char Filename[256];
			//strcpy_s(Filename, "person_");
			//_itoa_s(nFrmNum, FileNum, 10);//������ת��Ϊ�ַ���
			//strcat_s(Filename, FileNum);//��β������ַ���
			//strcat_s(Filename, ".BMP");
			//
			//pImg2 = cvCreateImage(cvGetSize(pFrame), pFrame->depth, pFrame->nChannels);
			//cvCopy(pFrame, pImg2);//���Ƶ�ǰ֡ͼƬ
			//cvSaveImage((LPCSTR)Filename, pImg2);//��ͼ��д���ļ�

			//����ǵ�һ֡����Ҫ�����ڴ棬����ʼ��
			if(nFrmNum == 1)
			{
				pBkImg = cvCreateImage(cvSize(pFrame->width, pFrame->height),  IPL_DEPTH_8U, 1);
				pFrImg = cvCreateImage(cvSize(pFrame->width, pFrame->height),  IPL_DEPTH_8U, 1);

				pBkMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
				pFrMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
				pFrameMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);

				//ת���ɵ�ͨ��ͼ���ٴ���
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
					pCapture = cvCaptureFromFile("person.avi"); //����������Ƶ 2011.06.07
				}
				cvCvtColor(pFrame, pFrImg, CV_BGR2GRAY);
				cvConvert(pFrImg, pFrameMat);//��ǰ���֡ת��Ϊ�����ʽ

				/****�ȸ�˹�˲�����ƽ��ͼ��ȥ����****/
				cvSmooth(pFrameMat, pFrameMat, CV_GAUSSIAN, 3, 0, 0);//���������˲�����

				//��ǰ֡������ͼ�������֡��������ǵ�һ֡����Ϊ����ͼ
				cvAbsDiff(pFrameMat, pBkMat, pFrMat);

				//��ֵ��ǰ��ͼ������OSTU��̬������ֵ
				cvThreshold(pFrMat, pFrImg, cvOSTU(pFrame), 255.0, CV_THRESH_BINARY);

				//�����ǰѡ�õ���ֵ
				printf("Threshold Value Is %d\n", cvOSTU(pFrame));

				//���浱ǰ����ͼ��
				pFrImg_2 = cvCreateImage(cvGetSize(pFrImg), pFrImg->depth, pFrImg->nChannels);
				cvCopy(pFrImg, pFrImg_2);

				/****������̬ѧ�˲��������㣩��ȥ������****/ 
				cvErode(pFrImg, pFrImg, 0, 5);
				cvDilate(pFrImg, pFrImg, 0, 5);//�ı����ʹ���

				//���±���ͼ�����á��ۻ���ֶ�̬���ɡ�
				cvRunningAvg(pFrameMat, pBkMat, 0.003, 0);//���� cvRunningAvg ��������ͼ�� image �ļ�Ȩ�ͣ��Լ��ۻ��� acc 
				                                          //ʹ�� acc ��Ϊ֡���е�һ�� running average(����ƽ��)��

				//������ת��Ϊͼ���ʽ��������background����ʾ
				cvConvert(pBkMat, pBkImg);

				/*******************��ͼ����Ŀ��,��ͨ����******************/
				cvTrack(pFrImg, pFrImg_2, pFrame);
				//cvUpdateMHI(pFrImg, pFrImg);

				//��ʾͼ��
				cvShowImage("video", pFrame);
				cvShowImage("background", pBkImg);
				cvShowImage("foreground", pFrImg_2);

				//����а����¼���������ѭ��;�˵ȴ�ҲΪcvShowImage�����ṩʱ�������ʾ
				//�ȴ�ʱ����Ը���CPU�ٶȵ���
								
			}

			/*********������Ӧ**********/

			int key = cvWaitKey(2);

			//��ͣ״̬
			if(signal == 0)
			{	
				if( (char)key == 27)//�˳�����
					break;
				else
				{
					if( (char)key == 32)//��������
					{
						signal = 1;
						key_sum --;
					}
					else
					{
						if ((char)key > 0)//�������֡����
						{
							pFrame = cvQueryFrame( pCapture );
							cvShowImage("video", pFrame);
						}
					}

				}

			}
			else//����״̬
			{
				if( (char)key == 27)//�˳�����
					break;

				switch((char)key)
			   {
				   //�ո���ͣ����
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

		//���ٴ���
		cvDestroyWindow("video");
		cvDestroyWindow("background");
		cvDestroyWindow("foreground");

		cvReleaseCapture(&pCapture);

		cvReleaseMat(&pFrameMat);
		cvReleaseMat(&pFrMat);
		cvReleaseMat(&pBkMat);

		//�ͷ�ͼ��;���
		cvReleaseImage(&pFrame);
		cvReleaseImage(&pFrImg);
		cvReleaseImage(&pBkImg);
		cvReleaseImage(&pFrImg_2);
		cvReleaseImage(&pImg2);

		return 0;
}


