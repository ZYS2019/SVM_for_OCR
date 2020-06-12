//SVM�����ѵ������
#include <opencv.hpp>
#include<highgui.hpp>
#include<ml.hpp>
#include<core.hpp>
#include <iostream>
#include <fstream>
#include<io.h>

#include<ctime>

using namespace cv;
using namespace std;

#define random(a,b) (rand()%(b-a)+a)   //���ɣ�a,b��֮��������
Size imageSize = Size(64, 64);    //�ı���Ĵ�С
Size meterSize = Size(800, 800);   //����ͼƬ�Ĵ�С������ָ���ͼ��

//-----------------------------------------------------------
//����ͼ���HOG����
//-----------------------------------------------------------
void coumputeHog(const Mat& src, vector<float> &descriptors)
{
	HOGDescriptor myHog = HOGDescriptor(imageSize, Size(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	myHog.compute(src.clone(), descriptors, Size(1, 1), Size(0, 0));

}

//-----------------------------------------------------------
//saveFilesFullpathFromFolderInFormat������
//��ָ���ļ���·�����ҵ�����ָ����׺�����ļ���
//��·��������fileFullpath������
//���룺path �ļ���·��
//      format ��׺��
//�����fileFullpath  path������ָ����׺���ļ�������·��
//------------------------------------------------------------
void saveFilesFullpathFromFolderInFormat(string path, vector<string>& fileFullpath, string format) {
	_finddata_t fileInfo;
	string s;
	const char* filePath = s.assign(path).append("\\*").append(format).c_str();  //��stringת��Ϊconst char*����
	intptr_t fileHandle = _findfirst(filePath, &fileInfo); //��ȡ��һ���ļ���Ϣ �ļ��������Ϊlong����ֲ�����[Win10ƽ̨��long�����ļ������crash]
	string f;
	if (fileHandle == -1) {
		cout << "error\n";
	}
	else {
		f = path + string("\\") + string(fileInfo.name);
		fileFullpath.push_back(f);
	}
	while (_findnext(fileHandle, &fileInfo) == 0) {
		f = path + string("\\") + string(fileInfo.name);
		fileFullpath.push_back(f);
	}
	_findclose(fileHandle);
}

int main(int argc, char** argv) {
	int TRAIN = 0;
	string imageName;
	signed imageLabel;
	vector<Mat> vecImages;
	vector<int> vecLabels;
	CvSVM *mySVM = new CvSVM();
	CvSVMParams params = CvSVMParams();
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 1e-10);
	vector<float> vecDescriptors;

	if (TRAIN) //�Ƿ���Ҫѵ��
	{
		string s1 = "D:\\ZYS\\ָ��ʽ��\\dataset\\JPEG";
		vector<string> imgFiles;
		vector<string> vTxtFiles;
		//�ҳ��ļ��������д�jpg��׺���ļ�
		string sImg = ".jpg";
		saveFilesFullpathFromFolderInFormat(s1, imgFiles, sImg);

		int imgNum = imgFiles.size();
		for (int i = 0; i < imgNum; i++)
		{
			Mat src = imread(imgFiles[i], 0);
			float zoomx = (float)800 / src.cols;
			float zoomy = (float)800 / src.rows;
			resize(src, src, meterSize);

			vector<Rect> vTxtRect;
			//�ҳ���ͼ���Ӧ��txt�ļ�
			string sTxtFile = imgFiles[i];
			int offindex = sTxtFile.find(".jpg");
			sTxtFile.replace(offindex, 4, ".txt");

			//��ȡimg��Ӧ��txt�ļ�
			fstream in;
			in.open(sTxtFile, ios::in);
			if (!in.is_open())
			{
				cout << "can not find" << sTxtFile << endl;
				system("pause");
			}
			//��ȡ������������vecImages
			string buff;
			int row = 0;  //����row
			while (getline(in, buff)) {
				vector<int> temp;
				//string -> char
				char *s_input = (char *)buff.c_str();
				const char * split = ",";
				//��","Ϊ�ָ�������ַ���
				char *p = strtok(s_input, split);
				while (p != NULL) {
					//char *  -> int
					int a = atof(p);
					temp.push_back(a);
					p = strtok(NULL, split);
				}
				vector<cv::Point> contours;
				for (int n = 0; n < 7; n=n + 2)
				{
					Point temp2((int)temp[n] * zoomx, (int)temp[n + 1] * zoomy);
					contours.push_back(temp2);
				}
				Rect rect = boundingRect(contours);
				/*int x = temp[0];
				int y = temp[1];
				int width = temp[2] - temp[0];
				int height = temp[5] - temp[3];
				Rect rect(x, y, width, height);*/
				vTxtRect.push_back(rect);
				Mat roi;
				src(rect).copyTo(roi);
				resize(roi, roi, imageSize);
				vecImages.push_back(roi);
				vecLabels.push_back(1);
			}
			int w = src.cols;
			int h = src.rows;
			//ÿ��ͼҪ����һЩ���������������12��
			srand((int)time(0));
			for (int j = 0; j < 12;)
			{
				int x = random(0, (int)(w - 175*zoomx));
				int y = random(0, (int)(h - 130*zoomy));
				Rect negRect(x, y, (int)170*zoomx, (int)125*zoomy);
				int chongdie=0;
				for (int n = 0; n < vTxtRect.size(); n++)
				{
					if ((negRect&vTxtRect[n]).area())
					{
						chongdie=1;
					}
				}
				if (chongdie == 1)
				{
					continue;
				}
				Mat negMat;
				src(negRect).copyTo(negMat);
				resize(negMat, negMat, imageSize);
				vecImages.push_back(negMat);
				vecLabels.push_back(0);
				j++;
			}
		}
		//��������Hog������ѵ��SVM
		Mat dataDescriptors;
		Mat dataResponse = (Mat)vecLabels;
		for (size_t i = 0; i < vecImages.size(); i++)
		{
			Mat src = vecImages[i];
			Mat tempRow;
			coumputeHog(src, vecDescriptors);
			if (i == 0)
			{
				dataDescriptors = Mat::zeros(vecImages.size(), vecDescriptors.size(), CV_32FC1);
			}
			tempRow = ((Mat)vecDescriptors).t();
			tempRow.row(0).copyTo(dataDescriptors.row(i));
		}

		mySVM->train(dataDescriptors, dataResponse, Mat(), Mat(), params);
		string svmName = to_string(0611) + "_mysvm.xml";
		mySVM->save(svmName.c_str());
	}
	else {
		mySVM->load("393_mysvm.xml");
	}	

	//-----------------Ԥ�ⵥ��ͼƬ-------------------------
	/*string testPath="C:\\Users\\SZJ\\Desktop\\������ͼ\\negTxt.jpg";
	Mat testMat = imread(testPath, 0);
	resize(testMat, testMat, imageSize);
	vector<float> imageDescriptor;
	coumputeHog(testMat, imageDescriptor);
	Mat testDescriptor = Mat::zeros(1, imageDescriptor.size(), CV_32FC1);
	for (size_t i = 0; i < imageDescriptor.size(); i++)
	{
		testDescriptor.at<float>(0, i) = imageDescriptor[i];
	}
	float  label = mySVM->predict(testDescriptor, false);
	cout << label << endl;
	imshow("test image", testMat);
	waitKey(0);*/
	//--------------��������----------------------
	Mat testMeterMat = imread("D:\\ZYS\\ָ��ʽ��\\dataset\\JPEG\\1.jpg");
	Mat testMeterGray,zoMeter; //zoMeter������ʾ��

	cvtColor(testMeterMat, testMeterGray, COLOR_BGR2GRAY);
	float zoomx = (float)800 / testMeterGray.cols;
	float zoomy = (float)800 / testMeterGray.rows;
	resize(testMeterMat, zoMeter, meterSize);
	resize(testMeterGray, testMeterGray, meterSize);
	int width = testMeterGray.cols;
	int height = testMeterGray.rows;
	//������С
	int w = 55;
	int h = 39;
	
	vector<cv::Rect> textRect;
	for (int i = 0; i < height - h; i=i+2)
	{
		for (int j = 0; j < width - w; j = j + 2)
		{
			Rect winMatRect(j, i, w, h);
			Mat winMat;
			testMeterGray(winMatRect).copyTo(winMat);
			//����������ͼ������SVM
			resize(winMat, winMat, imageSize);
			vector<float> imageDescriptor;
			coumputeHog(winMat, imageDescriptor);
			Mat testDescriptor = Mat::zeros(1, imageDescriptor.size(), CV_32FC1);
			for (size_t i = 0; i < imageDescriptor.size(); i++)
			{
				testDescriptor.at<float>(0, i) = imageDescriptor[i];
			}
			float  label = mySVM->predict(testDescriptor, false);
			if (label == 1)
			{
				textRect.push_back(winMatRect);
			}
		}
	}
	/*for (int r = 0; r < textRect.size(); r++)
	{
		Rect finTextRect((int)textRect[r].x/zoomx, (int)textRect[r].y/zoomy, (int)textRect[r].width/zoomx, (int)textRect[r].height/zoomy);
		rectangle(testMeterMat, finTextRect, Scalar(0, 0, 255),4);
	}*/
	for (int r = 0; r < textRect.size(); r++)
	{
		rectangle(zoMeter, textRect[r], Scalar(0, 0, 255), 1);
	}
	imshow("result", zoMeter);
	waitKey(0);
	delete mySVM;
	return 0;
}
