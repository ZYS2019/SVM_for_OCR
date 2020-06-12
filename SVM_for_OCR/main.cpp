//SVM多分类训练测试
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

#define random(a,b) (rand()%(b-a)+a)   //生成（a,b）之间的随机数
Size imageSize = Size(64, 64);    //文本框的大小
Size meterSize = Size(800, 800);   //输入图片的大小，整个指针表图像。

//-----------------------------------------------------------
//计算图像的HOG特征
//-----------------------------------------------------------
void coumputeHog(const Mat& src, vector<float> &descriptors)
{
	HOGDescriptor myHog = HOGDescriptor(imageSize, Size(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	myHog.compute(src.clone(), descriptors, Size(1, 1), Size(0, 0));

}

//-----------------------------------------------------------
//saveFilesFullpathFromFolderInFormat函数：
//从指定文件夹路径中找到所有指定后缀名的文件，
//将路径保存在fileFullpath向量中
//输入：path 文件夹路径
//      format 后缀名
//输出：fileFullpath  path中所有指定后缀名文件的完整路径
//------------------------------------------------------------
void saveFilesFullpathFromFolderInFormat(string path, vector<string>& fileFullpath, string format) {
	_finddata_t fileInfo;
	string s;
	const char* filePath = s.assign(path).append("\\*").append(format).c_str();  //将string转化为const char*类型
	intptr_t fileHandle = _findfirst(filePath, &fileInfo); //读取第一个文件信息 文件句柄类型为long会出现不兼容[Win10平台用long声明文件句柄会crash]
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

	if (TRAIN) //是否需要训练
	{
		string s1 = "D:\\ZYS\\指针式表\\dataset\\JPEG";
		vector<string> imgFiles;
		vector<string> vTxtFiles;
		//找出文件夹中所有带jpg后缀的文件
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
			//找出与图像对应的txt文件
			string sTxtFile = imgFiles[i];
			int offindex = sTxtFile.find(".jpg");
			sTxtFile.replace(offindex, 4, ".txt");

			//读取img对应的txt文件
			fstream in;
			in.open(sTxtFile, ios::in);
			if (!in.is_open())
			{
				cout << "can not find" << sTxtFile << endl;
				system("pause");
			}
			//提取正样本，放入vecImages
			string buff;
			int row = 0;  //行数row
			while (getline(in, buff)) {
				vector<int> temp;
				//string -> char
				char *s_input = (char *)buff.c_str();
				const char * split = ",";
				//以","为分隔符拆分字符串
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
			//每张图要生成一些负样本，随机生产12张
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
		//样本计算Hog特征，训练SVM
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

	//-----------------预测单张图片-------------------------
	/*string testPath="C:\\Users\\SZJ\\Desktop\\测试用图\\negTxt.jpg";
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
	//--------------滑动窗口----------------------
	Mat testMeterMat = imread("D:\\ZYS\\指针式表\\dataset\\JPEG\\1.jpg");
	Mat testMeterGray,zoMeter; //zoMeter用来显示。

	cvtColor(testMeterMat, testMeterGray, COLOR_BGR2GRAY);
	float zoomx = (float)800 / testMeterGray.cols;
	float zoomy = (float)800 / testMeterGray.rows;
	resize(testMeterMat, zoMeter, meterSize);
	resize(testMeterGray, testMeterGray, meterSize);
	int width = testMeterGray.cols;
	int height = testMeterGray.rows;
	//滑窗大小
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
			//将窗口所在图像送入SVM
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
