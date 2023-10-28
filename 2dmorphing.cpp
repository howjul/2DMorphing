#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image_abstract.h>
using namespace dlib;

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include <iostream>
#include <vector>
using namespace std;

// ----------------------------------------------------------------------------------------

/*
// 在输入图像上使用dlib的面部关键点检测器检测68个面部关键点。
*/

void faceLandmarkDetection(dlib::array2d<unsigned char>& img, shape_predictor sp, std::vector<Point2f>& landmark)
{
	dlib::frontal_face_detector detector = get_frontal_face_detector(); //使用dlib的面部检测器
	//dlib::pyramid_up(img); //将图像放大，以便面部检测器能够检测到更多的面部细节。
	std::vector<dlib::rectangle> dets = detector(img); //使用面部检测器检测图像中的面部
	cout << "被检测到的面部个数为：" << dets.size() << endl;

	full_object_detection shape = sp(img, dets[0]);//用形状检测器来预测第一个检测到的人脸的面部特征点

	//遍历面部特征点，提取每个点的xy坐标，这些坐标以Point2f的形式存储在landmark中
	for (int i = 0; i < shape.num_parts(); ++i)
	{
		float x = shape.part(i).x();
		float y = shape.part(i).y();
		landmark.push_back(Point2f(x, y));
	}

	return;
}


/*
// 向输入图像的关键点集合添加八个关键点。
// 添加的八个关键点是图像的四个角点，以及图像四个边的中间点。
*/

void addKeypoints(std::vector<Point2f>& points, Size imgSize)
{
	points.push_back(Point2f(1, 1));
	points.push_back(Point2f(1, imgSize.height - 1));
	points.push_back(Point2f(imgSize.width - 1, imgSize.height - 1));
	points.push_back(Point2f(imgSize.width - 1, 1));
	points.push_back(Point2f(1, imgSize.height / 2));
	points.push_back(Point2f(imgSize.width / 2, imgSize.height - 1));
	points.push_back(Point2f(imgSize.width - 1, imgSize.height / 2));
	points.push_back(Point2f(imgSize.width / 2, 1));
	return;
}


/*
// 一个窗口显示多个图像的函数
*/

void inshowmany(const std::string& window_name, const std::vector<Mat>& images)
{
	int n_images = images.size();
	int size;
	int x, y;
	int w, h;

	// 检查图像的数量是否合理
	if (n_images <= 0)
	{
		cout << "Number of images are too small ..." << endl;
		return;
	}
	else if (n_images > 12)
	{
		cout << "Number of images are too large ..." << endl;
		return;
	}

	// 计算每个图像的大小
	else if (n_images == 1)
	{
		size = 400;
		w = h = 1;
	}
	else if (n_images == 2)
	{
		size = 400;
		w = 2; h = 1;
	}
	else if (n_images == 3 || n_images == 4)
	{
		size = 400;
		w = 2; h = 2;
	}
	else if (n_images == 5 || n_images == 6)
	{
		size = 300;
		w = 3; h = 2;
	}
	else if (n_images == 7 || n_images == 8)
	{
		size = 300;
		w = 3; h = 3;
	}
	else
	{
		size = 200;
		w = 4; h = 3;
	}

	// 创建一个大图像，用于显示所有的图像
	Mat DispImage = Mat::zeros(Size(100 + size * w, 30 + size * h), CV_8UC3);

	// 将每个图像复制到大图像中
	for (int i = 0, m = 20, n = 20; i < n_images; i++, m += (20 + size))
	{
		int max_size = (images[i].cols > images[i].rows) ? images[i].cols : images[i].rows;
		double scale = (double)((double)max_size / size);

		if (i % w == 0 && m != 20)
		{
			m = 20;
			n += 20 + size;
		}

		Mat imgROI = DispImage(Rect(m, n, (int)((double)images[i].cols / scale), (int)((double)images[i].rows / scale)));
		resize(images[i], imgROI, Size((int)((double)images[i].cols / scale), (int)((double)images[i].rows / scale)));
	}

	// 显示大图像  
	imshow(window_name, DispImage);

	return;
}


/*
// 在计算morph图像上关键点位置
*/

void morpKeypoints(const std::vector<Point2f>& points1, const std::vector<Point2f>& points2, std::vector<Point2f>& pointsMorph, double alpha)
{
	for (int i = 0; i < points1.size(); i++)
	{
		float x, y;
		// 对于每个关键点，计算变形图像中的关键点位置
		x = (1 - alpha) * points1[i].x + alpha * points2[i].x;
		y = (1 - alpha) * points1[i].y + alpha * points2[i].y;

		pointsMorph.push_back(Point2f(x, y));
	}
}


/*
// 对变形图像的关键点执行 Delaunay 三角剖分。
*/

//定义了用于存储三角形和它们的索引之间对应关系的结构。
struct correspondens {
	std::vector<int> index;
};

void delaunayTriangulation(const std::vector<Point2f>& points1, const std::vector<Point2f>& points2,
	std::vector<Point2f>& pointsMorph, double alpha, std::vector<correspondens>& delaunayTri, Size imgSize)
{
	//计算变形图像的关键点并打印
	morpKeypoints(points1, points2, pointsMorph, alpha);
	for (int i = 0; i < pointsMorph.size(); ++i) {
		cout << pointsMorph[i].x << " " << pointsMorph[i].y;
	}
	cout << endl;

	Rect rect(0, 0, imgSize.width, imgSize.height); //创建一个矩形，用于 Delaunay 三角剖分

	cv::Subdiv2D subdiv(rect);
	for (std::vector<Point2f>::iterator it = pointsMorph.begin(); it != pointsMorph.end(); it++) //将关键点插入到 subdiv 中
		subdiv.insert(*it);
	std::vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList); //获取矩形区域的三角形列表

	//将三角形列表转换为三角形索引列表
	for (size_t i = 0; i < triangleList.size(); ++i)
	{
		//获取三角形的三个顶点的坐标
		std::vector<Point2f> pt;
		correspondens ind;
		Vec6f t = triangleList[i];
		pt.push_back(Point2f(t[0], t[1]));
		pt.push_back(Point2f(t[2], t[3]));
		pt.push_back(Point2f(t[4], t[5]));

		//检查三角形是否在矩形内
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			int count = 0;
			for (int j = 0; j < 3; ++j)
				//对于三角形的每个顶点，检查它是否接近 pointsMorph 中的任何点。如果是，记录匹配点的索引在 ind.index 中。
				for (size_t k = 0; k < pointsMorph.size(); k++)
					if (abs(pt[j].x - pointsMorph[k].x) < 1.0 && abs(pt[j].y - pointsMorph[k].y) < 1.0)
					{
						ind.index.push_back(k);
						count++;
					}
			// 如果三角形的所有三个顶点都在 pointsMorph 中有对应的点，那么这意味着该三角形在变形区域内，代码将 ind 结构添加到 delaunayTri 向量中。
			if (count == 3)
				delaunayTri.push_back(ind);
		}
	}
}


/*
// 在一个三角形中应用仿射变换
*/
void applyAffineTransform(Mat& warpImage, Mat& src, std::vector<Point2f>& srcTri, std::vector<Point2f>& dstTri)
{
	Mat warpMat = getAffineTransform(srcTri, dstTri); //计算仿射变换矩阵
	// 将仿射变换应用于输入图像
	warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, BORDER_REFLECT_101);
}


/*
// face morphing的核心函数。
// 通过将两个输入图像中的三角形集转换到变形图像，将两个输入图像变形为变形图像。
*/

void morphTriangle(Mat& img1, Mat& img2, Mat& img, std::vector<Point2f>& t1, std::vector<Point2f>& t2, std::vector<Point2f>& t, double alpha)
{
	// 在输入图像上计算三角形的边界框
	Rect r = cv::boundingRect(t);
	Rect r1 = cv::boundingRect(t1);
	Rect r2 = cv::boundingRect(t2);

	// 计算三角形的相对位置
	std::vector<Point2f> t1Rect, t2Rect, tRect;
	std::vector<Point> tRectInt;
	for (int i = 0; i < 3; ++i)
	{
		tRect.push_back(Point2f(t[i].x - r.x, t[i].y - r.y));
		tRectInt.push_back(Point(t[i].x - r.x, t[i].y - r.y));

		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
	}

	Mat mask = Mat::zeros(r.height, r.width, CV_32FC3);
	fillConvexPoly(mask, tRectInt, Scalar(1.0, 1.0, 1.0), 16, 0); // 在mask上填充三角形

	Mat img1Rect, img2Rect;
	img1(r1).copyTo(img1Rect); // 将输入图像的三角形区域复制到img1Rect
	img2(r2).copyTo(img2Rect);

	Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type()); // 创建一个空的图像，用于存储变形后的图像
	Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());

	applyAffineTransform(warpImage1, img1Rect, t1Rect, tRect); // 将仿射变换应用于输入图像
	applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);

	Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2; // 将两个变形图像混合在一起，以获得最终的变形图像

	// 将变形图像复制到输出图像中
	multiply(imgRect, mask, imgRect);
	multiply(img(r), Scalar(1.0, 1.0, 1.0) - mask, img(r));
	img(r) = img(r) + imgRect;
}


/*
// 将两个输入图像morph为morph图像。
// 首先获取三角形集合的关键点对应关系，然后调用核心函数。
*/
void morp(Mat& img1, Mat& img2, Mat& imgMorph, double alpha, const std::vector<Point2f>& points1, const std::vector<Point2f>& points2, const std::vector<correspondens>& triangle)
{
	// 将两个输入图像morph为morph图像。
	img1.convertTo(img1, CV_32F);// 将图像转换为浮点数格式
	img2.convertTo(img2, CV_32F);

	std::vector<Point2f> points;
	morpKeypoints(points1, points2, points, alpha);// 输入为两个图像的关键点，输出为morph图像的关键点

	// 对morph图像进行Delaunay三角剖分，建立对应关系
	int x, y, z;
	int count = 0;
	for (int i = 0; i < triangle.size(); ++i)
	{
		correspondens corpd = triangle[i];
		x = corpd.index[0];
		y = corpd.index[1];
		z = corpd.index[2];
		std::vector<Point2f> t1, t2, t;
		t1.push_back(points1[x]);
		t1.push_back(points1[y]);
		t1.push_back(points1[z]);

		t2.push_back(points2[x]);
		t2.push_back(points2[y]);
		t2.push_back(points2[z]);

		t.push_back(points[x]);
		t.push_back(points[y]);
		t.push_back(points[z]);

		// 对morph图像的三角形进行morph
		morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha);
	}
}


int main(int argc, char** argv)
{
	//检查参数个数是否正确
	if (argc != 3)
	{
		cout << "参数个数错误！" << endl;
		return 0;
	}

	//-------------- 步骤1：加载两个输入图像 --------------------------------------------       
	//定义dlib的面部关键点检测器
	shape_predictor sp;
	deserialize("shape_predictor_68_face_landmarks.dat") >> sp; //加载面部关键点检测器，这里使用了已经训练好的模型

	//用dlib读取图片
	dlib::array2d<unsigned char> img1, img2;
	dlib::load_image(img1, argv[1]);
	dlib::load_image(img2, argv[2]);
	std::vector<Point2f> landmarks1, landmarks2;

	//利用opencv读取图片，并进行检查
	Mat img1CV = imread(argv[1]);
	Mat img2CV = imread(argv[2]);
	Mat img1CV_display = imread(argv[1]);
	Mat img2CV_display = imread(argv[2]);
	if (img1CV.data && img2CV.data && img1CV_display.data && img2CV_display.data) {
		cout << "图片已经通过opencv被读取" << endl;
	}
	else {
		cout << "图片读取失败" << endl;
		return -1;
	}


	//----------------- 步骤2：检测面部关键点 ---------------------------------------------
	//通过sp模型来检测img的面部关键点，并保存到landmarks中
	faceLandmarkDetection(img1, sp, landmarks1);
	faceLandmarkDetection(img2, sp, landmarks2);
	cout << "landmark1检测到的关键点数量为" << landmarks1.size() << endl;
	cout << "landmark2检测到的关键点数量为" << landmarks2.size() << endl;

	//增加8个关键点，分别是图像的四个角点，以及四条边的中点
	addKeypoints(landmarks1, img1CV.size());
	addKeypoints(landmarks2, img2CV.size());

	//分别在两个display图像中加入关键点
	for (int i = 0; i < landmarks1.size(); ++i) {
		circle(img1CV_display, landmarks1[i], 2, CV_RGB(255, 0, 0), -2);
	}
	for (int i = 0; i < landmarks2.size(); ++i) {
		circle(img2CV_display, landmarks2[i], 2, CV_RGB(255, 0, 0), -2);
	}

	//显示两个原始图像
	std::vector<Mat> imgs;
	imgs.push_back(img1CV_display);
	imgs.push_back(img2CV_display);
	inshowmany("The origin picture", imgs);
	cv::waitKey(0);

	//--------------- 步骤三：渐变 ----------------------------------------------
	std::vector<Mat> resultImage;
	resultImage.push_back(img1CV);
	cout << "add the first image" << endl;
	for (double alpha = 0.1; alpha < 1; alpha += 0.1)
	{
		Mat imgMorph = Mat::zeros(img1CV.size(), CV_32FC3); //创建一个空的morph图像
		std::vector<Point2f> pointsMorph;

		std::vector<correspondens> delaunayTri;
		//对morph图像进行Delaunay三角剖分，建立对应关系
		delaunayTriangulation(landmarks1, landmarks2, pointsMorph, alpha, delaunayTri, img1CV.size());
		cout << "done " << alpha << " delaunayTriangulation..." << delaunayTri.size() << endl;

		//对图像进行morph
		morp(img1CV, img2CV, imgMorph, alpha, landmarks1, landmarks2, delaunayTri);
		cout << "done " << alpha << " morph.........." << endl;

		resultImage.push_back(imgMorph); //将morph图像加入到结果图像中
		cout << "add the" << alpha * 10 + 1 << "image" << endl;
	}
	resultImage.push_back(img2CV);
	cout << "resultImage number is" << resultImage.size() << endl;


	//----------- 步骤四：输出图像和视频 --------------------------------
	// 保存图片
	for (int i = 0; i < resultImage.size(); ++i)
	{
		string st = argv[1];
		char t[20];
		sprintf(t, "%d", i);
		st = st + t;
		st = st + ".jpg";
		imwrite(st, resultImage[i]);
	}
	std::vector<Mat> pic;

	for (int i = 0; i < resultImage.size(); ++i)
	{
		string filename = argv[1];
		char t[20];
		sprintf(t, "%d", i);
		filename = filename + t;
		filename = filename + ".jpg";
		pic.push_back(imread(filename));
	}

	// 保存视频
	string vedioName = argv[1];
	vedioName = "." + vedioName + argv[2];
	vedioName = vedioName + ".mp4";
	VideoWriter output_src(vedioName, 0x7634706d, 5, resultImage[0].size());
	for (int i = 0; i < pic.size(); ++i)
	{

		output_src << pic[i];
	}
	cout << "vedio wrighted....." << endl;


	return 0;

}