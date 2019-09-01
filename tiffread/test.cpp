#include "octree.h"

#include <iostream>  
#include <iomanip>
#include <vector>
#include <ctime>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/opencv.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>



using namespace std;
using namespace cv;
//using namespace glm;

int main()
{
	vector<String> files;     //store the path
	vector<Mat> images;       //store the iamges
	vector<Mat> images2;
	
	glob("E:/semester2/project/code/cthead8/*.tif", files, true);

	//the number of iamges
	int num = files.size();           //
	cout << "read" << num << "images" << endl;

	/*
	//output the path of iamges
	for (int i = 0; i < num; i++){
		cout << "NO." << i + 1 << "file's path is：" << files[i] << endl;
	}
	*/

	//read the iamges
	for (int i = 0; i < num; i++){
		Mat img = imread(files[i], IMREAD_GRAYSCALE);     //read greyscale iamges
		if (img.empty()){
			cerr << files[i] << " can't be loaded!" << endl;
			continue;
		}
		images.push_back(img);          //store the images in vector
	}

	/*
	//show the iamges
	for (int i = 0; i < num; i++){
		string name = format("%d", 1 + i);    //显示图像
		imshow(name, images[i]);
	}
	*/

	//get the parameters of the iamges
	int height = num;
	int rows = images.at(1).rows;
	int cols = images.at(1).cols;
	int dims = images.at(1).dims;
	//int channels = images.at(1).channels;

	cout << "heights: " << height << endl;
	cout << "rows: " << rows << endl;
	cout << "cols: " << cols << endl;
	cout << "dims: " << dims << endl;
	//cout << "channels: " << channels << endl;

	//output iamge 
	int width = 256;
	int length = 256;

	/*
	//try to output the first iamge 
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			cout << (int)images.at(0).at<uchar>(i, j) <<" ";
			if (j == cols) {
				cout << endl;
			}
		}
	}
	*/
	//cout << (float)images.at(0).at<uchar>(0, 0) << endl;
	
	//a 3d array
	int*** array3d;
	array3d = new int**[height];
	for (int i = 0; i < height; i++){
		array3d[i] = new int*[rows];
	}
	for (int i = 0; i < height; i++){
		for (int j = 0; j < rows; j++){
			array3d[i][j] = new int[cols];
		}
	}
	
	//Mat to array
	for (int i = 0; i < height; ++i){
		for (int j = 0; j < rows; ++j){
			for (int k = 0; k < cols; ++k){
				array3d[i][j][k] = (int)images.at(i).at<uchar>(j, k);
			}
		}
	}

	//array to Mat
	for (size_t i = 0; i < height; i++)
	{
		Mat img(rows, cols, CV_8UC1, Scalar::all(0));
		for (int j = 0; j < rows; ++j)	{
			for (int k = 0; k < cols; ++k){
				img.at<uchar>(j, k) = (uchar)array3d[i][j][k];
			}
		}

		images2.push_back(img);
	}

	/*
	//show the array to Mat images
	
	for (int i = 0; i < num; i++) {
		string name = format("%d", 1 + i);    //显示图像
		imshow(name, images2[i]);
	}
	*/

	/*
	//try to output the first image in the 3d array
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			cout << array3d[0][i][j]<< " ";
			if (j == cols) {
				cout << endl;
			}
		}
	}
	*/
	/*
	//输出三维数组
	cout << "输出三维数组" << endl;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < rows; j++){
			for (int k = 0; k < cols; k++)
				cout << (int)array3d[i][j][k] << ' ';
			cout << endl;
		}
		cout << endl;
	}
	*/
	//xvalue
	float*** xvalue;
	xvalue = new float**[height];
	for (int i = 0; i < height; i++) {
		xvalue[i] = new float*[rows];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < rows; j++) {
			xvalue[i][j] = new float[cols];
		}
	}
	//yvalue
	float*** yvalue;
	yvalue = new float**[height];
	for (int i = 0; i < height; i++) {
		yvalue[i] = new float*[rows];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < rows; j++) {
			yvalue[i][j] = new float[cols];
		}
	}
	//zvalue
	float*** zvalue;
	zvalue = new float**[height];
	for (int i = 0; i < height; i++) {
		zvalue[i] = new float*[rows];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < rows; j++) {
			zvalue[i][j] = new float[cols];
		}
	}
	
	cout << fixed;
	cout << setprecision(7);

	glm::mat4 model=glm::mat4(1.0f);
	//model = glm::rotate(model, glm::radians(-180.0f), glm::vec3(1.0f, 127.0f, 49.0f));
	cout << "model: " << endl;
	cout << model[0][0] << " " << model[0][1] << " " << model[0][2] << " " << model[0][3] << " " << endl;
	cout << model[1][0] << " " << model[1][1] << " " << model[1][2] << " " << model[1][3] << " " << endl;
	cout << model[2][0] << " " << model[2][1] << " " << model[2][2] << " " << model[2][3] << " " << endl;
	cout << model[3][0] << " " << model[3][1] << " " << model[3][2] << " " << model[3][3] << " " << endl;

	glm::mat4 view=glm::mat4(1.0f);
	// 注意，我们将矩阵向我们要进行移动场景的反方向移动。
	view = glm::translate(view, glm::vec3(0.0f, 0.0f, -100.0f));
	cout << "view: " << endl;
	cout << view[0][0] << " " << view[0][1] << " " << view[0][2] << " " << view[0][3] << " " << endl;
	cout << view[1][0] << " " << view[1][1] << " " << view[1][2] << " " << view[1][3] << " " << endl;
	cout << view[2][0] << " " << view[2][1] << " " << view[2][2] << " " << view[2][3] << " " << endl;
	cout << view[3][0] << " " << view[3][1] << " " << view[3][2] << " " << view[3][3] << " " << endl;

	glm::mat4 projection = glm::mat4(1.0f);
	projection = glm::perspective(glm::radians(90.0f), (float)256 / (float)256, 0.1f, 100.0f);
	//projection = glm::ortho(0, 256, 0, 256, 0, 100);
	cout << "projection: " << endl;
	cout << projection[0][0] << " " << projection[0][1] << " " << projection[0][2] << " " << projection[0][3] << " " << endl;
	cout << projection[1][0] << " " << projection[1][1] << " " << projection[1][2] << " " << projection[1][3] << " " << endl;
	cout << projection[2][0] << " " << projection[2][1] << " " << projection[2][2] << " " << projection[2][3] << " " << endl;
	cout << projection[3][0] << " " << projection[3][1] << " " << projection[3][2] << " " << projection[3][3] << " " << endl;
	
	

	//color value
	unsigned char **outimage;
	outimage = new unsigned char*[width];
	for (size_t i = 0; i < width; i++) {
		outimage[i] = new unsigned char[length];
	}

	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < width; j++)
		{
			outimage[i][j] = 0;
		}
	}

	unsigned char **outimage2;
	outimage2 = new unsigned char*[width];
	for (size_t i = 0; i < width; i++) {
		outimage2[i] = new unsigned char[length];
	}

	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < width; j++)
		{
			outimage2[i][j] = 0;
		}
	}

	unsigned char **outimage3;
	outimage3 = new unsigned char*[width];
	for (size_t i = 0; i < width; i++) {
		outimage3[i] = new unsigned char[length];
	}

	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < width; j++)
		{
			outimage3[i][j] = 0;
		}
	}

	unsigned char **outimage4;
	outimage4 = new unsigned char*[width];
	for (size_t i = 0; i < width; i++) {
		outimage4[i] = new unsigned char[length];
	}

	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < width; j++)
		{
			outimage4[i][j] = 0;
		}
	}

	unsigned char **outimage5;
	outimage5 = new unsigned char*[width];
	for (size_t i = 0; i < width; i++) {
		outimage5[i] = new unsigned char[length];
	}

	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < width; j++)
		{
			outimage5[i][j] = 0;
		}
	}

	unsigned char **outimage6;
	outimage6 = new unsigned char*[width];
	for (size_t i = 0; i < width; i++) {
		outimage6[i] = new unsigned char[length];
	}

	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < width; j++)
		{
			outimage6[i][j] = 0;
		}
	}

	//depth value
	float **outimagedepth;
	outimagedepth = new float*[width];
	for (size_t i = 0; i < width; i++) {
		outimagedepth[i] = new float[length];
	}

	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < width; j++)
		{
			outimagedepth[i][j] = 1000;
		}
	}

	//window parameter
	float xwmax = -10000;
	float xwmin = 10000;
	float ywmax = -10000;
	float ywmin = 10000;
	//viewport parameter
	float xvmax = 255;
	float xvmin = 0;
	float yvmax = 255;
	float yvmin = 0;

	/*
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < cols; k++) {
				glm::vec4 oldposition(j*1.0f, k*1.0f, i*1.0f, 1);
				//cout << oldposition.x <<" "<< oldposition.y <<" "<< oldposition.z << endl;
				glm::vec4 newposition(0.0f, 0.0f, 0.0f, 0.0f);
				newposition = projection * view * model * oldposition;
				float x = newposition.x;
				float y = newposition.y;
				float z = newposition.z;
				//cout <<j<<","<<k<<","<<i<<"的新坐标："<< x << " " << y << " " << z << endl;
				xvalue[i][j][k] = x;
				yvalue[i][j][k] = y;
				zvalue[i][j][k] = z;

				//find max and min for window 
				if (x > xwmax){
					xwmax = x;
				}
				if (x < xwmin){
					xwmin = x;
				}
				if (y > ywmax) {
					ywmax = y;
				}
				if (y < ywmin) {
					ywmin = y;
				}
			}
		}
	}

	//output window parameter
	cout << "xwmax: " << xwmax << endl;
	cout << "xwmin: " << xwmin << endl;
	cout << "ywmax: " << ywmax << endl;
	cout << "ywmin: " << ywmin << endl;


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < cols; k++) {

				float x = xvalue[i][j][k];
				float y = yvalue[i][j][k];
				float z = zvalue[i][j][k];

				float xv = xvmin + (x - xwmin)*(xvmax - xvmin) / (xwmax - xwmin);
				float yv = yvmin + (y - ywmin)*(yvmax - yvmin) / (ywmax - ywmin);

				int ximage = round(xv);
				int yimage = round(yv);

				if (z < outimagedepth[ximage][yimage])
				{
					outimagedepth[ximage][yimage] = z;
					outimage[ximage][yimage] = array3d[i][j][k];
				}
			}
		}
	}
	*/

	/*
	for (size_t i = 0; i < 1; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			cout << outimage[i][j] << endl;
		}
	}
	*/

	/*
	//array to Mat
	Mat outputimage(length, width, CV_8UC1, Scalar::all(0));
	for (size_t i = 0; i < length; i++)
	{
		for (size_t j = 0; j < length; j++)
		{
			outputimage.at<uchar>(i, j) = (uchar)outimage[i][j];
		}
	}
	string name = "output iamge";
	imshow(name,outputimage);
	*/

	/*
	for (int j = 0; j < 10; j++) {
		cout << "slice1的x值：" << zvalue[1][0][j] <<endl;
		}
	*/

	//line box test
	/*
	glm::vec3 eyeposition(1.5, 1.5, 5);
	for (size_t ximage = 0; ximage < 4; ximage++) {
		for (size_t yimage = 0; yimage < 4; yimage++) {
			glm::vec3 pointposition(ximage*1.0f, yimage*1.0f, 0.0f);
			glm::vec3 directionvector = pointposition - eyeposition;
			cout << "pixel:" << ximage << " " << yimage << ":" << endl;
			for (size_t i = 1; i < 5; i++) {
				for (size_t j = 0; j < 4; j++) {
					for (size_t k = 0; k < 4; k++) {
						glm::vec3 voxelposition(j*1.0f, k*1.0f, i*1.0f);
						float x = j;
						float y = k;
						float z = i;

						//bool
						bool intersect = true;
						glm::vec3 vmin((x - 0.5)*1.0f, (y - 0.5)*1.0f, (z - 0.5)*1.0f);
						glm::vec3 vmax((x + 0.5)*1.0f, (y + 0.5)*1.0f, (z + 0.5)*1.0f);


						float tmin = (vmin.x - eyeposition.x) / directionvector.x;
						float tmax = (vmax.x - eyeposition.x) / directionvector.x;
						if (tmin > tmax) {
							float temp = tmax;
							tmax = tmin;
							tmin = temp;
						}

						float tymin = (vmin.y - eyeposition.y) / directionvector.y;
						float tymax = (vmax.y - eyeposition.y) / directionvector.y;
						if (tymin > tymax) {
							float temp = tymax;
							tymax = tymin;
							tymin = temp;
						}

						if ((tmin > tymax) || (tymin > tmax)) {
							intersect = false;
						}

						if (tymin > tmin) {
							tmin = tymin;
						}
						if (tymax < tmax) {
							tmax = tymax;
						}


						float tzmin = (vmin.z - eyeposition.z) / directionvector.z;
						float tzmax = (vmax.z - eyeposition.z) / directionvector.z;
						if (tzmin > tzmax) {
							float temp = tzmax;
							tzmax = tzmin;
							tzmin = temp;
						}

						if ((tmin > tzmax) || (tzmin > tmax)) {
							intersect = false;
						}

						if (tzmin > tmin) {
							tmin = tzmin;
						}
						if (tzmax < tmax) {
							tmax = tzmax;
						}

						if (intersect) {
							cout << j << " " << k << " " << i << endl;
						}
					}
				}
			}
		}
	}
	*/
      
float maxIntensity = 0;
float maxIntensity2 = 0;
float maxIntensity3 = 0;
float maxIntensity4 = 0;
float maxIntensity5 = 0;
float maxIntensity6 = 0;

Octree<int> octree(256);

for (int i = 0; i < height; ++i) {
	for (int j = 0; j < rows; ++j) {
		for (int k = 0; k < cols; ++k) {
			octree(i, j, k) = array3d[i][j][k];//z x y 
		}
	}
}
    /*
    //octree
    for (size_t x = 0; x < width - 1; x++) {
    	for (size_t y = 0; y < length - 1; y++) {
    		glm::vec3 p(x*1.0f + 0.5f, y*1.0f + 0.5f, 98.0f);
    		glm::vec3 q(x*1.0f + 0.5f, y*1.0f + 0.5f, 0.0f);
    
    		glm::vec3 v = q - p;
    
    		float tStep = 0.01;
    
    		float totalIntensity = 0.00;
    		//cout << "pixel: " << x << " " << y << endl;
    
    		for (float t = 0.01; t < 1; t = t + tStep) {
    			glm::vec3 r = p + v * t;
    			float x = r.x;
    			float y = r.y;
    			float z = r.z;
    			//cout << "sample point: " << x << " " << y << " " << z << endl;
    			int x0 = floor(x);
    			int y0 = floor(y);
    			int z0 = floor(z);
    			int x1 = x0 + 1;
    			int y1 = y0 + 1;
    			int z1 = z0 + 1;
    			float xd = (x - x0) / (x1 - x0);
    			float yd = (y - y0) / (y1 - y0);
    			float zd = (z - z0) / (z1 - z0);
    			float c = octree(z0 ,x0, y0) * (1 - xd)*(1 - yd)*(1 - zd) +
					      octree(z0, x1, y0) * xd*(1 - yd)*(1 - zd) +
					      octree(z0, x0, y1) * (1 - xd)*yd*(1 - zd) +
					      octree(z1, x0, y0) * (1 - yd)*(1 - yd)*zd +
					octree(z1, x1, y0) * xd*(1 - yd)*zd +
					octree(z1, x0, y1) * (1 - xd)*yd*zd +
					octree(z0, x1, y1) * xd*yd*(1 - zd) +
					octree(z1, x1, y1) * xd*yd*zd;
    			//cout << "value: " << c << endl;
    
    			float alpha = c/255;
    			float intensity = c;
    
    			totalIntensity = totalIntensity * (1 - alpha) + intensity * alpha;
    
    		}
    
    		outimage[x][y] = totalIntensity;
    		//cout << "intensity" << totalIntensity << endl;
    		if (totalIntensity > maxIntensity)
    		{
    			maxIntensity = totalIntensity;
    			//cout << "maxIntensity changed to: " << maxIntensity << endl;
    		}
    	}
    }
    */

    //time
    clock_t start, end;
    clock_t start2, end2;
    clock_t start3, end3;
    clock_t start4, end4;
    clock_t start5, end5;
    clock_t start6, end6;
	/*
	for every pixel in the imageplane
		vec3 startpoint;
		vec3 endpoint;

		vec3 directionvector = endpoint-startpoint;

		float tStep = 0.01;// sample steplength
		float totalIntensity = 0.00;//output value of the ray
		
		for (float t = 0.01; t < 1; t = t + tStep) {//sample along the ray
			glm::vec3 sample = startpoint + directionvector * t;
			float x = sample.x;
			float y = sample.y;
			float z = sample.z;
			int x0 = floor(x);
			int y0 = floor(y);
			int z0 = floor(z);
			int x1 = x0 + 1;
			int y1 = y0 + 1;
			int z1 = z0 + 1;
			float xd = (x - x0) / (x1 - x0);
			float yd = (y - y0) / (y1 - y0);
			float zd = (z - z0) / (z1 - z0);
			float c = array3d[z0][x0][y0] * (1 - xd)*(1 - yd)*(1 - zd) +
				array3d[z0][x1][y0] * xd*(1 - yd)*(1 - zd) +
				array3d[z0][x0][y1] * (1 - xd)*yd*(1 - zd) +
				array3d[z1][x0][y0] * (1 - yd)*(1 - yd)*zd +
				array3d[z1][x1][y0] * xd*(1 - yd)*zd +
				array3d[z1][x0][y1] * (1 - xd)*yd*zd +
				array3d[z0][x1][y1] * xd*yd*(1 - zd) +
				array3d[z1][x1][y1] * xd*yd*zd;
	
			float alpha = c / 255;
			float intensity = c;

			totalIntensity = totalIntensity * (1 - alpha) + intensity * alpha;
		}
	}
	*/
	///*
	//direct volume rendering
    //direction 1
	start = clock();
	for (size_t x = 0; x < width-1; x++){
		for (size_t y = 0; y < length-1; y++){
			glm::vec3 p(x*1.0f+0.5f, y*1.0f+0.5f, 98.0f);
			glm::vec3 q(x*1.0f+0.5f, y*1.0f+0.5f, 0.0f);

			glm::vec3 v = q - p;

			float tStep = 0.01;

			float totalIntensity = 0.00;
			//cout << "pixel: " << x << " " << y << endl;

			for (float t = 0.01; t < 1; t=t+tStep){
				glm::vec3 r = p + v * t;
				float x = r.x;
				float y = r.y;
				float z = r.z;
				//cout << "sample point: " << x << " " << y << " " << z << endl;
				int x0 = floor(x);
				int y0 = floor(y);
				int z0 = floor(z);
				int x1 = x0 + 1;
				int y1 = y0 + 1;
				int z1 = z0 + 1;
				float xd = (x - x0) / (x1 - x0);
				float yd = (y - y0) / (y1 - y0);
				float zd = (z - z0) / (z1 - z0);
				float c = array3d[z0][x0][y0] * (1 - xd)*(1 - yd)*(1 - zd) +
					      array3d[z0][x1][y0] * xd*(1 - yd)*(1 - zd) +
					      array3d[z0][x0][y1] * (1 - xd)*yd*(1 - zd) +
				   	      array3d[z1][x0][y0] * (1 - yd)*(1 - yd)*zd +
					      array3d[z1][x1][y0] * xd*(1 - yd)*zd +
					      array3d[z1][x0][y1] * (1 - xd)*yd*zd +
					      array3d[z0][x1][y1] * xd*yd*(1 - zd) +
					      array3d[z1][x1][y1] * xd*yd*zd;
				//cout << "value: " << c << endl;

				//float alpha = 0.5;
				float alpha = c/255;
				float intensity = c;

				totalIntensity = totalIntensity * (1 - alpha) + intensity * alpha;
				
			}

			//cout << "intensity" << totalIntensity << endl;
			if (totalIntensity > maxIntensity)
			{
				maxIntensity = totalIntensity;
				//cout << "maxIntensity changed to: " << maxIntensity << endl;
			}

			//outimage[x][y] = totalIntensity;
			outimage[x][y] = totalIntensity / maxIntensity * 255;

			
		}
	}
	end = clock();
	//*/

	cout << "maxIntensity: " << maxIntensity << endl;

	//direction 2
	start2 = clock();
	for (size_t x = 0; x < width - 1; x++) {
		for (size_t y = 0; y < length - 1; y++) {
			glm::vec3 p(x*1.0f + 0.5f, y*1.0f + 0.5f, 0.0f);
			glm::vec3 q(x*1.0f + 0.5f, y*1.0f + 0.5f, 98.0f);

			glm::vec3 v = q - p;

			float tStep = 0.01;

			float totalIntensity = 0.00;
			//cout << "pixel: " << x << " " << y << endl;

			for (float t = 0.01; t < 1; t = t + tStep) {
				glm::vec3 r = p + v * t;
				float x = r.x;
				float y = r.y;
				float z = r.z;
				//cout << "sample point: " << x << " " << y << " " << z << endl;
				int x0 = floor(x);
				int y0 = floor(y);
				int z0 = floor(z);
				int x1 = x0 + 1;
				int y1 = y0 + 1;
				int z1 = z0 + 1;
				float xd = (x - x0) / (x1 - x0);
				float yd = (y - y0) / (y1 - y0);
				float zd = (z - z0) / (z1 - z0);
				float c = array3d[z0][x0][y0] * (1 - xd)*(1 - yd)*(1 - zd) +
					array3d[z0][x1][y0] * xd*(1 - yd)*(1 - zd) +
					array3d[z0][x0][y1] * (1 - xd)*yd*(1 - zd) +
					array3d[z1][x0][y0] * (1 - yd)*(1 - yd)*zd +
					array3d[z1][x1][y0] * xd*(1 - yd)*zd +
					array3d[z1][x0][y1] * (1 - xd)*yd*zd +
					array3d[z0][x1][y1] * xd*yd*(1 - zd) +
					array3d[z1][x1][y1] * xd*yd*zd;
				//cout << "value: " << c << endl;

				//float alpha = 0.5;
				float alpha = c / 255;
				float intensity = c;

				totalIntensity = totalIntensity * (1 - alpha) + intensity * alpha;

			}

			//cout << "intensity" << totalIntensity << endl;
			if (totalIntensity > maxIntensity2)
			{
				maxIntensity2 = totalIntensity;
				//cout << "maxIntensity changed to: " << maxIntensity << endl;
			}

			//outimage2[x][y] = totalIntensity;
			outimage2[x][y] = totalIntensity / maxIntensity * 255;

			
		}
	}
	end2 = clock();
	//*/

	cout << "maxIntensity2: " << maxIntensity2 << endl;

	//direction 3
	start3 = clock();
	for (size_t i = 0; i < width - 1; i++) {
		for (size_t j = 0; j < length - 1; j++) {
			float istep = 98.0 / 257.0;
			glm::vec3 p(j*1.0f + 0.5f,255.0f, (98 - (i + 1)*istep)*1.0f);
			glm::vec3 q(j*1.0f + 0.5f,0.0f, (98 - (i + 1)*istep)*1.0f);

			glm::vec3 v = q - p;

			float tStep = 0.003;

			float totalIntensity = 0.00;
			//cout << "pixel: " << x << " " << y << endl;

			for (float t = 0.01; t < 1; t = t + tStep) {
				glm::vec3 r = p + v * t;
				float x = r.x;
				float y = r.y;
				float z = r.z;
				//cout << "sample point: " << x << " " << y << " " << z << endl;
				int x0 = floor(x);
				int y0 = floor(y);
				int z0 = floor(z);
				int x1 = x0 + 1;
				int y1 = y0 + 1;
				int z1 = z0 + 1;
				float xd = (x - x0) / (x1 - x0);
				float yd = (y - y0) / (y1 - y0);
				float zd = (z - z0) / (z1 - z0);
				float c = array3d[z0][x0][y0] * (1 - xd)*(1 - yd)*(1 - zd) +
					array3d[z0][x1][y0] * xd*(1 - yd)*(1 - zd) +
					array3d[z0][x0][y1] * (1 - xd)*yd*(1 - zd) +
					array3d[z1][x0][y0] * (1 - yd)*(1 - yd)*zd +
					array3d[z1][x1][y0] * xd*(1 - yd)*zd +
					array3d[z1][x0][y1] * (1 - xd)*yd*zd +
					array3d[z0][x1][y1] * xd*yd*(1 - zd) +
					array3d[z1][x1][y1] * xd*yd*zd;
				//cout << "value: " << c << endl;

				//float alpha = 0.5;
				float alpha = c / 255;
				float intensity = c;

				totalIntensity = totalIntensity * (1 - alpha) + intensity * alpha;

			}

			//cout << "intensity" << totalIntensity << endl;

			if (totalIntensity > maxIntensity3)
			{
				maxIntensity3 = totalIntensity;
				//cout << "maxIntensity changed to: " << maxIntensity << endl;
			}

			//outimage3[i][j] = totalIntensity;
			outimage3[i][j] = totalIntensity / maxIntensity3 * 255;

			
			
		}
	}
	end3 = clock();
	//*/
	cout << "maxIntensity3: " << maxIntensity3 << endl;


	//direction 4
	start4 = clock();
	for (size_t i = 0; i < width - 1; i++) {
		for (size_t j = 0; j < length - 1; j++) {
			float istep = 98.0 / 257.0;
			glm::vec3 p(j*1.0f + 0.5f, 0.0f, (i + 1)*istep*1.0f);
			glm::vec3 q(j*1.0f + 0.5f, 255.0f, (i + 1)*istep*1.0f);

			glm::vec3 v = q - p;

			float tStep = 0.003;

			float totalIntensity = 0.00;
			//cout << "pixel: " << x << " " << y << endl;

			for (float t = 0.01; t < 1; t = t + tStep) {
				glm::vec3 r = p + v * t;
				float x = r.x;
				float y = r.y;
				float z = r.z;
				//cout << "sample point: " << x << " " << y << " " << z << endl;
				int x0 = floor(x);
				int y0 = floor(y);
				int z0 = floor(z);
				int x1 = x0 + 1;
				int y1 = y0 + 1;
				int z1 = z0 + 1;
				float xd = (x - x0) / (x1 - x0);
				float yd = (y - y0) / (y1 - y0);
				float zd = (z - z0) / (z1 - z0);
				float c = array3d[z0][x0][y0] * (1 - xd)*(1 - yd)*(1 - zd) +
					array3d[z0][x1][y0] * xd*(1 - yd)*(1 - zd) +
					array3d[z0][x0][y1] * (1 - xd)*yd*(1 - zd) +
					array3d[z1][x0][y0] * (1 - yd)*(1 - yd)*zd +
					array3d[z1][x1][y0] * xd*(1 - yd)*zd +
					array3d[z1][x0][y1] * (1 - xd)*yd*zd +
					array3d[z0][x1][y1] * xd*yd*(1 - zd) +
					array3d[z1][x1][y1] * xd*yd*zd;
				//cout << "value: " << c << endl;

				//float alpha = 0.5;
				float alpha = c/255;
				float intensity = c;

				totalIntensity = totalIntensity * (1 - alpha) + intensity * alpha;

			}
			//cout << "intensity" << totalIntensity << endl;
			
			if (totalIntensity > maxIntensity4)
			{
				maxIntensity4 = totalIntensity;
				//cout << "maxIntensity changed to: " << maxIntensity << endl;
			}

			//outimage4[i][j] = totalIntensity;
			outimage4[i][j] = totalIntensity / maxIntensity4 * 255;
			
		}
	}
	end4 = clock();
	//*/
	cout << "maxIntensity4: " << maxIntensity4 << endl;

	//direction 5
	start5 = clock();
	for (size_t i = 0; i < width - 1; i++) {
		for (size_t j = 0; j < length - 1; j++) {
			float istep = 98.0 / 256.0;
			glm::vec3 p(0.0f, j*1.0f + 0.5f, (i + 1)*istep*1.0f);
			glm::vec3 q(255.0f, j*1.0f + 0.5f, (i + 1)*istep*1.0f);

			glm::vec3 v = q - p;

			float tStep = 0.003;

			float totalIntensity = 0.00;
			//cout << "pixel: " << x << " " << y << endl;

			for (float t = 0.01; t < 1; t = t + tStep) {
				glm::vec3 r = p + v * t;
				float x = r.x;
				float y = r.y;
				float z = r.z;
				//cout << "sample point: " << x << " " << y << " " << z << endl;
				int x0 = floor(x);
				int y0 = floor(y);
				int z0 = floor(z);
				int x1 = x0 + 1;
				int y1 = y0 + 1;
				int z1 = z0 + 1;
				float xd = (x - x0) / (x1 - x0);
				float yd = (y - y0) / (y1 - y0);
				float zd = (z - z0) / (z1 - z0);
				float c = array3d[z0][x0][y0] * (1 - xd)*(1 - yd)*(1 - zd) +
					array3d[z0][x1][y0] * xd*(1 - yd)*(1 - zd) +
					array3d[z0][x0][y1] * (1 - xd)*yd*(1 - zd) +
					array3d[z1][x0][y0] * (1 - yd)*(1 - yd)*zd +
					array3d[z1][x1][y0] * xd*(1 - yd)*zd +
					array3d[z1][x0][y1] * (1 - xd)*yd*zd +
					array3d[z0][x1][y1] * xd*yd*(1 - zd) +
					array3d[z1][x1][y1] * xd*yd*zd;
				//cout << "value: " << c << endl;

				//float alpha = 0.5;
				float alpha = c / 255;
				float intensity = c;

				totalIntensity = totalIntensity * (1 - alpha) + intensity * alpha;

			}
			//cout << "intensity" << totalIntensity << endl;

			if (totalIntensity > maxIntensity5)
			{
				maxIntensity5 = totalIntensity;
				//cout << "maxIntensity changed to: " << maxIntensity << endl;
			}

			outimage5[i][j] = totalIntensity;
			//outimage5[i][j] = totalIntensity / maxIntensity5 * 255;

		}
	}
	end5 = clock();
	//*/
	cout << "maxIntensity5: " << maxIntensity5 << endl;

	//direction 6
	start6 = clock();
	for (size_t i = 0; i < width - 1; i++) {
		for (size_t j = 0; j < length - 1; j++) {
			float istep = 98.0 / 256.0;
			glm::vec3 p(255.0f, j*1.0f + 0.5f, ((i + 1)*istep)*1.0f);
			glm::vec3 q(0.0f, j*1.0f + 0.5f, ((i + 1)*istep)*1.0f);

			glm::vec3 v = q - p;

			float tStep = 0.003;

			float totalIntensity = 0.00;
			//cout << "pixel: " << x << " " << y << endl;

			for (float t = 0.01; t < 1; t = t + tStep) {
				glm::vec3 r = p + v * t;
				float x = r.x;
				float y = r.y;
				float z = r.z;
				//cout << "sample point: " << x << " " << y << " " << z << endl;
				int x0 = floor(x);
				int y0 = floor(y);
				int z0 = floor(z);
				int x1 = x0 + 1;
				int y1 = y0 + 1;
				int z1 = z0 + 1;
				float xd = (x - x0) / (x1 - x0);
				float yd = (y - y0) / (y1 - y0);
				float zd = (z - z0) / (z1 - z0);
				float c = array3d[z0][x0][y0] * (1 - xd)*(1 - yd)*(1 - zd) +
					array3d[z0][x1][y0] * xd*(1 - yd)*(1 - zd) +
					array3d[z0][x0][y1] * (1 - xd)*yd*(1 - zd) +
					array3d[z1][x0][y0] * (1 - yd)*(1 - yd)*zd +
					array3d[z1][x1][y0] * xd*(1 - yd)*zd +
					array3d[z1][x0][y1] * (1 - xd)*yd*zd +
					array3d[z0][x1][y1] * xd*yd*(1 - zd) +
					array3d[z1][x1][y1] * xd*yd*zd;
				//cout << "value: " << c << endl;

				//float alpha = 0.5;
				float alpha = c / 255;
				float intensity = c;

				totalIntensity = totalIntensity * (1 - alpha) + intensity * alpha;

			}
			//cout << "intensity" << totalIntensity << endl;

			if (totalIntensity > maxIntensity6)
			{
				maxIntensity6 = totalIntensity;
				//cout << "maxIntensity changed to: " << maxIntensity << endl;
			}

			//outimage6[i][j] = totalIntensity;
			outimage6[i][j] = totalIntensity / maxIntensity6 * 255;

		}
	}
	end6 = clock();
	//*/
	cout << "maxIntensity6: " << maxIntensity6 << endl;


	/*
	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < length; j++) {
			outimage[i][j] = outimage[i][j] / maxIntensity * 255;
		}
	}
	*/
	//array to Mat 1
	Mat outputimage(length, width, CV_8UC1, Scalar::all(0));
	for (size_t i = 0; i < length; i++){
		for (size_t j = 0; j < length; j++){
			outputimage.at<uchar>(i, j) = (uchar)outimage[i][j];
		}
	}
	string name = "output iamge";
	imshow(name,outputimage);


	//array to Mat 2
	Mat outputimage2(length, width, CV_8UC1, Scalar::all(0));
	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < length; j++) {
			outputimage2.at<uchar>(i, j) = (uchar)outimage2[i][j];
		}
	}
	string name2 = "output iamge2";
	imshow(name2, outputimage2);

	//array to Mat 3
	Mat outputimage3(length, width, CV_8UC1, Scalar::all(0));
	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < length; j++) {
			outputimage3.at<uchar>(i, j) = (uchar)outimage3[i][j];
		}
	}
	string name3 = "output iamge3";
	imshow(name3, outputimage3);

	//array to Mat 4
	Mat outputimage4(length, width, CV_8UC1, Scalar::all(0));
	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < length; j++) {
			outputimage4.at<uchar>(i, j) = (uchar)outimage4[i][j];
		}
	}
	string name4 = "output iamge4";
	imshow(name4, outputimage4);

	//array to Mat 5
	Mat outputimage5(length, width, CV_8UC1, Scalar::all(0));
	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < length; j++) {
			outputimage5.at<uchar>(i, j) = (uchar)outimage5[i][j];
		}
	}
	string name5 = "output iamge5";
	imshow(name5, outputimage5);

	//array to Mat 6
	Mat outputimage6(length, width, CV_8UC1, Scalar::all(0));
	for (size_t i = 0; i < length; i++) {
		for (size_t j = 0; j < length; j++) {
			outputimage6.at<uchar>(i, j) = (uchar)outimage6[i][j];
		}
	}
	string name6 = "output iamge6";
	imshow(name6, outputimage6);
	/*
	for (size_t ximage = 0; ximage < length; ximage++) {
		for (size_t yimage = 0; yimage < width; yimage++) {
			//point on the far plane
			glm::vec3 point(ximage*1.0f, yimage*1.0f, -101.0f);
			//eye direction
			glm::vec3 direction = point - eye;

			cout << "pixel:" << ximage << " " << yimage << ":" << endl;

			for (size_t i = 0; i < height; i++) {
				for (size_t j = 0; j < rows; j++) {
					for (size_t k = 0; k < cols; k++) {
						glm::vec4 oldposition(j*1.0f, k*1.0f, i*1.0f, 1);
						//cout << oldposition.x <<" "<< oldposition.y <<" "<< oldposition.z << endl;
						glm::vec4 newposition(0.0f, 0.0f, 0.0f, 0.0f);
						newposition = view * model * oldposition;
						float x = newposition.x;
						float y = newposition.y;
						float z = newposition.z;
						//cout <<j<<","<<k<<","<<i<<"的新坐标："<< x << " " << y << " " << z << endl;
						xvalue[i][j][k] = x;
						yvalue[i][j][k] = y;
						zvalue[i][j][k] = z;

						//bool
						bool intersect = true;

						//bounds
						glm::vec3 bounds[2];
						glm::vec3 vmin((x - 0.5)*1.0f, (y - 0.5)*1.0f, (z - 0.5)*1.0f);
						glm::vec3 vmax((x + 0.5)*1.0f, (y + 0.5)*1.0f, (z + 0.5)*1.0f);

						float tmin = (vmin.x - eye.x) / direction.x;
						float tmax = (vmax.x - eye.x) / direction.x;
						if (tmin > tmax) {
							float temp = tmax;
							tmax = tmin;
							tmin = temp;
						}

						float tymin = (vmin.y - eye.y) / direction.y;
						float tymax = (vmax.y - eye.y) / direction.y;
						if (tymin > tymax) {
							float temp = tymax;
							tymax = tymin;
							tymin = temp;
						}

						if ((tmin > tymax) || (tymin > tmax)) {
							intersect = false;
						}

						if (tymin > tmin) {
							tmin = tymin;
						}
						if (tymax < tmax) {
							tmax = tymax;
						}


						float tzmin = (vmin.z - eye.z) / direction.z;
						float tzmax = (vmax.z - eye.z) / direction.z;
						if (tzmin > tzmax) {
							float temp = tzmax;
							tzmax = tzmin;
							tzmin = temp;
						}

						if ((tmin > tzmax) || (tzmin > tmax)) {
							intersect = false;
						}

						if (tzmin > tmin) {
							tmin = tzmin;
						}
						if (tzmax < tmax) {
							tmax = tzmax;
						}

						if (intersect) {
							cout << j << " " << k << " " << i << endl;
						}

					}
				}
			}
		}
	}
	*/

	cout << "total time1: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
	cout << "total time2: " << (double)(end2 - start2) / CLOCKS_PER_SEC << "s" << endl;
	cout << "total time3: " << (double)(end3 - start3) / CLOCKS_PER_SEC << "s" << endl;
	cout << "total time4: " << (double)(end4 - start4) / CLOCKS_PER_SEC << "s" << endl;
	cout << "total time5: " << (double)(end5 - start5) / CLOCKS_PER_SEC << "s" << endl;
	cout << "total time6: " << (double)(end6 - start6) / CLOCKS_PER_SEC << "s" << endl;
	// 等待6000 ms后窗口自动关闭    
	waitKey(600000);
	system("pause");

	return 0;
}