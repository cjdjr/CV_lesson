#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv.hpp>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <iostream>
using namespace cv;
using namespace std;

void readImages(string dirName,vector<Mat> &images)
{
    cout<<"reading images..."<<endl;
    string pattern_jpg = dirName+"*.jpg";
    vector<cv::String> image_files;
    cv::glob(pattern_jpg, image_files);
    images.resize(image_files.size());
    for(int i=0;i<image_files.size();++i)
    {
        cout << image_files[i] << endl;
   		Mat tmp=imread(image_files[i]);
        tmp.copyTo(images[i]);
    }
}
double match(const Mat& img,const Mat& templ,int is_show=0,int id=1)
{
    if(templ.cols>img.cols||templ.rows>img.rows) return 0;
    Mat result;
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    result.create(result_cols, result_rows, CV_32FC1);

    matchTemplate(img, templ, result, CV_TM_CCOEFF_NORMED);
    //cout<<result<<endl;
    //normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
    //cout<<result<<endl;
    if(is_show) imshow("light"+to_string(id),result);
    double minVal = 1000000;
    double maxVal=-1000000;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;
    //cout << "匹配度：" << minVal << endl;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());


    //cout << "匹配度：" << minVal << endl;

    matchLoc = maxLoc;
    Mat paint=img.clone();
    circle(paint, Point(matchLoc.x + templ.cols/2, matchLoc.y + templ.rows/2), 2, Scalar(0, 0, 255),-1);//在图像中画出特征点，2是圆的半径
    rectangle(paint, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 1, 1, 0);

    if(is_show)
    {
        imshow("img"+to_string(id), paint);
        cout<<id<<":"<<endl;
        cout<<"匹配度："<<maxVal<<endl;
    }
    return maxVal;
}
void match_templates(const vector<Mat> &templates,int id)
{
    //imshow("ce"+to_string(id),templates[id]);

    Mat img=imread("./faceSamples/0439.jpg");
    Mat templ=templates[id].clone();
    double mx=0;
    int argmx=0;
    for(int i=0;;++i)
    {
        double num=match(img,templ);
        cout<<"i="<<i<<"  match value ="<<num<<endl;
        if(num>mx) mx=num,argmx=i;
        resize(templ, templ, Size(),0.9,0.9);
        if(templ.cols*20<img.cols||templ.rows*20<img.rows) break;
    }
    cout<<"argmax="<<argmx<<endl;
    templ=templates[id].clone();
    resize(templ,templ,Size(),pow(0.9,argmx),pow(0.9,argmx));
    imshow("ce"+to_string(id),templ);
    match(img,templ,1,id);
}
int main(int argc, char **argv)
{
    string dirName = "faceSamples/";
    // Read images in the directory
   /* vector<Mat> images;
    readImages(dirName, images);*/

    dirName="templates/";
    vector<Mat> templates;
    readImages(dirName,templates); //0673

    for(int i=0;i<1;++i) match_templates(templates,i);

    cout<<clock()<<endl;
    waitKey(0);

    return 0;
}
