#ifndef _EigenFeature

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv.hpp>
#include <cmath>
#include <sstream>
#include <limits>
#include <cstdlib>
#include <iostream>
#include <assert.h>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
using namespace cv;
using namespace std;
using namespace rapidjson;  //引入rapidjson命名空间
double sqr(double x)
{
    return x*x;
}
struct EigenFeature
{
private:
    int NUM_EIGEN_FACES=3;

    PCA pca;
    Mat sigma;
    vector<Mat> templates;
    double calDifference(const Mat &img,int);
    Mat createDataMatrix();

public:
    int height=15,width=24;
    EigenFeature(){}
    EigenFeature(int num):NUM_EIGEN_FACES(num){} ;
    void readTemplates(string dirName,int leftup,int rightdown);
    void align_templ_size(int x,int y);
    void calEigen();
    double match(const Mat& img,int is_show,int &xx,int &yy);

};

void EigenFeature::readTemplates(string dirName,int leftup,int rightdown)
{
    string pattern_jpg = dirName+"*.jpg";
    string pattern_json = dirName+"*.json";
    vector<cv::String> image_files;
    vector<cv::String> image_jsons;
    cv::glob(pattern_jpg, image_files);
    cv::glob(pattern_json,image_jsons);
    templates.resize(image_files.size());
    for(int i=0;i<image_files.size();++i)
    {
        std::ifstream in(image_jsons[i]);
        std::ostringstream t;
        t << in.rdbuf();
        std::string content = t.str();
        Document document;
        document.Parse<0>(content.c_str());
        int x1,y1,x2,y2;
        y1=int(document["shapes"][leftup]["points"][0].GetArray()[0].GetFloat());
        x1=int(document["shapes"][leftup]["points"][0].GetArray()[1].GetFloat());
        y2=int(document["shapes"][rightdown]["points"][0].GetArray()[0].GetFloat());
        x2=int(document["shapes"][rightdown]["points"][0].GetArray()[1].GetFloat());
        Mat tmp(imread(image_files[i],IMREAD_GRAYSCALE),Rect(y1,x1,y2-y1,x2-x1));
        tmp.copyTo(templates[i]);
    }
}
void EigenFeature::align_templ_size(int x,int y)
{
    height=x,width=y;
    for(auto &x:templates) resize(x,x,Size(width,height));
    sigma=Mat(height,width,CV_32F);
    double S=0;
    for(int i=0;i<x;++i)
        for(int j=0;j<y;++j)
        {
            double mean=0;
            for(auto x:templates) mean+=x.at<uchar>(i,j);
            mean/=templates.size();
            double sum=0;
            for(auto x:templates) sum+=sqr(mean-x.at<uchar>(i,j));
            sum/=templates.size();
            sum=sqrt(sum);
            sum=1.0/sum;
            sigma.at<float>(i,j)=sum;
            S+=sum;
        }
    for(int i=0;i<x;++i)
        for(int j=0;j<y;++j)
            sigma.at<float>(i,j)/=S;
    sigma=sigma.reshape(1,1);
}

Mat EigenFeature::createDataMatrix()
{
    Mat data(static_cast<int>(templates.size()), templates[0].rows * templates[0].cols , CV_32F);
    for(int i = 0; i < templates.size(); i++)
    {
        Mat image = templates[i].reshape(1,1);
        image.copyTo(data.row(i));
    }
    return data;
}

void EigenFeature::calEigen()
{
    Size sz = templates[0].size();
    Mat data = createDataMatrix();
    pca=PCA(data, Mat(), PCA::DATA_AS_ROW, NUM_EIGEN_FACES);

}
double EigenFeature::calDifference(const Mat &img,int is_show)
{
    if(is_show)
        imshow("origin",img);
    Mat img1=img.clone();
    img1=img1.reshape(1,1);
    Mat tmp=pca.project(img1);
    Mat output=pca.backProject(tmp);
    if(is_show==1)
    {
        Mat tmp=output.clone().reshape(1,img.rows);
        double sum=0;
        sum=sqrt(sum/(img.cols*img.rows));
        cout<<"sum="<<sum<<endl;
        normalize(tmp, tmp, 0, 1, NORM_MINMAX, -1, Mat());
        Mat imgshow=img.clone();
        imshow("before_pca",imgshow);
        imshow("after_pca",tmp);
    }
    normalize(output,output,0,256,NORM_MINMAX,-1,Mat());
    normalize(img1,img1,0,256,NORM_MINMAX,-1,Mat());
    if(is_show==1)
    {
        for(int i=0;i<img.rows*img.cols;++i)
            cout<<output.at<float>(0,i)-img1.at<uchar>(0,i)<<endl;
    }

    double ans=0;
    for(int i=0;i<img.rows*img.cols;++i)
        ans+=sigma.at<float>(0,i)*abs(output.at<float>(0,i)-img1.at<uchar>(0,i));

    if(is_show==1) cout<<"value="<<ans<<endl;
    return ans;
}

double EigenFeature::match(const Mat &img,int is_show,int &xx,int &yy)
{

    //cout<<height<<" "<<img.rows<<" "<<width<<" "<<img.cols<<endl;
    if(height>img.rows||width>img.cols) return 1000000000000;

    int ansx=-1,ansy=-1;
    Mat value=Mat::zeros(img.rows,img.cols,CV_32F);
    double mi=1000000000000;
    for(int i=0;i+height<=img.rows;++i)
        for(int j=0;j+width<=img.cols;++j)
        {
            double tmp;
            //if(i==110&&j==103) tmp=calDifference(Mat(img,Rect(j,i,width,height)),1);
            //else
            tmp=calDifference(Mat(img,Rect(j,i,width,height)),0);
            //if(i==228&&j==91) imshow("cnm",Mat(img,Rect(j,i,width,height)));
            value.at<float>(i,j)=(float)tmp;
            if(tmp<mi) mi=tmp,ansx=i,ansy=j;
            //break;
        }

    xx=ansx,yy=ansy;
    if(is_show)
    {
        cout<<"ansx="<<ansx<<endl;
        cout<<"ansy="<<ansy<<endl;
        Point matchLoc(ansy,ansx);
        cout<<matchLoc.x<<endl;
        cout<<matchLoc.y<<endl;
        Mat paint=img.clone();
        circle(paint, Point(matchLoc.x + width/2, matchLoc.y + height/2), 2, Scalar(0, 0, 255),-1);//在图像中画出特征点，2是圆的半径
        rectangle(paint, matchLoc, Point(matchLoc.x + width, matchLoc.y + height), Scalar(0, 255, 0), 1, 1, 0);
        imshow("img", paint);
        waitKey(500);
        cout<<"distance : "<<mi<<endl;
    }
    return mi;
}
#endif // _EigenFeature
