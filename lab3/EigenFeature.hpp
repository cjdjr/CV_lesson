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
    int height=100,width=100;
    PCA pca;

    Mat sigma;
    vector<Mat> templates;
    double calDifference(const Mat &img,int);
    Mat createDataMatrix();

public:
    EigenFeature(){}
    EigenFeature(int num):NUM_EIGEN_FACES(num){} ;
    void readTemplates(string dirName);
    void align_templ_size(int x,int y);
    void calEigen();
    double match(const Mat& img,int is_show=0);

};

void EigenFeature::readTemplates(string dirName)
{
    //cout<<"reading templates..."<<endl;
    string pattern_jpg = dirName+"*.jpg";
    string pattern_json = dirName+"*.json";
    vector<cv::String> image_files;
    vector<cv::String> image_jsons;
    cv::glob(pattern_jpg, image_files);
    cv::glob(pattern_json,image_jsons);
    templates.resize(image_files.size());
    for(int i=0;i<image_files.size();++i)
    {
        //cout << image_files[i] << endl;
        std::ifstream in(image_jsons[i]);
        std::ostringstream t;
        t << in.rdbuf();
        std::string content = t.str();
        //content="{\"hello\" : \"ligoudan\"}";

        Document document;
        document.Parse<0>(content.c_str());

        int x1,y1,x2,y2;
        //cout<<document["shapes"][0]["points"][0].GetArray()<<endl;

        y1=int(document["shapes"][0]["points"][0].GetArray()[0].GetFloat());
        x1=int(document["shapes"][0]["points"][0].GetArray()[1].GetFloat());
        y2=int(document["shapes"][1]["points"][0].GetArray()[0].GetFloat());
        x2=int(document["shapes"][1]["points"][0].GetArray()[1].GetFloat());

        //cout<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<endl;
        //break;
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
            //cout<<i<<" "<<j<<" "<<sum<<endl;
            S+=sum;
        }
    for(int i=0;i<x;++i)
        for(int j=0;j<y;++j)
            sigma.at<float>(i,j)/=S;
    sigma=sigma.reshape(1,1);
}

Mat EigenFeature::createDataMatrix()
{
    //cout << "Creating data matrix from templates ..."<<endl;
    Mat data(static_cast<int>(templates.size()), templates[0].rows * templates[0].cols , CV_32F);
    for(int i = 0; i < templates.size(); i++)
    {
        Mat image = templates[i].reshape(1,1);
        image.copyTo(data.row(i));
    }
    //cout << " DONE" << endl;
    return data;
}

void EigenFeature::calEigen()
{
    Size sz = templates[0].size();
    Mat data = createDataMatrix();
    //cout << "Calculating PCA ...";
    pca=PCA(data, Mat(), PCA::DATA_AS_ROW, NUM_EIGEN_FACES);
    //MeansTemplates=pca.mean;
    //cout << " DONE"<< endl;

    /*templates_pca.resize(templates.size());
    for(int i=0;i<templates.size();++i)
    {
        templates_pca[i]=pca.project(templates[i].reshape(1,1));
        //normlize(templates_pca[i],templates_pca[i],0,1,NORM_MINMAX)
    }*/

    /*Mat tmp=pca.mean.reshape(1,sz.height);
    normalize(tmp,tmp,0,1,NORM_MINMAX,CV_32F);
    imshow("wmr",tmp);*/

    /*Mat tmp=pca.project(templates[0].reshape(1,1));
    cout<<tmp<<endl;
    Mat output=pca.backProject(tmp).reshape(1,templates[0].rows);
    normalize(output, output, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("haha",output);*/

    //cout<<pca.backProject(tmp)<<endl;
    //tmp=pca.project(templates[1].reshape(1,1));
    //cout<<tmp<<endl;

    /*Mat averageFace = pca.mean.reshape(1,sz.height);
    Mat dst;
    normalize(averageFace, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    Mat output(averageFace.size(),CV_8UC1);
    for(int i=0;i<output.rows;++i)
        for(int j=0;j<output.cols;++j)
            output.at<uchar>(i,j)=int(averageFace.at<float>(i,j));*/
    //resize(averageFace, output, Size(), 2, 2);
    //imshow("Result", dst);
    //imshow("Result1", output);
}
double EigenFeature::calDifference(const Mat &img,int is_show)
{
    //assert(height==img.rows&&width==img.cols);
    //cout<<"enter..."<<endl;
    //cout<<img<<endl;
    if(is_show)
        imshow("origin",img);
    Mat img1=img.clone();
    //bilateralFilter (img1, img1, 3, 500, 1000);

    img1=img1.reshape(1,1);
    Mat tmp=pca.project(img1);
    Mat output=pca.backProject(tmp);
    //cout<<tmp<<endl;
    //cout<<output<<endl;
    if(is_show==1)
    {
        Mat tmp=output.clone().reshape(1,img.rows);
        //cout<<img1<<endl;
        //cout<<output<<endl;
        double sum=0;
        /*for(int i=0;i<img.rows;++i)
        {
            for(int j=0;j<img.cols;++j)
                cout<<tmp.at<float>(i,j)-img.at<uchar>(i,j)<<" ",sum+=sqr(tmp.at<float>(i,j)-img.at<uchar>(i,j));
            cout<<endl;
        }*/
        sum=sqrt(sum/(img.cols*img.rows));
        cout<<"sum="<<sum<<endl;
        normalize(tmp, tmp, 0, 1, NORM_MINMAX, -1, Mat());
        Mat imgshow=img.clone();
        //normalize(imgshow,imgshow,0,1,NORM_MINMAX,-1,Mat());
        imshow("before_pca",imgshow);
        imshow("after_pca",tmp);
    }
    //double s1=0,s2=0;
    //double sigma1=0,sigma2=0;
    normalize(output,output,0,256,NORM_MINMAX,-1,Mat());

    normalize(img1,img1,0,256,NORM_MINMAX,-1,Mat());
    if(is_show==1)
    {
        for(int i=0;i<img.rows*img.cols;++i)
            cout<<output.at<float>(0,i)-img1.at<uchar>(0,i)<<endl;
    }
    //for(int i=0;i<img.rows*img.cols;++i) s1+=output.at<float>(0,i),s2+=img1.at<uchar>(0,i);
    //s1/=img.cols*img.rows;
    //s2/=img.cols*img.rows;
    //for(int i=0;i<img.rows*img.cols;++i) sigma1+=sqr(output.at<float>(0,i)-s1),sigma2+=sqr(1.0*img1.at<uchar>(0,i)-s2);
    //sigma1/=img.cols*img.rows;
    //sigma2/=img.cols*img.rows;
    //sigma1=sqrt(sigma1);
    //sigma2=sqrt(sigma2);
    //cout<<output<<endl;
    double ans=0;
    for(int i=0;i<img.rows*img.cols;++i)
        ans+=sigma.at<float>(0,i)*abs(output.at<float>(0,i)-img1.at<uchar>(0,i));
        //ans+=abs(output.at<float>(0,i)-img1.at<uchar>(0,i));
    ans=(ans);
    //cout<<ans<<endl;
    if(is_show==1) cout<<"value="<<ans<<endl;
    return ans;
    //cout<<"wmr"<<endl;
    /*double mi=1000000000;
    for(auto &x:templates_pca)
    {
        double ans=0;
       /* for(int i=0;i<tmp.cols;++i)
            ans+=sqr(tmp.at<float>(0,i)-x.at<float>(0,i));
        ans/=tmp.cols;
        ans=sqrt(ans);
        ans=-cv::compareHist(x, tmp, CV_COMP_CORREL);

        mi=min(mi,ans);
        //mi+=ans;
    }*/
   // cout<<sqrt(ans)<<endl;
   // mi/=templates_pca.size();
}

double EigenFeature::match(const Mat &img,int is_show)
{
    /*Mat tmp=pca.project(templates[0].reshape(1,1));
    cout<<tmp<<endl;
    Mat output=pca.backProject(tmp).reshape(1,templates[0].rows);
    normalize(output, output, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("haha",output);*/
    cout<<height<<" "<<img.rows<<" "<<width<<" "<<img.cols<<endl;
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
    //cout<<value<<endl;
    //normalize(value, value, 0, 1, NORM_MINMAX, -1, Mat());
    //imshow("value",value);
    //cout<<value<<endl;

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
