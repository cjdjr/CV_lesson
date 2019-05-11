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

void Add_Gaussian_noise(const Mat &img,Mat &ans,double sigma)
{
    RNG rng;
    ans=img.clone();
    Size imgsize=ans.size();
    for(int i=0;i<imgsize.height;++i)
        for(int j=0;j<imgsize.width;++j)
        {
            double noise=rng.gaussian(sigma);
            for(int k=0;k<3;++k)
            {
                int val=int(int(ans.at<Vec3b>(i,j)[k])+noise);
                if(val<0) val=0;
                if(val>255) val=255;
                //cout<<val<<endl;
                ans.at<Vec3b>(i,j)[k]=val;
            }
        }
    return;
}
double sqr(int x)
{
    return x*x;
}
double cal2dis(const Mat &a,const Mat &b)
{
    double ans=0;
    Size imgsize=a.size();
    return int(imgsize.height)*int(imgsize.width)*3;
    for(int i=0;i<imgsize.height;++i)
        for(int j=0;j<imgsize.width;++j)
            for(int k=0;k<3;++k)
                ans+=sqr(a.at<Vec3b>(i,j)[k]-b.at<Vec3b>(i,j)[k]);
    ans/=int(imgsize.height)*int(imgsize.width)*3;
    return ans;
}
Mat padding(const Mat &img,int x,int y,int r)
{
    Mat ans=Mat::zeros(2*r+1,2*r+1, CV_LOAD_IMAGE_COLOR);
    for(int i=x-r;i<=x+r;++i)
        for(int j=y-r;j<=y+r;++j)
            ans.at<Vec3b>(i-(x-r),j-(y-r))=img.at<Vec3b>(i,j);
    return ans;
}
void NL_Means(const Mat &img,Mat &ans,int Dp,int Ds,double h)
{
    ans=img.clone();
    Size imgsize=img.size();
    for(int i=0;i<imgsize.height;++i)
        for(int j=0;j<imgsize.width;++j)
        {
            if(i%10==0&&j%10==0) cout<<i<<" "<<j<<endl;
            Mat centre=padding(img,i,j,Dp);
            //if(i==30&&j==30) cout<<centre<<endl;
            double b=0,g=0,r=0;
            double s=0;
            double mx=0;
            for(int p=max(0,i-Ds);p<min(i+Ds,imgsize.height);p+=1)
                for(int q=max(0,j-Ds);q<min(j+Ds,imgsize.width);q+=1)
                {
                    if(p==i&&q==j) continue;
                    Mat tmp=padding(img,p,q,Dp);
                    //cout<<tmp<<endl;
                    double num=-cal2dis(centre,tmp)/(h*h);
                    //if(i==0&&j==5) cout<<"debug : "<<num<<endl;
                    double dis=exp(num);
                    //if(i==10&&j==10) cout<<"debug : "<<num<<endl;
                    if(num<=-5) dis=0;//      // 保持边缘
                    //cout<<"("<<i<<","<<j<<")="<<dis<<endl;
                    s+=dis;
                    mx=max(mx,dis);
                    b+=dis*img.at<Vec3b>(p,q)[0];
                    g+=dis*img.at<Vec3b>(p,q)[1];
                    r+=dis*img.at<Vec3b>(p,q)[2];
                }
            b+=mx*img.at<Vec3b>(i,j)[0];
            g+=mx*img.at<Vec3b>(i,j)[1];
            r+=mx*img.at<Vec3b>(i,j)[2];
            s+=mx;
            if(s>0)
            {
                //cout<<"("<<i<<","<<j<<")="<<s<<endl;
                b/=s,g/=s,r/=s;
                ans.at<Vec3b>(i,j)[0]=int(b);
                ans.at<Vec3b>(i,j)[1]=int(g);
                ans.at<Vec3b>(i,j)[2]=int(r);
                //cout<<"("<<i<<","<<j<<")="<<b<<","<<g<<","<<r<<endl;
            }
            //if(i==30&&j==30) cout<<"debug"<<s<<endl;
        }
    return;
}

vector<vector<Mat>> frame;
void fastNL_Means(const Mat &origin_img,Mat &ans,int Dp,int Ds,double h)
{
    Mat img=Mat::zeros(Size(origin_img.rows+2*Dp, origin_img.cols+2*Dp), CV_8UC3);
    copyMakeBorder(origin_img,img,Dp,Dp,Dp,Dp,BORDER_REFLECT_101);
    //ans=origin_img.clone();
    Size imgsize=origin_img.size();
    frame.resize(imgsize.height);
    for(int i=0;i<imgsize.height;++i) frame[i].resize(imgsize.width);
    for(int i=0;i<imgsize.height;++i)
        for(int j=0;j<imgsize.width;++j)
        {
            frame[i][j]=padding(img,i+Dp,j+Dp,Dp);
        }
    ans=origin_img.clone();
    for(int i=0;i<imgsize.height;++i)
        for(int j=0;j<imgsize.width;++j)
        {
            //if(i%10==0&&j%10==0) cout<<i<<" "<<j<<endl;
            //if(i==30&&j==30) cout<<frame[i][j]<<endl;
            double b=0,g=0,r=0;
            double s=0;
            double mx=0;
            for(int p=max(0,i-Ds);p<min(i+Ds,imgsize.height);p+=1)
                for(int q=max(0,j-Ds);q<min(j+Ds,imgsize.width);q+=1)
                {
                    if(p==i&&q==j) continue;
                    double num=-cal2dis(frame[i][j],frame[p][q])/(h*h);
                    //if(i==0&&j==5) cout<<"debug : "<<num<<endl;
                    double dis=exp(num);
                    //if(i==10&&j==10) cout<<"debug : "<<num<<endl;
                    if(num<=-5) dis=0;      // 保持边缘
                    //cout<<"("<<i<<","<<j<<")="<<dis<<endl;
                    s+=dis;
                    mx=max(mx,dis);
                    b+=dis*img.at<Vec3b>(p,q)[0];
                    g+=dis*img.at<Vec3b>(p,q)[1];
                    r+=dis*img.at<Vec3b>(p,q)[2];
                }
            b+=mx*img.at<Vec3b>(i,j)[0];
            g+=mx*img.at<Vec3b>(i,j)[1];
            r+=mx*img.at<Vec3b>(i,j)[2];
            s+=mx;
            if(s>0)
            {
                //cout<<"("<<i<<","<<j<<")="<<s<<endl;
                b/=s,g/=s,r/=s;
                ans.at<Vec3b>(i,j)[0]=int(b);
                ans.at<Vec3b>(i,j)[1]=int(g);
                ans.at<Vec3b>(i,j)[2]=int(r);
                //cout<<"("<<i<<","<<j<<")="<<b<<","<<g<<","<<r<<endl;
            }
            //if(i==30&&j==30) cout<<"debug"<<s<<endl;
        }
    return;
}
int main(int argc, char *argv[])
{
    //freopen("ce.out","w",stdout);
    Mat img = imread("1.jpg", CV_LOAD_IMAGE_COLOR);
    if(img.empty())
       return -1;
    imshow("img",img);
    Mat noise_img;
    Add_Gaussian_noise(img,noise_img,5);
    imshow("noise_img",noise_img);
    //imwrite("4_noise.jpg",noise_img);
    Mat denoise_img;
    //NL_Means(noise_img,denoise_img,1,10,10);
    fastNL_Means(noise_img,denoise_img,1,10,10);
    //cv::fastNlMeansDenoisingColored(noise_img,denoise_img,6,3,3,15);
    imshow("denoise_img",denoise_img);
    cout<<clock()<<endl;
    //imwrite("denoise5.jpg",denoise_img);
    //cv::fastNlMeansDenoisingColored(noise_img,denoise_img,6,3,3,21);
    //imshow("denoise_img_std",denoise_img);
    //imwrite("4_denoise.jpg",denoise_img);
    waitKey(0);


    return 0;
}

