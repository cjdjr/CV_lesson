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
const double THRESHOLD=exp(-5);

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
                ans.at<Vec3b>(i,j)[k]=val;
            }
        }
    return;
}
int sqr(int x)
{
    return x*x;
}
int cal2dis(const Vec3b &a,const Vec3b &b)
{
    return sqr(a[0]-b[0])+sqr(a[1]-b[1])+sqr(a[2]-b[2]);
}
double cal2dis(const Mat &a,const Mat &b)
{
    double ans=0;
    Size imgsize=a.size();
    for(int i=0;i<imgsize.height;++i)
        for(int j=0;j<imgsize.width;++j)
        {
            for(int k=0;k<3;++k)
                ans+=sqr(int(a.at<Vec3b>(i,j)[k])-int(b.at<Vec3b>(i,j)[k]));
        }
    ans/=int(imgsize.height)*int(imgsize.width)*3;
    return ans;
}
Mat padding(const Mat &img,int x,int y,int r)
{
    Mat ans=Mat::zeros(2*r+1,2*r+1, CV_8UC3);
    for(int i=x-r;i<=x+r;++i)
        for(int j=y-r;j<=y+r;++j)
            for(int k=0;k<3;++k)
                ans.at<Vec3b>(i-(x-r),j-(y-r))[k]=img.at<Vec3b>(i,j)[k];
    return ans;
}
void NL_Means(const Mat &img,Mat &ans,int Dp,int Ds,double h)
{
    ans=img.clone();
    Size imgsize=img.size();
    for(int i=0;i<imgsize.height;++i)
        for(int j=0;j<imgsize.width;++j)
        {
            //if(i%10==0&&j%10==0) cout<<i<<" "<<j<<endl;
            Mat centre=padding(img,i,j,Dp);
            //if(i==30&&j==30) cout<<centre<<endl;
            double b=0,g=0,r=0;
            double s=0;
            double mx=0;
            for(int p=max(0,i-Ds);p<min(i+Ds+1,imgsize.height);p+=1)
                for(int q=max(0,j-Ds);q<min(j+Ds+1,imgsize.width);q+=1)
                {
                    if(p==i&&q==j) continue;
                    Mat tmp=padding(img,p,q,Dp);
                    //if(i==30&&j==30&&p+1==i&&q+1==j) cout<<"debug "<<-cal2dis(centre,tmp)<<endl;
                    //cout<<tmp<<endl;
                    double num=-cal2dis(centre,tmp)/(h*h);
                    //if(i==0&&j==5) cout<<"debug : "<<num<<endl;
                    double dis=exp(num);
                    //if(i==10&&j==10) cout<<"debug : "<<num<<endl;
                  //  if(num<=-5) dis=0;//      // ±£³Ö±ßÔµ

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



void fastNL_Means(const Mat &origin_img,Mat &ans,int Dp,int Ds,double h)
{
    Mat img=Mat::zeros(Size(origin_img.rows+2*Dp, origin_img.cols+2*Dp), CV_8UC3);
    copyMakeBorder(origin_img,img,Dp,Dp,Dp,Dp,BORDER_REFLECT_101);
    Size imgsize=origin_img.size();
    vector<vector<double>> s,mx,b,g,r;
    vector<vector<long long>> presum;
    s.resize(imgsize.height);
    b.resize(imgsize.height);
    g.resize(imgsize.height);
    r.resize(imgsize.height);
    mx.resize(imgsize.height);
    presum.resize(img.rows+1);
    for(int i=0;i<=img.rows;++i) presum[i].resize(img.cols+1);
    for(int i=0;i<imgsize.height;++i)
    {
        b[i].resize(imgsize.width);
        g[i].resize(imgsize.width);
        r[i].resize(imgsize.width);
        s[i].resize(imgsize.width);
        mx[i].resize(imgsize.width);
        for(int j=0;j<imgsize.width;++j) s[i][j]=mx[i][j]=b[i][j]=g[i][j]=r[i][j]=0.0;
    }
    //cout<<clock()<<endl;
    for(int p=-Ds;p<=Ds;p+=1)
        for(int q=-Ds;q<=Ds;q+=1)
        {
            //cout<<p<<" "<<q<<endl;
            if(p==0&&q==0) continue;
            for(int i=0;i<=img.rows;++i)
                for(int j=0;j<=img.cols;++j)
                    presum[i][j]=0;
            for(int i=0;i<img.rows;++i)
                for(int j=0;j<img.cols;++j)
                    if(i+p>=0&&i+p<img.rows&&j+q>=0&&j+q<img.cols)
                    {
                        presum[i+1][j+1]=presum[i][j+1]+presum[i+1][j]-presum[i][j]+cal2dis(img.at<Vec3b>(i,j),img.at<Vec3b>(i+p,j+q));
                    }
            for(int i=0;i<imgsize.height;++i)
                for(int j=0;j<imgsize.width;++j)
                    if(i+p>=0&&i+p<imgsize.height&&j+q>=0&&j+q<imgsize.width)
                    {
                        double dis=presum[i+Dp+Dp+1][j+Dp+Dp+1]-presum[i-Dp-1+Dp+1][j+Dp+Dp+1]-presum[i+Dp+Dp+1][j-Dp-1+Dp+1]+presum[i-Dp-1+Dp+1][j-Dp-1+Dp+1];
                        dis=dis/(3*sqr(2*Dp+1));
                        //dis=exp(-dis/(h*h));
                        //dis=-dis/(h*h);
                        if(dis<=THRESHOLD) dis=0;
                        s[i][j]+=dis;
                        mx[i][j]=max(mx[i][j],dis);
                        b[i][j]+=dis*origin_img.at<Vec3b>(i+p,j+q)[0];
                        g[i][j]+=dis*origin_img.at<Vec3b>(i+p,j+q)[1];
                        r[i][j]+=dis*origin_img.at<Vec3b>(i+p,j+q)[2];
                    }
        }
    ans=origin_img.clone();
    for(int i=0;i<imgsize.height;++i)
        for(int j=0;j<imgsize.width;++j)
        {
            b[i][j]+=mx[i][j]*origin_img.at<Vec3b>(i,j)[0];
            g[i][j]+=mx[i][j]*origin_img.at<Vec3b>(i,j)[1];
            r[i][j]+=mx[i][j]*origin_img.at<Vec3b>(i,j)[2];
            s[i][j]+=mx[i][j];
            if(s[i][j]>0)
            {
                b[i][j]/=s[i][j],g[i][j]/=s[i][j],r[i][j]/=s[i][j];
                ans.at<Vec3b>(i,j)[0]=int(b[i][j]);
                ans.at<Vec3b>(i,j)[1]=int(g[i][j]);
                ans.at<Vec3b>(i,j)[2]=int(r[i][j]);
            }
        }
}
int main(int argc, char *argv[])
{

    Mat img = imread("lena.jpg");
    if(img.empty())
       return -1;
    imshow("img",img);
    Mat noise_img;
    Add_Gaussian_noise(img,noise_img,5);
    imshow("noise_img",noise_img);
    Mat denoise_img;

    //NL_Means(noise_img,denoise_img,1,10,10);
    fastNL_Means(noise_img,denoise_img,1,10,10);
    //cv::fastNlMeansDenoisingColored(noise_img,denoise_img,6,3,3,10);
    cout<<clock()<<endl;
    imshow("denoise_img",denoise_img);

    waitKey(0);


    return 0;
}

