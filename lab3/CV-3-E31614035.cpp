#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv.hpp>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <iostream>
#include "EigenFeature.hpp"
using namespace cv;
using namespace std;
const int NUM_EIGEN_FACES=10;
int main(int argc, char **argv)
{
    string dirName="train/";

    Mat test=imread("faceSamples/0598.jpg",IMREAD_GRAYSCALE);
    GaussianBlur(test,test,Size(5,5),0,0);
    //Mat test=imread("test.png",IMREAD_GRAYSCALE);
    //cout<<test.rows<<endl;



    int ansx=-1,ansy=-1;
    double mi=1000000;
    int nowx=25*3,nowy=43*3;
    for(int i=0;i<30;++i)
    {
        if(nowx<=1.0*test.rows/15||nowy<=1.0*test.cols/8) break;
        EigenFeature templ(3);

        templ.readTemplates(dirName);
        templ.align_templ_size(nowx,nowy);

        templ.calEigen();

        double value=templ.match(test,1);

        if(value==1000000000000) break;
        cout<<nowx<<" "<<nowy<<" "<<value<<endl;
        if(nowx>=1.0*test.rows/15&&nowy>=1.0*test.cols/8)
        {
            if(value<mi) mi=value,ansx=nowx,ansy=nowy;
        }
        else
            if(value<mi)
                mi=value,ansx=nowx,ansy=nowy;
        nowx=0.9*nowx,nowy=0.9*nowy;
    }

    //66 115
    EigenFeature templ(3);
    templ.readTemplates(dirName);
    templ.align_templ_size(ansx,ansy);
    templ.calEigen();
    templ.match(test,1);



    cout<<clock()<<endl;
    waitKey(0);

    return 0;
}
