#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv.hpp>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <iostream>
#include "EigenFeature.hpp"
#define mp make_pair
using namespace cv;
using namespace std;
EigenFeature lefteye(3),righteye(3),nose(3),leftmouth(1),rightmouth(1);
vector<Mat> testpicture;
vector<vector<pair<int,int>>> testlabel;
const double THRESHOLD=0.4;
void readTestdata(string dirName)
{
    string pattern_jpg = dirName+"*.jpg";
    string pattern_json = dirName+"*.json";
    vector<cv::String> image_files;
    vector<cv::String> image_jsons;
    cv::glob(pattern_jpg, image_files);
    cv::glob(pattern_json,image_jsons);
    testpicture.resize(image_files.size());
    testlabel.resize(image_files.size());
    for(auto &x:testlabel) x.resize(8);
    for(int i=0;i<image_files.size();++i)
    {
        std::ifstream in(image_jsons[i]);
        std::ostringstream t;
        t << in.rdbuf();
        std::string content = t.str();
        Document document;
        document.Parse<0>(content.c_str());

        for(int j=0;j<8;++j)
        {
            int y=int(document["shapes"][j]["points"][0].GetArray()[0].GetFloat());
            int x=int(document["shapes"][j]["points"][0].GetArray()[1].GetFloat());
            testlabel[i][j]=make_pair(x,y);
        }

        Mat tmp=imread(image_files[i],IMREAD_GRAYSCALE);
        GaussianBlur(tmp,tmp,Size(7,7),0,0);
        tmp.copyTo(testpicture[i]);
    }
}
double IoU(pair<int,int> a,pair<int,int> b,pair<int,int> c,pair<int,int> d)
{
    int X=(b.first-a.first+1)*(b.second-a.second+1);
    int Y=(d.first-c.first+1)*(d.second-c.second+1);
    int I=max(0,min(b.first-c.first,d.first-a.first))*max(0,min(b.second-c.second,d.second-a.second));
    return 1.0*I/(X+Y-I);
}
bool solve(EigenFeature& feature,Mat &origin,pair<int,int> &leftup,pair<int,int> &rightdown,int id)
{
    Mat test=origin.clone();
    int nowx=test.rows;
    int nowy=test.cols;
    int ansx=-1,ansy=-1,xx=-1,yy=-1;
    double mi=1000000000;
    for(int i=0;i<30;++i)
    {
        if(3*feature.height>=nowx&&3*feature.width>=nowy) break;

        int tmpx,tmpy;
        //cout<<"enter test"<<endl;
        double value=feature.match(test,0,tmpx,tmpy);
        //cout<<"ok"<<endl;
        //cout<<"value="<<value<<endl;
        //cout<<"end test"<<endl;
        if(15*feature.height>=nowx&&15*feature.width>=nowy)
            if(value<mi) mi=value,ansx=nowx,ansy=nowy,xx=tmpx,yy=tmpy;
        nowx=0.9*nowx,nowy=0.9*nowy;
        resize(origin,test,Size(nowy,nowx));
    }

   // resize(origin,test,Size(ansy,ansx));
    //feature.match(test,1,xx,yy);

    int height=feature.height*origin.rows/ansx;
    int width=feature.width*origin.cols/ansy;
    ansx=xx*origin.rows/ansx;
    ansy=yy*origin.cols/ansy;


//    cout<<"ansx="<<ansx<<endl;
//    cout<<"ansy="<<ansy<<endl;
//    Point matchLoc(ansy,ansx);
//    cout<<matchLoc.x<<endl;
//    cout<<matchLoc.y<<endl;
//    Mat paint=origin.clone();
//    circle(paint, Point(matchLoc.x + width/2, matchLoc.y + height/2), 2, Scalar(0, 0, 255),-1);//在图像中画出特征点，2是圆的半径
//    rectangle(paint, matchLoc, Point(matchLoc.x + width, matchLoc.y + height), Scalar(0, 255, 0), 1, 1, 0);
//    imshow("img", paint);
//    cout<<"distance : "<<mi<<endl;

    double value=IoU(mp(ansx,ansy),mp(ansx+height-1,ansy+width-1),leftup,rightdown);
        Point matchLoc(ansy,ansx);
        Mat paint=origin.clone();
        circle(paint, Point(matchLoc.x + width/2, matchLoc.y + height/2), 2, Scalar(0, 0, 255),-1);//在图像中画出特征点，2是圆的半径
        rectangle(paint, matchLoc, Point(matchLoc.x + width, matchLoc.y + height), Scalar(0, 255, 0), 1, 1, 0);
        imwrite("rightmoutherror/"+to_string(id)+".jpg", paint);
    if(value>=THRESHOLD) return 1;else return 0;

}
int main(int argc, char **argv)
{
    string dirName="train/";
    /*
    readTemplates(dirName);
    align_templ_size(height,width);
    calEigen();*/
    lefteye.readTemplates(dirName,0,1);
    lefteye.align_templ_size(15,24);
    lefteye.calEigen();

    righteye.readTemplates(dirName,2,3);
    righteye.align_templ_size(15,24);
    righteye.calEigen();

    nose.readTemplates(dirName,4,5);
    nose.align_templ_size(12,24);
    nose.calEigen();

    leftmouth.readTemplates(dirName,6,7);
    leftmouth.align_templ_size(8,8);
    leftmouth.calEigen();

    rightmouth.readTemplates(dirName,8,9);
    rightmouth.align_templ_size(8,8);
    rightmouth.calEigen();

    dirName="test/";
    readTestdata(dirName);
    //imshow("4",testpicture[3]);
    int correct_lefteye=0;
    int correct_righteye=0;
    int correct_eye=0;
    int correct_nose=0;
    int correct_leftmouth=0;
    int correct_rightmouth=0;
    for(int i=0;i<testpicture.size();++i)
    {
        int ans;
        cout<<"lefteye: "<<i<<endl;
         ans=solve(lefteye,testpicture[i],testlabel[i][0],testlabel[i][1],i);
        correct_lefteye+=ans;
        if(ans) cout<<"correct!"<<endl;else cout<<"error!"<<endl;

        cout<<"rightteye: "<<i<<endl;
         ans=solve(righteye,testpicture[i],testlabel[i][2],testlabel[i][3],i);
        correct_righteye+=ans;
        if(ans) cout<<"correct!"<<endl;else cout<<"error!"<<endl;

        cout<<"eye: "<<i<<endl;
         ans=solve(lefteye,testpicture[i],testlabel[i][0],testlabel[i][1],i)|solve(righteye,testpicture[i],testlabel[i][2],testlabel[i][3],i);
        correct_eye+=ans;
        if(ans) cout<<"correct!"<<endl;else cout<<"error!"<<endl;

        cout<<"nose: "<<i<<endl;
         ans=solve(nose,testpicture[i],testlabel[i][4],testlabel[i][5],i);
        correct_nose+=ans;
        if(ans) cout<<"correct!"<<endl;else cout<<"error!"<<endl;

        cout<<"leftmouth: "<<i<<endl;
         ans=solve(leftmouth,testpicture[i],testlabel[i][6],testlabel[i][7],i);
        correct_leftmouth+=ans;
        if(ans) cout<<"correct!"<<endl;else cout<<"error!"<<endl;

        cout<<"rightmouth: "<<i<<endl;
         ans=solve(rightmouth,testpicture[i],testlabel[i][8],testlabel[i][9],i);
        correct_rightmouth+=ans;
        if(ans) cout<<"correct!"<<endl;else cout<<"error!"<<endl;
        //break;
    }
    cout<<"lefteye: "<<correct_lefteye<<endl;
    cout<<"righteye: "<<correct_righteye<<endl;
    cout<<"eye: "<<correct_eye<<endl;
    cout<<"nose: "<<correct_nose<<endl;
    cout<<"leftmouth: "<<correct_leftmouth<<endl;
    cout<<"rightmouth: "<<correct_rightmouth<<endl;

    cout<<clock()<<endl;
    waitKey(0);

    return 0;
}
