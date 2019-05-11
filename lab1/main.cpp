#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<iostream>
#include<ctime>
#include<cmath>
#include<sstream>
#include<iomanip>
#define CV_COMP_CHISQR_ALT CV_COMP_CORREL
using namespace cv;
using namespace std;
int fps;        //֡��
int MARGIN;     //ÿ��MARGIN֡ȡһ֡�����з�����Ҳ�������ڷ�����֡������֡��/MARGIN
int MIN_SCENE_MARGIN;   //������ͷ����С�������λ��MARGIN
int T;      //�жϵ�i֡�Ƿ��Ǳ�Ե֡��ʱ����[i-T,i+T]����������Сֵ��T�ĵ�λ��MARGIN
double hsv_THRESHOLD;        //��ֵ����һ֡�����ָ���THRESHOLD����ô������Ϊ���Ǳ�Ե֡
double gray_THRESHOLD;

vector<vector<double>> hsv_d;
vector<vector<double>> gray_d;

vector<Mat> hsv_hist_per_frame;
vector<Mat> gray_hist_per_frame;
vector<Mat> origin_frame;
void cal_hsv_hist(const Mat &origin_frame,Mat &hsv_hist)
{
    Mat hsv;
    cvtColor(origin_frame, hsv, CV_BGR2HSV);        //ת��hsv
    int Channels[]={0,1};
    float hueRanges[] = { 0,180 };
    float satRanges[] = { 0,256 };
    int HistSize[] = { 30,32 };
    const float* HistRanges[] = { hueRanges,satRanges };
    calcHist(&hsv,1,Channels,noArray(),hsv_hist,2,HistSize,HistRanges,true,false);
    normalize(hsv_hist, hsv_hist, 1, 0, NORM_MINMAX);
}
void cal_rgb_hist(const Mat &origin_frame,Mat &rgb_hist)
{
    Mat rgb=origin_frame.clone();
    int Channels[]={0,1,2};
    float hueRanges[] = { 0,256 };
    float satRanges[] = { 0,256 };
    float C[]={0,256};
    int HistSize[] = { 32,32,32 };
    const float* HistRanges[] = { hueRanges,satRanges ,C};
    calcHist(&rgb,1,Channels,noArray(),rgb_hist,3,HistSize,HistRanges,true,false);
    normalize(rgb_hist, rgb_hist, 1, 0, NORM_MINMAX);
}

void cal_gray_hist(const Mat &origin_frame,Mat &gray_hist)
{
    Mat gray;
    cvtColor(origin_frame, gray, CV_BGR2GRAY);

    //����Ҷ�ֱ��ͼ
    int channels[] = { 0 };//�������ֱ��ͼ��ͨ����ͨ����Ŀ=ֱ��ͼά��
    int histDim = 1;//ֱ��ͼ��ά�ȣ�����ֻ���㵥ͨ���Ҷ�ͼ��ֱ��ͼ�������1ά
    int histSize[] = { 32 };//ֱ��ͼ�ĳ��ȣ�256���Ҷȼ��������Է�Ϊ256������
    float ranges[] = { 0,256 };//����ֱ��ͼ�ĻҶȼ���Χ��[0,255].256�����ޣ����������ޣ�
    const float* histRanges[] = { ranges };
    calcHist(
        &gray,//ͼ�����ָ��
        1,//ֻ��һ��ͼ����Ҫ����ֱ��ͼ
        channels, //�Ҷ�ͼ��ֻ��һ��ͨ����������ֱ��ͼ����ͨ��0
        noArray(),//���ڼ���ֱ��ͼ��ͼ������noArray()��ʾȫͼ��Ҫ�������
        gray_hist, //���������
        histDim, //ֱ��ͼ��ά�ȣ�1����ֻ��һ��ͨ����ֻ�ܼ���1άֱ��ͼ��
        histSize, //ֱ��ͼ�ĳ��ȣ�ֱ��ͼ��bin��Ŀ�����ѻҶ�ֵ��ΪhistSize�����䣬ÿ�����䶼ͳ������ûҶ������������Ŀ��
        histRanges,//ֱ��ͼ�ĻҶ�ֵ��Χ��uniform����£�ֻ�涨�Ҷ�ֵ�����޺�����
        true//��uniform��ʽ����ֱ��ͼ������histRanges�涨�ĻҶ�����ȷ�ΪhistSize��С���䣬ÿ������ֱ�ͳ��������Ŀ
    );
    normalize(gray_hist, gray_hist, 1, 0, NORM_MINMAX);
}
void work(const Mat &origin_frame,int id)
{
    Mat hsv_hist;
    cal_hsv_hist(origin_frame,hsv_hist);
    Mat gray_hist;
    cal_rgb_hist(origin_frame,gray_hist);
    if(id%fps==0)
    {
        cout<<"processing "<<id/30<<endl;
    }
    hsv_hist_per_frame.push_back(hsv_hist);
    gray_hist_per_frame.push_back(gray_hist);
}

//��������Ӧ��ֵ�Լ���������d
void init()
{
    int N=origin_frame.size();
    hsv_d.resize(N);
    for(int i=0;i<N;++i) hsv_d[i].resize(2*T+1);
    for(int i=0;i<N;++i)
        for(int j=1;j<=2*T;++j)
            if(i-j>=0)
                hsv_d[i][j]=-compareHist(hsv_hist_per_frame[i],hsv_hist_per_frame[i-j],CV_COMP_CHISQR_ALT);

    gray_d.resize(N);
    for(int i=0;i<N;++i) gray_d[i].resize(2*T+1);
    for(int i=0;i<N;++i)
        for(int j=1;j<=2*T;++j)
            if(i-j>=0)
                gray_d[i][j]=-compareHist(gray_hist_per_frame[i],gray_hist_per_frame[i-j],CV_COMP_CHISQR_ALT);

    srand(19260817);
    hsv_THRESHOLD=0;
    for(int turn=1;turn<=500;++turn)
    {
        double s=100000;
        for(int cas=1;cas<=500;++cas)
        {
            int u=rand()%N;
            int v=rand()%N;
            if(abs(u-v)<=MIN_SCENE_MARGIN*60) continue;
            s=min(s,-compareHist(hsv_hist_per_frame[u],hsv_hist_per_frame[v],CV_COMP_CHISQR_ALT));
        }

        hsv_THRESHOLD+=s;
    }
    hsv_THRESHOLD/=500;

    gray_THRESHOLD=0;
    for(int turn=1;turn<=500;++turn)
    {
        double s=100000;
        for(int cas=1;cas<=500;++cas)
        {
            int u=rand()%N;
            int v=rand()%N;
            if(abs(u-v)<=MIN_SCENE_MARGIN*60) continue;
            s=min(s,-compareHist(gray_hist_per_frame[u],gray_hist_per_frame[v],CV_COMP_CHISQR_ALT));
        }
        gray_THRESHOLD+=s;
    }
    gray_THRESHOLD/=500;
}

double cal_hsv_dis(int id)
{
    double dis=100000;
    if(id==0) return dis;
    for(int j=max(0,id-T);j<id;++j) dis=min(dis,hsv_d[id][id-j]);

    for(int l=max(0,id-T);l<id;++l)
        for(int r=id+1;r<=min(int(hsv_hist_per_frame.size())-1,id+T);++r)
            dis=min(dis,hsv_d[r][r-l]);

    return dis;
}
double cal_gray_dis(int id)
{
    double dis=100000;
    if(id==0) return dis;
    for(int j=max(0,id-T);j<id;++j) dis=min(dis,gray_d[id][id-j]);

    for(int l=max(0,id-T);l<id;++l)
        for(int r=id+1;r<=min(int(gray_hist_per_frame.size())-1,id+T);++r)
            dis=min(dis,gray_d[r][r-l]);

    return dis;
}
Mat hsv[2];
Mat hist[2];
int main()
{
    /*��ȡ��Ƶ*/
    cout<<"������Ҫ��ȡ��Ƶ���ļ�����"<<endl;
    string name;
    cin>>name;
    VideoCapture capture(name);
    fps = int(round(capture.get(5)));

    MARGIN=3;
    MIN_SCENE_MARGIN=int(round(fps*1.0/MARGIN));    //����MIN_SCENE_MARGIN��ֵΪ1���֡��
    T=MIN_SCENE_MARGIN/2;     //���������СT
    cout<<fps<<endl;

    Mat frame;
    int num=0;
    for (;;)
    {
        capture >> frame;
        if (frame.empty()) break;
        ++num;
        if(num%MARGIN!=0) continue;
        Mat tmp=frame.clone();
        origin_frame.push_back(tmp);
        work(frame,num);
    }


    init();

    cout<<"hsv_THRESHOLD="<<hsv_THRESHOLD<<endl;
    cout<<"gray_THRESHOLD="<<gray_THRESHOLD<<endl;

    vector<int> margin_frame;
    margin_frame.clear();

    for(int i=0;i<int(hsv_hist_per_frame.size());++i)
    {
        double hsv_dis,gray_dis;
        hsv_dis=cal_hsv_dis(i);
        gray_dis=cal_gray_dis(i);
        //������ͷ�ļ��Ҫ����һ�����ƣ�������
        if(!margin_frame.empty()&&i-*margin_frame.rbegin()<MIN_SCENE_MARGIN) hsv_dis=gray_dis=-10000;

        if(hsv_dis>=hsv_THRESHOLD&&gray_dis>=gray_THRESHOLD)
        {
            margin_frame.push_back(i);
            imwrite("../pictures/"+to_string(margin_frame.size())+"("+to_string(i)+")"+".jpg",origin_frame[i]);
        }
    }
    for(int i=0;i<margin_frame.size();++i)
    {
        stringstream ss;
        string name;
        ss<<setfill('0')<<setw(3)<<i+1;
        ss<<"-1.jpg";
        ss>>name;
        imwrite("../answer/"+name,origin_frame[margin_frame[i]]);
        ss.clear();
        ss.str("");
        ss<<setfill('0')<<setw(3)<<i+1;
        ss<<"-2.jpg";
        ss>>name;
        imwrite("../answer/"+name,(i<int(margin_frame.size())-1)?origin_frame[margin_frame[i+1]-1]:*origin_frame.rbegin());
    }
    waitKey(0);
    return 0;
}




