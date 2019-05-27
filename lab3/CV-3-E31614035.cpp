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
const int NUM_EIGEN_FACES=10;
void readImages(string dirName,vector<Mat> &images)
{
    cout<<"reading images..."<<endl;
    string pattern_jpg = dirName+"*.png";
    vector<cv::String> image_files;
    cv::glob(pattern_jpg, image_files);
    images.resize(image_files.size());
    for(int i=0;i<image_files.size();++i)
    {
        cout << image_files[i] << endl;
   		Mat tmp=imread(image_files[i],IMREAD_GRAYSCALE);
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

    Mat img=imread("./faceSamples/0189.jpg");
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
void align(vector<Mat> &templates)
{
    int row=0,col=0;
    for(auto x:templates)
    {
        row+=x.rows;
        col+=x.cols;
    }
    row=1.0*row/templates.size();
    col=1.0*col/templates.size();
    for(auto &x:templates)
    {
        //cout<<x.rows<<" "<<x.cols<<"---> ";
        resize(x,x,Size(col,row));
        //cout<<row<<" "<<col<<endl;
    }
    int a[100][100];
    memset(a,0,sizeof(a));
    for(auto x:templates)
    {
        for(int i=0;i<row;++i)
            for(int j=0;j<col;++j)
                a[i][j]+=x.at<uchar>(i,j);
    }
}
static  Mat createDataMatrix(const vector<Mat> &images)
{
    cout << "Creating data matrix from images ..."<<endl;
    // Allocate space for all images in one data matrix.
    // The size of the data matrix is
    //
    // ( w  * h  * 1, numImages )
    //
    // where,
    //
    // w = width of an image in the dataset.
    // h = height of an image in the dataset.
    // 3 is for the 3 color channels.
    // numImages = number of images in the dataset.
    Mat data(static_cast<int>(images.size()), images[0].rows * images[0].cols , CV_32F);
    for(int i = 0; i < images.size(); i++)
    {
        //cout<<i<<" "<<images[i].rows<<" "<<images[i].cols<<endl;
        Mat image = images[i].reshape(1,1);
    // Copy the long vector into one row of the destm
        //cout<<i<<" "<<image.rows<<" "<<image.cols<<endl;
        image.copyTo(data.row(i));
    }
    cout << " DONE" << endl;
    return data;
}
void EigenFeature(vector<Mat> &templates)
{
    //templates.resize(2);
    //templates[1]=templates[0].clone();
    align(templates);
    //for(int i=0;i<templates.size();++i) imshow("debug"+to_string(i),templates[i]);
    //cout<<templates[0]<<endl;
    Size sz = templates[0].size();
    cout<<"size :"<<sz.height<<" "<<sz.width<<endl;
    Mat data = createDataMatrix(templates);
    cout << "Calculating PCA ...";
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, NUM_EIGEN_FACES);
    //cout<<pca.eigenvectors.size()<<endl;
    //cout<<pca.mean.size()<<endl;
    cout << " DONE"<< endl;

    /*Mat tmp=pca.project(templates[0].reshape(1,1));
    cout<<tmp<<endl;
    Mat output=pca.backProject(tmp).reshape(1,templates[0].rows);
    normalize(output, output, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("haha",output);*/

    Mat test=imread("./test.png",IMREAD_GRAYSCALE);
    resize(test,test,sz);
    Mat tmp=pca.project(test.reshape(1,1));
    cout<<tmp<<endl;
    Mat output=pca.backProject(tmp).reshape(1,test.rows);
    normalize(output, output, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("haha",output);

    //cout<<pca.backProject(tmp)<<endl;
    //tmp=pca.project(templates[1].reshape(1,1));
    //cout<<tmp<<endl;

/*    Mat averageFace = pca.mean.reshape(1,sz.height);
    Mat dst;
    normalize(averageFace, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    Mat output(averageFace.size(),CV_8UC1);
    for(int i=0;i<output.rows;++i)
        for(int j=0;j<output.cols;++j)
            output.at<uchar>(i,j)=int(averageFace.at<float>(i,j));
    //resize(averageFace, output, Size(), 2, 2);
    //imshow("Result", dst);
    //imshow("Result1", output);


    Mat eigenVectors = pca.eigenvectors;

    for(int i = 0; i < NUM_EIGEN_FACES; i++)
    {
        Mat eigenFace = eigenVectors.row(i).reshape(1,sz.height);
        //eigenFaces.push_back(eigenFace);
        //namedWindow(to_string(i), CV_WINDOW_AUTOSIZE);
        /*cout<<eigenFace<<endl;
        Mat output(eigenFace.size(),CV_8UC1);
        for(int i=0;i<output.rows;++i)
            for(int j=0;j<output.cols;++j)
                output.at<uchar>(i,j)=int(eigenFace.at<float>(i,j));
        imshow(to_string(i),eigenFace);*/
    //}
}
int main(int argc, char **argv)
{
    string dirName = "faceSamples/";
    // Read images in the directory
   /* vector<Mat> images;
    readImages(dirName, images);*/

    dirName="templates/left_eyes/";
    vector<Mat> templates;
    readImages(dirName,templates); //0673

    //for(int i=0;i<1;++i) match_templates(templates,i);
    //imshow("ce",templates[0]);
    EigenFeature(templates);
    cout<<clock()<<endl;
    waitKey(0);

    return 0;
}
