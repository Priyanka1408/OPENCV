
/// TO PRINT A TEXT
/*
#include <opencv2/core/core.hpp>             // Used only for image processing
#include <opencv2/imgproc/imgproc.hpp>       // Used only for image processing
#include <opencv2/highgui/highgui.hpp>      // Used only for image processing
#include <iostream>

using namespace cv;


int main()
{
   Mat image(512,512,CV_8UC3,Scalar(0,0,0));
   putText(image,
           "hello",
           Point(100,image.rows/10),
           FONT_HERSHEY_DUPLEX,
           1.0,
           CV_RGB(255,0,0),
           0.1);

namedWindow("display",cv::WINDOW_AUTOSIZE);
imshow("display",image);
waitKey(0);
return 0;
}
*/




///TO DRAW A CIRCLE
/*
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;
int main()
{
    Mat img(480,640,CV_8UC1);
    img.setTo(0);
    Point center(img.cols/2,img.rows/2);
    int radius= img.rows/2;
    circle(img,
           center,
           radius,
           128,
           10);
           imshow("circle",img);
           waitKey(0);
           return 0;
}
*/




///TO DRAW A SQUARE ALONG WITH THE CIRCLE
/*
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;
int main()
{
    Mat img(480,640,CV_8UC1);
    img.setTo(0);
    Point center(img.cols/2,img.rows/2);
    int radius= img.rows/2;
    circle(img,
           center,
           radius,
           128,
           10);
Mat img1=img;
rectangle(img,
          center - Point(radius,radius),
          center + Point(radius,radius),
          255,
          10);
imshow("img1",img1);
imshow("circle",img);
waitKey(0);
return 0;
}
*/



///TO DISPLAY THE CIRCLE AND THE SQUARE WITH THE CIRCLE IN DIFFERENT DISPLAY WINDOWS
/*
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;
int main()
{
    Mat img(480,640,CV_8UC1);
    img.setTo(0);
    Point center(img.cols/2,img.rows/2);
    int radius= img.rows/2;
    circle(img,
           center,
           radius,
           128,
           10);
//Mat img1=img;
Mat img1;
img.copyTo(img1);
rectangle(img,
          center - Point(radius,radius),
          center + Point(radius,radius),
          255,
          10);

imshow("img1",img1);
imshow("img",img);
waitKey(0);
return 0;
}
*/




///TO DISPLAY A SOLID SQUARE
/*
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;
int main()
{
    Mat img(480,640,CV_8UC1);
    img.setTo(0);
    Point center(img.cols/2,img.rows/2);
    int radius= img.rows/2;
    circle(img,
           center,
           radius,
           128,
           10);
//Mat img1=img;
Mat img1;
img.copyTo(img1);
rectangle(img,
          center - Point(radius,radius),
          center + Point(radius,radius),
          255,
          10);
    Mat img2;
cvtColor(img1,img2,CV_GRAY2BGR);
int inscribed_radius=radius/sqrt(2);
Rect rect(
          center - Point(inscribed_radius,inscribed_radius),
          center + Point(inscribed_radius,inscribed_radius));
    Mat roi=img2(rect);
    roi.setTo(Scalar(0,185,118));

imshow("img2",img2);
imshow("img1",img1);
imshow("img",img);
waitKey(0);
return 0;
}
*/



///TO READ AND DISPLAY IMAGES FROM A VIDEO FILE
/*
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/video/video.hpp>
using namespace cv;
int main()
{
    VideoCapture input("Wildlife.wmv");
    Mat img;
    for(;;)
    {
     if(!input.read(img))
        break;
     imshow("img",img);
     char c=waitKey();
     if(c==27)
        break;
    }
}

*/



/// TO DISPLAY THE SPHERICAL OBJECTS IN A VEDIO FILE
/*
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/video/video.hpp>
#include<opencv2/opencv_modules.hpp>

using namespace cv;
using namespace std;
int main()
{
    VideoCapture input("Wildlife.wmv");
    Mat img;
    cv::OrbFeaturesFinder detector;
    vector<KeyPoint>keypoints;

    for(;;)
    {
     if(!input.read(img))
        break;

    detector(img, Mat(),keypoints);
for (size_t i=0; i<keypoints.size(); i++)
    circle(img,keypoints[i].pt, 2,
    Scalar(0,0,255),1);
imshow("img",img);
char c=waitKey();
if(c==27)
    break;
    }
    return 0;
}
*/




/// NEWTON RAPHSON MENTHOD
/*
#include <iostream>
#include <cmath>

using namespace std;

int main ()

{

float y = 9;
float epsilon = 0.1;
float guess = y/2.0;
int numberOfGuesses = 0;

while (abs(pow(guess,3)-y)>=epsilon && guess<y)
{
numberOfGuesses++;
guess = guess - ((pow(guess,3)-y)/(3*pow(guess,2)));
cout << "guess " << numberOfGuesses << " is " << guess <<endl;
cout << "Number of Guesses = " << numberOfGuesses << endl;
}

cout << guess << " is close to the cube root of " << y <<endl;
return 0;

}
*/


/// TO CONVERT AN IMAGE TO SOBEL FILTER
/*
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>

using namespace std;
using namespace cv;
int main()
{
    Mat img= imread("tiger.jpg");

    Mat dst;
    Sobel(img,dst,CV_32F,1,1);
    namedWindow("imageshow", WINDOW_FREERATIO);
    imshow("imageshow",dst/256);
    waitKey();
return 0;
}
*/




///RATIONAL NUMBERS
/*
#include<iostream>

using namespace std;

//CLASS
class calculator
{
      public:
        int sum(int num1,int num2,int den1,int den2);
        int mul(int num1,int num2,int den1,int den2);
        int sub(int num1,int num2,int den1,int den2);
        int div(int num1,int num2,int den1,int den2);

};
//ADDING FUNCTION
int calculator::sum(int num1,int den1,int num2,int den2)
{
  int num3,den3;
if(den1==den2)
   {
    num3= num1+num2;
    cout<<"The answer is"<<endl;
    cout<<num3<<"/"<<den1;
   }
else if(den1!=den2)
   {
     num3= (den2*num1)+(num2*den1);
     den3= den1*den2;
    cout<<"The answer is"<<endl;
    cout<<num3<<"/"<<den3;
   }
return 0;
}
//MULTIPLYING FUNCTION
int calculator::mul(int num1,int num2,int den1,int den2)
{   int num3,den3;
    num3= num1*num2;
    den3= den1*den2;
    cout<<"The answer is"<<endl;
    cout<<num3<<"/"<<den3;
    return 0;
}
//SUBTRACTING FUNCTION
int calculator::sub(int num1,int den1,int num2,int den2)
{
int num3,den3;
  if(den1==den2)
   {
    num3= num1-num2;
    cout<<"The answer is"<<endl;
    cout<<num3<<"/"<<den1;
   }
  else
   {
    num3= (den2*num1)-(num2*den1);
    den3= den1*den2;
    cout<<"The answer is"<<endl;
    cout<<num3<<"/"<<den3;
   }
return 0;
}
//DIVIDING FUNCTION
int calculator::div(int num1,int den1,int num2,int den2)
{  int num3,den3;
     num3= num1*den2;
     den3= den1*num2;
    cout<<"The answer is"<<endl;
    cout<<num3<<"/"<<den3;
return 0;

}
 //MAIN PROGRAM
#include<iostream>
using namespace std;
int main()
{
    int num1,num2,den1,den2,choice;
    calculator obj;
    cout<<"enter the numerator of 1st number"<<endl;
    cin>>num1;
    cout<<"enter the denominator of 1st number"<<endl;
    cin>>den1;
    cout<<"enter the numerator of 2nd number"<<endl;
    cin>>num2;
    cout<<"enter the denominator of 2nd number"<<endl;
    cin>>den2;
    cout<<"1.addition"<<endl;
    cout<<"2.multiplication"<<endl;
    cout<<"3.subtraction"<<endl;
    cout<<"4.division"<<endl;
    cout<<"5.remainder"<<endl;
    cout<<" Press any option"<<endl;
    cin>>choice;
    switch(choice)
    {
    case 1:
         obj.sum(num1,den1,num2,den2);
    break;

    case 2:
         obj.mul(num1,den1,num2,den2);
    break;

    case 3:
          obj.sub(num1,den1,num2,den2);
    break;

    case 4:
        obj.div(num1,den1,num2,den2);
    break;
    }

return 0;
}

*/



///TO ADD SOBEL FILTER TO A VIDEO
/*
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>

using namespace std;
using namespace cv;
int main()
{ Mat img;
  Mat dst;
  VideoCapture input("Wildlife.wmv");
  VideoWriter output(
  "Wildlife_sobel.avi",
  CV_FOURCC('X' , 'V' ,'I' , 'D' ),
  30,
  Size(input.get(CV_CAP_PROP_FRAME_WIDTH),
       input.get(CV_CAP_PROP_FRAME_HEIGHT)));

  for(;;)
  {
      if(!input.read(img))
        break;
      Sobel(img,dst,CV_8U,1,1);
     // output.write(dst);
      //imshow("img",img);
      imshow("DST",dst);
      char c= waitKey(30);
      if(c==' ')
        break;
  }
  return 0;
}
*/



/// USING CANNY FILTER
/*
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;


Mat src, src_gray;
Mat dst, detected_edges;


int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";


void CannyThreshold(int, void*)
{
  blur( src_gray, detected_edges, Size(3,3) );
  imshow( "blur", detected_edges);
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
  dst = Scalar::all(0);
  src.copyTo( dst, detected_edges);
  imshow( window_name, dst);
 }


int main( int argc, char** argv )
{
  //src = imread( argv[1] );
  src= imread("star.jpg");
  if( !src.data )
  { return -1; }

  dst.create( src.size(), src.type() );
  cvtColor( src, src_gray, CV_BGR2GRAY );
  namedWindow( "display", CV_WINDOW_AUTOSIZE );
  createTrackbar( "Min Threshold:", "display", &lowThreshold, max_lowThreshold, CannyThreshold );
  CannyThreshold(0, 0);
  waitKey(0);

  return 0;
  }
*/


/*
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

int main ()

{
    Mat frame1, frame2;
    VideoCapture cap;
    cap.open("Wildlife.wmv");
    cap.read(frame1) ;
    //cap.read(frame2) ;

    double FPS = cap.get(CV_CAP_PROP_FPS);
    cout << "FPS" << FPS << endl;

    double TFC = cap.get(CV_CAP_PROP_FRAME_COUNT);
    cout << "Total Frame Count" << TFC << endl;
    char ch = 0;

    while (cap.isOpened() && ch!=27)
    {
        imshow("videoPlayBack", frame1);
        if ((cap.get(CV_CAP_PROP_POS_FRAMES)+2)< TFC)
        {
            cout << cap.get(CV_CAP_PROP_POS_FRAMES) << endl;
            cap.read(frame2);
            frame1 = frame2.clone();
        }
        else
        {
            cout<<"end of video"<<endl;
            break;
        }
        ch = waitKey(1);
    }

    if (ch!=27)
    {
        waitKey(0);
    }

    return 0;
}
*/






///SMOOTHING OF IMAGE
/*
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;


int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;

Mat src; Mat dst;
char window_name[] = "Filter Demo 1";


int display_caption( const char* caption );
int display_dst( int delay );


 int main( int argc, char** argv )
 {
   namedWindow( window_name, CV_WINDOW_AUTOSIZE );


   src = imread( "sachin.jpg");

   display_caption( "Original Image" );


   dst = src.clone();

   display_dst( DELAY_CAPTION );

   cout << " came out " << endl;

   display_caption( "Homogeneous Blur" );

   for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
       { blur( src, dst, Size( i, i ), Point(-1,-1) );
         display_dst( DELAY_BLUR );
         }
   cout << " came out " << endl;


    display_caption( "Gaussian Blur" );

    for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
        { GaussianBlur( src, dst, Size( i, i ), 0, 0 );
          display_dst( DELAY_BLUR ); }

   cout << " came out " << endl;
     display_caption( "Median Blur" );

     for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
         { medianBlur ( src, dst, i );
           display_dst( DELAY_BLUR );}

     cout << " came out " << endl;
     display_caption( "Bilateral Blur" );

     for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
         { bilateralFilter ( src, dst, i, i*2, i/2 );
           display_dst( DELAY_BLUR ); }

     cout << " came out " << endl;
     display_caption( "End: Press a key!" );
     //imshow( window_name, dst );
     waitKey(0);
     return 0;
 }

 int display_caption( const char* caption )
 {
   dst = Mat::zeros( src.size(), src.type() );
   putText( dst, caption,
            Point( src.cols/4, src.rows/2),
            CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255) );

   imshow( window_name, dst );
   int c = waitKey( DELAY_CAPTION );
   if( c >= 0 ) { return -1; }
   return 0;
  }

  int display_dst( int delay )
  {
    imshow( window_name, dst );
    int c = waitKey ( delay );
    if( c >= 0 ) { return -1; }
    return 0;
  }

*/





///HISTOGRAM CALCULATION
/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


///function main
int main( int argc, char** argv )
{
  Mat src, dst;

  /// Load image
  src = imread("chessBoard.jpg" );

  if( !src.data )
    { return -1; }

  /// Separate the image in 3 places ( B, G and R )
  vector<Mat> bgr_planes;
  split( src, bgr_planes );

  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImageb( hist_h, hist_w, CV_8UC3, Scalar( 255,255,255) );
  Mat histImageg( hist_h, hist_w, CV_8UC3, Scalar( 255,255,255) );
  Mat histImager( hist_h, hist_w, CV_8UC3, Scalar( 255,255,255) );


  /// Normalize the result to [ 0, histImage.rows ]
  normalize(b_hist, b_hist, 0, histImageb.rows, NORM_MINMAX, -1, Mat() );
  normalize(g_hist, g_hist, 0, histImageg.rows, NORM_MINMAX, -1, Mat() );
  normalize(r_hist, r_hist, 0, histImager.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImageb, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
      line( histImageg, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
      line( histImager, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
  }

  /// Display
  //namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  imshow("calcHist Demo b", histImageb );
  imshow("calcHist Demo g", histImageg );
  imshow("calcHist Demo r", histImager );

  waitKey(0);

  return 0;
}
*/



///FINDING CONTOURS
/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void* );

///function main
int main( int argc, char** argv )
{
  /// Load source image and convert it to gray
  src = imread("star.jpg");

  /// Convert image to gray and blur it
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );

  /// Create Window
  const char* source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, src );

  createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
  thresh_callback( 1, 0 );

  waitKey(0);
  return(0);
}

///function thresh_callback
void thresh_callback(int, void* )
{
  Mat canny_output;
  vector<vector<Point> > contours;
  //vector<vector <Point>> contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using canny
  Canny( src_gray, canny_output, thresh, thresh*2, 3 );
  /// Find contours
  findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Draw contours
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
     }

  /// Show in a window
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
}
*/



///TO CREATE BOUNDARY BOXES
/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void* );

///function main
int main( int argc, char** argv )
{
  /// Load source image and convert it to gray
  src = imread("star.jpg");

  /// Convert image to gray and blur it
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );

  /// Create Window
  const char* source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, src );

  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
  thresh_callback( 0, 0 );

  waitKey(0);
  return(0);
}

///function thresh_callback
void thresh_callback(int, void* )
{
  Mat threshold_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using Threshold
  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
  /// Find contours
  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Approximate contours to polygons + get bounding rects and circles
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  vector<Point2f>center( contours.size() );
  vector<float>radius( contours.size() );

  for( int i = 0; i < contours.size(); i++ )
     { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
     }


  /// Draw polygonal contour + bonding rects + circles
  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
       circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
     }

  /// Show in a window
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
}
*/




///TO PRINT EVERY 100TH FRAME IN A VIDEO
/*
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <sstream>

using namespace std;
using namespace cv;


int main ()

{

VideoCapture cap;

Mat frame1;
Mat frame2;

cap.open("Wildlife.wmv");

double FPS = cap.get(CV_CAP_PROP_FPS);
cout << "FPS = " << FPS << endl;

double TFC = cap.get(CV_CAP_PROP_FRAME_COUNT);
cout << "totalFrameCount = " << TFC << endl;

cap.read(frame1);
//cap.read(frame2);

char ch = 0;

int loopCount = 0;

//ostringstream oss;

while (cap.isOpened() && ch != 27)

{

string frameName = "frameWindow";
//imshow("frames", frame1);
double index = cap.get(CV_CAP_PROP_POS_FRAMES);

if (fmod(index,100)==0)
{

imshow(frameName, frame1);
cout << index << endl;

}

if ((index+2) < TFC)
{

cap.read(frame2);
frame1 = frame2;

}

else
{
cout << "end of video" << endl;
break;
}

loopCount++;

ch = waitKey(10);
}

if (ch != 27)
        {
         waitKey(0);
        }

return 0;
}
*/





///TO DETECT STRAIGHT LINES (HOUGH TRANSFORM)
/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

void help()
{
 cout << "\nThis program demonstrates line finding with the Hough transform.\n"
         "Usage:\n"
         "./houghlines <image_name>, Default is pic1.jpg\n" << endl;
}

int main(int argc, char** argv)
{
 const char* filename = argc >= 2 ? argv[1] : "star.jpg";

 Mat src = imread(filename, 0);
 if(src.empty())
 {
     help();
     cout << "can not open " << filename << endl;
     return -1;
 }

 Mat dst, cdst;
 Canny(src, dst, 50, 200, 3);
 cvtColor(dst, cdst, CV_GRAY2BGR);

 #if 0
  vector<Vec2f> lines;
  HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );

  for( size_t i = 0; i < lines.size(); i++ )
  {
     float rho = lines[i][0], theta = lines[i][1];
     Point pt1, pt2;
     double a = cos(theta), b = sin(theta);
     double x0 = a*rho, y0 = b*rho;
     pt1.x = cvRound(x0 + 1000*(-b));
     pt1.y = cvRound(y0 + 1000*(a));
     pt2.x = cvRound(x0 - 1000*(-b));
     pt2.y = cvRound(y0 - 1000*(a));
     line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
  }
 #else
  vector<Vec4i> lines;
  HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
  for( size_t i = 0; i < lines.size(); i++ )
  {
    Vec4i l = lines[i];
    line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
  }
 #endif
 imshow("source", src);
 imshow("detected lines", cdst);

 waitKey();

 return 0;
}
*/



///HOUGH CIRCLE DETECTOR
/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;
//function main
int main(int argc, char** argv)
{
  Mat src, src_gray;

  /// Read the image
  src = imread("circles.jpg", 1 );

  if( !src.data )
    { return -1; }

  /// Convert it to gray
  cvtColor( src, src_gray, CV_BGR2GRAY );
  imshow("gray",src_gray);

  /// Reduce the noise so we avoid false circle detection
  GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );
  imshow("blur",src_gray);

  vector<Vec3f> circles;

  /// Apply the Hough Transform to find the circles
  HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/32, 200,50,0 ,0 );


  /// Draw the circles detected
  for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      circle( src, center, 2, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
   }

  /// Show your results
  namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
  imshow( "Hough Circle Transform Demo", src );

  waitKey(0);
  return 0;
}
*/





///CONVEX HULL
/*
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"
 #include <iostream>
 #include <stdio.h>
 #include <stdlib.h>

 using namespace cv;
 using namespace std;

 Mat src; Mat src_gray;
 int thresh = 100;
 int max_thresh = 255;
 RNG rng(12345);

 /// Function header
 void thresh_callback(int, void* );

//function main
int main( int argc, char** argv )
 {
   /// Load source image and convert it to gray
   src = imread("star.jpg", 1 );

   /// Convert image to gray and blur it
   cvtColor( src, src_gray, CV_BGR2GRAY );
   blur( src_gray, src_gray, Size(3,3) );

   /// Create Window
   const char* source_window = "Source";
   namedWindow( source_window, CV_WINDOW_AUTOSIZE );
   imshow( source_window, src );

   createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
   thresh_callback( 0, 0 );

   waitKey(0);
   return(0);
 }

 //function thresh_callback
 void thresh_callback(int, void* )
 {
   Mat src_copy = src.clone();
   Mat threshold_output;
   vector<vector<Point> > contours;
   vector<Vec4i> hierarchy;

   /// Detect edges using Threshold
   threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );

   /// Find contours
   findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

   /// Find the convex hull object for each contour
   vector<vector<Point> >hull( contours.size() );
   for( int i = 0; i < contours.size(); i++ )
      {  convexHull( Mat(contours[i]), hull[i], false ); }

   /// Draw contours + hull results
   Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
   for( int i = 0; i< contours.size(); i++ )
      {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        drawContours( drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
      }

   /// Show in a window
   namedWindow( "Hull demo", CV_WINDOW_AUTOSIZE );
   imshow( "Hull demo", drawing );
 }
*/






///TO DRAW LINES IN AN IMAGE
/*
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <sstream>

using namespace std;
using namespace cv;


int main ()

{
 //Mat img(240,320,CV_8UC3,Scalar(0,0,0));
 Mat img= imread("star.jpg");

 //cvtColor(img,img,COLOR_RGB2GRAY);




 static const int width = img.cols;
 static const int height = img.rows;

cout << "width = cols = " << width;
cout << "height = rows = " << height;

 Point mid_bottom, mid_top,top,bottom,b,t,mid_bottom1,mid_top1;

mid_bottom.x = width/1;
mid_bottom.y = 0;
mid_top.x = width/1;
mid_top.y = height;



mid_bottom1.x = 0;
mid_bottom1.y = width/8;
mid_top1.x = height;
mid_top1.y = width/8;



bottom.x = width/10;
bottom.y = 0;
top.x = width/10;
top.y = height;


b.x = width/4;
b.y = 0;
t.x = width/4;
t.y = height;


line(img, mid_bottom, mid_top,Scalar(255,0,0), 5);
//line(img, mid_bottom1, mid_top1,Scalar(255,0,0), 5);
//line(img, bottom,top,Scalar(0,255,0), 5);
line(img, b,t,Scalar(0,0,255), 5);



imshow("image",img);
waitKey();


return 0;
}
*/




///TO DRAW TWO LINES IN A VIDEO
/*
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <sstream>

using namespace std;
using namespace cv;


int main ()

{

VideoCapture cap;

Mat frame1;
Mat frame2;

cap.open("Wildlife.wmv");

double FPS = cap.get(CV_CAP_PROP_FPS);
cout << "FPS = " << FPS << endl;

double TFC = cap.get(CV_CAP_PROP_FRAME_COUNT);
cout << "totalFrameCount = " << TFC << endl;

cap.read(frame1);

char ch = 0;

int loopCount = 0;

while (cap.isOpened() && ch != 27)

{

double index = cap.get(CV_CAP_PROP_POS_FRAMES);

if (fmod(index,8)==0)
{

 static const int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
 static const int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);


Point mid_bottom, mid_top,top,bottom,b,t,mid_bottom1,mid_top1;

mid_bottom.x = width-200;
mid_bottom.y = 0;
mid_top.x = width-200;
mid_top.y = height;


b.x = width/6;
b.y = 0;
t.x = width/6;
t.y = height;

line(frame1, mid_bottom, mid_top,Scalar(255,0,0), 5);
line(frame1, b,t,Scalar(0,0,255), 5);


namedWindow("framedWindow",WINDOW_FREERATIO);
imshow("framedWindow", frame1);
cout << index << endl;

}

if ((index+2) < TFC)
{

cap.read(frame2);
frame1 = frame2;

}

else
{
cout << "end of video" << endl;
break;
}

loopCount++;

ch = waitKey(10);
}

if (ch != 27)
        {
         waitKey(0);
        }

return 0;
}
*/




/// TO CHANGE THE BRIGHTNESS AND CONTRAST
/*
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

double alpha; //Contrast control
int beta;  //Brightness control

int main( int argc, char** argv )
{
 /// Read image given by user
 Mat image = imread("ball.jpg");
 Mat new_image = Mat::zeros( image.size(), image.type() );

 /// Initialize values
 cout<<" Basic Linear Transforms "<<endl;
 cout<<"-------------------------"<<endl;
 cout<<"* Enter the alpha value [1.0-3.0]: ";
 cin>>alpha;
 cout<<"* Enter the beta value [0-100]: ";
 cin>>beta;

 /// Do the operation new_image(i,j) = alpha*image(i,j) + beta
 for( int y = 0; y < image.rows; y++ )
    { for( int x = 0; x < image.cols; x++ )
         { for( int c = 0; c < 3; c++ )
              {
      new_image.at<Vec3b>(y,x)[c] =
         saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
             }
    }
    }

 /// Create Windows
 namedWindow("Original Image", 1);
 namedWindow("New Image", 1);

 /// Show stuff
 imshow("Original Image", image);
 imshow("New Image", new_image);

 /// Wait until user press some key
 waitKey();
 return 0;
}
*/




/// TO CHANGE THE BRIGHTNESS AND CONTRAST (WIHTOUT FOR LOOPS)
/*
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

double alpha;
int beta;

int main( int argc, char** argv )
{
 /// Read image given by user
 Mat image = imread("ball.jpg");
 Mat new_image = Mat::zeros( image.size(), image.type() );

 /// Initialize values
 cout<<" Basic Linear Transforms "<<endl;
 cout<<"-------------------------"<<endl;
 cout<<"* Enter the alpha value [1.0-3.0]: ";cin>>alpha;
 cout<<"* Enter the beta value [0-100]: ";cin>>beta;

 image.convertTo(new_image, -1, alpha, beta);

 /// Create Windows
 namedWindow("Original Image", 1);
 namedWindow("New Image", 1);

 /// Show stuff
 imshow("Original Image", image);
 imshow("New Image", new_image);

 /// Wait until user press some key
 waitKey();
 return 0;
}
*/




///BASIC THRESHOLD OPERATION
/*
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

//Global Variables

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst;
const char* window_name = "Threshold Demo";

const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char* trackbar_value = "Value";

/// Function headers
void Threshold_Demo( int, void* );

//Main function
int main( int argc, char** argv )
{
  /// Load an image
  src = imread( "ball.jpg", 1 );
  imshow("original image",src);

  /// Convert the image to Gray
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Create a window to display results
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Create Trackbar to choose type of Threshold
  createTrackbar( trackbar_type,
                  window_name, &threshold_type,
                  max_type, Threshold_Demo );

  createTrackbar( trackbar_value,
                  window_name, &threshold_value,
                  max_value, Threshold_Demo );

  /// Call the function to initialize
  Threshold_Demo( 0, 0 );

  /// Wait until user finishes program
  while(true)
  {
    int c;
    c = waitKey( 20 );
    if( (char)c == 27 )
      { break; }
   }

}

void Threshold_Demo( int, void* )
{
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted



  threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );

  imshow( window_name, dst );
}
*/





///LAPLACE TRANSFORM
/*
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

//MAIN FUNCTION
int main( int argc, char** argv )
{
  Mat src, src_gray, dst;
  int kernel_size = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  const char* window_name = "Laplace Demo";

  int c;

  /// Load an image
  src = imread("parachute.jpg");
  imshow("original image",src);

  if( !src.data )
    { return -1; }

  //GAUSSIAN BLUR
  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
  imshow("blurred image",src);

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );
  imshow("gray image",src_gray);


  /// Create window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Apply Laplace function
  Mat abs_dst;

  Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( dst, abs_dst );

  /// Show what you got
  imshow( window_name, abs_dst );

  waitKey(0);

  return 0;
  }
*/




///ERODING AND DILATING
/*
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables
Mat src, erosion_dst, dilation_dst;

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

// Function Headers
void Erosion( int, void* );
void Dilation( int, void* );

//function main
int main( int argc, char** argv )
{
  /// Load an image
  src = imread("catt.jpg" );
  imshow("original image",src);

  if( !src.data )
  { return -1; }

  /// Create windows
  namedWindow( "Erosion Demo", CV_WINDOW_AUTOSIZE );
  namedWindow( "Dilation Demo", CV_WINDOW_AUTOSIZE );
  cvMoveWindow( "Dilation Demo", src.cols, 0 );

  /// Create Erosion Trackbar
  createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
                  &erosion_elem, max_elem,
                  Erosion );

  createTrackbar( "Kernel size:\n 2n +1", "Erosion Demo",
                  &erosion_size, max_kernel_size,
                  Erosion );

  /// Create Dilation Trackbar
  createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
                  &dilation_elem, max_elem,
                  Dilation );

  createTrackbar( "Kernel size:\n 2n +1", "Dilation Demo",
                  &dilation_size, max_kernel_size,
                  Dilation );

  /// Default start
  Erosion( 0, 0 );
  Dilation( 0, 0 );

  waitKey(0);
  return 0;
}

//function Erosion
void Erosion( int, void* )
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

  /// Apply the erosion operation
  erode( src, erosion_dst, element );
  imshow( "Erosion Demo", erosion_dst );
}

//function Dilation
void Dilation( int, void* )
{
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  dilate( src, dilation_dst, element );
  imshow( "Dilation Demo", dilation_dst );
}
*/




/// REMAPPING OF AN IMAGE
/*
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"
 #include <iostream>
 #include <stdio.h>

 using namespace cv;

 /// Global variables
 Mat src, dst;
 Mat map_x, map_y;
 const char* remap_window = "Remap demo";
 int ind = 0;

 /// Function Headers
 void update_map( void );

 //function main
 int main( int argc, char** argv )
 {
   /// Load the image
   src = imread("catt.jpg",1 );

  /// Create dst, map_x and map_y with the same size as src:
  dst.create( src.size(), src.type() );
  map_x.create( src.size(), CV_32FC1 );
  map_y.create( src.size(), CV_32FC1 );

  /// Create window
  namedWindow( remap_window, CV_WINDOW_AUTOSIZE );

  /// Loop
  while( true )
  {
    /// Each 1 sec. Press ESC to exit the program
    int c = waitKey( 1000 );

    if( (char)c == 27 )
      { break; }

    /// Update map_x & map_y. Then apply remap
    update_map();
    remap( src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );

    /// Display results
    imshow( remap_window, dst );
  }
  return 0;
 }

 //function update_map Fill the map_x and map_y matrices with 4 types of mappings
 void update_map( void )
 {
   ind = ind%4;

   for( int j = 0; j < src.rows; j++ )
   { for( int i = 0; i < src.cols; i++ )
       {
         switch( ind )
         {
           case 0:
             if( i > src.cols*0.25 && i < src.cols*0.75 && j > src.rows*0.25 && j < src.rows*0.75 )
               {
                 map_x.at<float>(j,i) = 2*( i - src.cols*0.25 ) + 0.5 ;
                 map_y.at<float>(j,i) = 2*( j - src.rows*0.25 ) + 0.5 ;
                }
             else
               { map_x.at<float>(j,i) = 0 ;
                 map_y.at<float>(j,i) = 0 ;
               }
                 break;
           case 1:
                 map_x.at<float>(j,i) = i ;
                 map_y.at<float>(j,i) = src.rows - j ;
                 break;
           case 2:
                 map_x.at<float>(j,i) = src.cols - i ;
                 map_y.at<float>(j,i) = j ;
                 break;
           case 3:
                 map_x.at<float>(j,i) = src.cols - i ;
                 map_y.at<float>(j,i) = src.rows - j ;
                 break;
         } // end of switch
       }
    }
  ind++;
}
*/




///AFFLINE TRANSFORMATION
/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/// Global variables
const char* source_window = "Source image";
const char* warp_window = "Warp";
const char* warp_rotate_window = "Warp + Rotate";

//function main
 int main( int argc, char** argv )
 {
   Point2f srcTri[3];
   Point2f dstTri[3];

   Mat rot_mat( 2, 3, CV_32FC1 );
   Mat warp_mat( 2, 3, CV_32FC1 );
   Mat src, warp_dst, warp_rotate_dst;

   /// Load the image
   src = imread("catt.jpg", 1 );

   /// Set the dst image the same type and size as src
   warp_dst = Mat::zeros( src.rows, src.cols, src.type() );

   /// Set your 3 points to calculate the  Affine Transform
   srcTri[0] = Point2f( 0,0 );
   srcTri[1] = Point2f( src.cols - 1, 0 );
   srcTri[2] = Point2f( 0, src.rows - 1 );

   dstTri[0] = Point2f( src.cols*0.0, src.rows*0.33 );
   dstTri[1] = Point2f( src.cols*0.85, src.rows*0.25 );
   dstTri[2] = Point2f( src.cols*0.15, src.rows*0.7 );

   /// Get the Affine Transform
   warp_mat = getAffineTransform( srcTri, dstTri );

   /// Apply the Affine Transform just found to the src image
   warpAffine( src, warp_dst, warp_mat, warp_dst.size() );

   // Rotating the image after Warp

   /// Compute a rotation matrix with respect to the center of the image
   Point center = Point( warp_dst.cols/2, warp_dst.rows/2 );
   double angle = -50.0;
   double scale = 0.1;

   /// Get the rotation matrix with the specifications above
   rot_mat = getRotationMatrix2D( center, angle, scale );

   /// Rotate the warped image
   warpAffine( warp_dst, warp_rotate_dst, rot_mat, warp_dst.size() );

   /// Show what you got
   namedWindow( source_window, CV_WINDOW_AUTOSIZE );
   imshow( source_window, src );

   namedWindow( warp_window, CV_WINDOW_AUTOSIZE );
   imshow( warp_window, warp_dst );

   namedWindow( warp_rotate_window, CV_WINDOW_AUTOSIZE );
   imshow( warp_rotate_window, warp_rotate_dst );

   /// Wait until user exits the program
   waitKey(0);

   return 0;
  }
*/




///BACKGROUND SUBTRACTION
/*
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <sstream>
using namespace cv;
using namespace std;
// Global variables
Mat frame; //current frame
Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
char keyboard; //input from keyboard

void processVideo();
void processImages(char* firstFrameFilename);

int main(int argc, char* argv[])
{
    //print help information
    //help();
    //check for the input parameter correctness

    //create GUI windows
    namedWindow("Frame");
    namedWindow("FG Mask MOG 2");
    //create Background Subtractor objects
    pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach
        //input data coming from a video
        processVideo();
    //destroy GUI windows
    destroyAllWindows();
    return EXIT_SUCCESS;
}
void processVideo() {
    //create the capture object
    VideoCapture capture("v6_bounce.mov");
    if(!capture.isOpened()){
        //error in opening the video input
        cout<< "Unable to open video file: "  << endl;
        exit(EXIT_FAILURE);
    }
    //read input data. ESC or 'q' for quitting
    keyboard = 0;
    while( keyboard != 'q' && keyboard != 27 ){
        //read the current frame
        if(!capture.read(frame)) {
            cout<< "Unable to read next frame." << endl;
            cout<< "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }
        //update the background model
        pMOG2->apply(frame, fgMaskMOG2);
        //get the frame number and write in on the current frame
        stringstream ss;
        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
                  cv::Scalar(255,255,255), -1);
        ss << capture.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        //show the current frame and the fg masks
        imshow("Frame", frame);
        imshow("FG Mask MOG 2", fgMaskMOG2);
        //get the input from the keyboard
        keyboard = (char)waitKey( 30 );
    }
    //delete capture object
    capture.release();
}
*/





/// TEMPLATE MATCHING
/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/// Global Variables
Mat img; Mat templ; Mat result;
const char* image_window = "Source Image";
const char* result_window = "Result window";

int match_method;
int max_Trackbar = 5;

/// Function Headers
void MatchingMethod( int, void* );

//function main
int main( int argc, char** argv )
{
  /// Load image and template
  img = imread("family.jpg",1 );
  cvtColor(img,img,CV_BGR2GRAY);
  templ = imread("face.jpg",1 );
  cvtColor(templ,templ,CV_BGR2GRAY);

  /// Create windows
  namedWindow( image_window, CV_WINDOW_FREERATIO );
  namedWindow( result_window, CV_WINDOW_FREERATIO );

  /// Create Trackbar
  const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
  createTrackbar( trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod );

  MatchingMethod( 0, 0 );

  waitKey(0);
  return 0;
}


 //function MatchingMethod
 //brief Trackbar callback

void MatchingMethod( int, void* )
{
  /// Source image to display
  Mat img_display;
  img.copyTo( img_display );

  /// Create the result matrix
  int result_cols =  img.cols - templ.cols + 1;
  int result_rows = img.rows - templ.rows + 1;

  result.create( result_rows, result_cols, CV_32FC1 );

  /// Do the Matching and Normalize
  matchTemplate( img, templ, result, match_method );
  normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

  /// Localizing the best match with minMaxLoc
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  Point matchLoc;

  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

  /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
  if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    { matchLoc = minLoc; }
  else
    { matchLoc = maxLoc; }

  /// Show me what you got
  rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
  rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );

  imshow( image_window, img_display );
  imshow( result_window, result );

  return;
}
*/




///ORB DETECTOR ( CODE NOT WORKING)
/*
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/video/video.hpp>

int main()
{
    using namespace cv;
    using namespace std;

  VideoCapture input("Wildlife.wmv");
  Mat img;

  cv::OrbFeaturesFinder detector;
  vector<KeyPoint>keypoints;

  for(;;)
    {
     if(!input.read(img))
        break;

        detector(img,Mat(),keypoints);
        for(size_t i=0; i<keypoints.size(); i++)
            circle(img,keypoints[i].pt, 2, Scalar(0,0,255),1);

     imshow("img",img);
     char c=waitKey();
     if(c==27)
        break;
    }
    return 0;
}
*/




///HOUGH TRANSFORM FOR A VIDEO
/*
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/video/video.hpp>
#include<opencv2/opencv_modules.hpp>

using namespace cv;
using namespace std;
int main()
{
    VideoCapture input("Wildlife.wmv");
    Mat img,img_gray,img_edges;


    for(;;)
    {
     if(!input.read(img))
        break;

        cvtColor(img,img_gray,CV_BGR2GRAY);
        Canny(img_gray,img_edges,230,460);
imshow("img_edges",img_edges);
char c=waitKey();
if(c==27)
    break;
    }
    return 0;
}
*/




///DETECTING LINES AND CIRCLES IN A VIDEO
/*
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/video/video.hpp>
#include<opencv2/opencv_modules.hpp>

using namespace cv;
using namespace std;
int main()
{
    VideoCapture input("Wildlife.wmv");
    Mat img,img_gray,img_edges;


    for(;;)
    {
     if(!input.read(img))
        break;

        cvtColor(img,img_gray,CV_BGR2GRAY);
        Canny(img_gray,img_edges,230,460);
        vector<Vec4i>lines;
        HoughLinesP(img_edges,lines,1,CV_PI/180,50,25,2);

        for(size_t i=0; i<lines.size(); i++)

          {
            Vec4i l= lines[i];
        line(img, Point(l[0], l[1]), Point(l[2], l[3]),Scalar(0,255,255),3);
          }

          GaussianBlur(img_gray,img_gray,Size(9,9),2,2);
    vector<Vec3f>circles;
    HoughCircles(img_gray,circles,CV_HOUGH_GRADIENT,1,10,100,25,1,25);

    for(size_t i=0; i<circles.size(); i++)
    {
        Vec3f c= circles[i];
        circle(img,Point(c[0],c[1]),
               c[2],Scalar(0,255,0),3);
    }

imshow("img",img);
char c=waitKey();
if(c==27)
    break;
    }
    return 0;
}
*/




///DRAWING TWO LINES IN BALL VIDEO
/*
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <sstream>

using namespace std;
using namespace cv;


int main ()

{

VideoCapture cap;

Mat frame1;
Mat frame2;

cap.open("v6_bounce.mov");

double FPS = cap.get(CV_CAP_PROP_FPS);
cout << "FPS = " << FPS << endl;

double TFC = cap.get(CV_CAP_PROP_FRAME_COUNT);
cout << "totalFrameCount = " << TFC << endl;

cap.read(frame1);

char ch = 0;

int loopCount = 0;

while (cap.isOpened() && ch != 27)

{

double index = cap.get(CV_CAP_PROP_POS_FRAMES);

//if (fmod(index,10)==0)
//{

 static const int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
 static const int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);


Point mid_bottom, mid_top,top,bottom,b,t,mid_bottom1,mid_top1;

mid_bottom.x = width-190;
mid_bottom.y = 0;
mid_top.x = width-190;
mid_top.y = height;


b.x = width/6;
b.y = 0;
t.x = width/6;
t.y = height;

line(frame1, mid_bottom, mid_top,Scalar(255,0,0), 2);
line(frame1, b,t,Scalar(0,0,255), 2);


//get the frame number and write in on the current frame
        stringstream ss;
        rectangle(frame1, Point(10, 2), Point(100,20),
                  Scalar(255,255,255), -1);
        ss << cap.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame1, frameNumberString.c_str(), cv::Point(15, 15),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));




namedWindow("framedWindow",WINDOW_FREERATIO);
imshow("framedWindow", frame1);
//cout << index << endl;

//}

if ((index+2) < TFC)
{

cap.read(frame2);
frame1 = frame2;

}

else
{
cout << "end of video" << endl;
break;
}

loopCount++;

ch = waitKey(40);
}

if (ch != 27)
        {
         waitKey(0);
        }

return 0;
}

*/



/*
///SECOND CODE FOR DRAWING REFERENCE LINES

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <sstream>

using namespace std;
using namespace cv;


int main ()

{

VideoCapture cap;

Mat frame1;
Mat frame2;


cap.open("v6_bounce.mov");


double FPS = cap.get(CV_CAP_PROP_FPS);
cout << "FPS = " << FPS << endl;

double TFC = cap.get(CV_CAP_PROP_FRAME_COUNT);
cout << "totalFrameCount = " << TFC << endl;

cap.read(frame1);
cap.read(frame2);

char ch = 0;

int loopCount = 0;

while (cap.isOpened() && ch != 27)

{


double index = cap.get(CV_CAP_PROP_POS_FRAMES);

Point mid_bottom, mid_top,b,t;


 int verticalLinePosition = (int)round((double)frame1.rows * 0.90);
    int verticalLinePosition2 = (int)round((double)frame1.rows * 0.20);

    mid_bottom.x = verticalLinePosition;
    mid_bottom.y = 50;

    mid_top.x = verticalLinePosition;
    mid_top.y = frame1.rows - 50;

    b.x = verticalLinePosition2;
    b.y = 50;

    t.x = verticalLinePosition2;
    t.y = frame1.rows - 50;

line(frame1, mid_bottom, mid_top,Scalar(0,0,0), 2);
line(frame1, b,t,Scalar(0,0,0), 2);


putText(frame1,
           "Reference line 1",
           Point(frame1.rows*0.90,frame1.rows*0.20),
           FONT_HERSHEY_DUPLEX,
           0.5,
           CV_RGB(255,0,0),
           0.3);



putText(frame1,
           "Reference line 2",
           Point(100,frame1.rows*0.20),
           FONT_HERSHEY_DUPLEX,
           0.5,
           CV_RGB(255,0,0),
           0.1);

putText(frame1,
           "Total frame count = 123 " ,
           Point(125,frame1.rows/27),
           FONT_HERSHEY_DUPLEX,
           0.5,
           CV_RGB(255,0,0),
           0.1);



putText(frame1,
           "Frames per sec = 25 " ,
           Point(140,frame1.rows/9),
           FONT_HERSHEY_DUPLEX,
           0.5,
           CV_RGB(0,0,0),
           0.1);


//get the frame number and write in on the current frame
        stringstream ss;
        rectangle(frame1, Point(10, 2), Point(100,20),
                  Scalar(255,255,255), -1);
        ss << cap.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame1, frameNumberString.c_str(), cv::Point(15, 15),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));


namedWindow("framedWindow",WINDOW_FREERATIO);
imshow("framedWindow", frame1);
//cout << index << endl;

//}

if ((index+2) < TFC)
{

cap.read(frame2);
frame1 = frame2;

}

else
{
cout << "end of video" << endl;
break;
}

loopCount++;

ch = waitKey(40);
}

if (ch != 27)
        {
         waitKey(0);
        }

return 0;
}

*/





/// HISTOGRAM EQUALIZATION


/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
  Mat src, dst;

  const char* source_window = "Source image";
  const char* equalized_window = "Equalized Image";

  src = imread("catt.jpg");

  if( !src.data )
    { cout<<"Usage: ./Histogram_Demo <path_to_image>"<<endl;
      return -1;}

  cvtColor( src, src, CV_BGR2GRAY );


  equalizeHist( src, dst );


  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  namedWindow( equalized_window, CV_WINDOW_AUTOSIZE );

  imshow( source_window, src );
  imshow( equalized_window, dst );

  waitKey(0);

  return 0;
}
*/




















