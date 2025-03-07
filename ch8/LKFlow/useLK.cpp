#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std; 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <boost/format.hpp>

int main( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: useLK path_to_dataset"<<endl;
        return 1;
    }
    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";
    
    ifstream fin( associate_file );
    if ( !fin ) 
    {
        cerr<<"I cann't find associate.txt!"<<endl;
        return 1;
    }

    boost::format fmt("%s/image_%d/%06d.png"); // 10 us
    string rgbd_dataset_path_ = "/home/zh/data/img";
    
    string rgb_file, depth_file, time_rgb, time_depth;
    list< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
    cv::Mat color, depth, last_color, gray, last_gray;
    bool first_img = false;
    
    for ( int index=500; index<1000; index++ )
    {
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        // color = cv::imread( path_to_dataset+"/"+rgb_file );
        // depth = cv::imread( path_to_dataset+"/"+depth_file, -1 );
        // cv::cvtColor ( color, gray, cv::COLOR_BGR2GRAY );
        gray =
            cv::imread((fmt % rgbd_dataset_path_ % 0 % index).str(),
                   cv::IMREAD_GRAYSCALE);
        depth =
            cv::imread((fmt % rgbd_dataset_path_ % 1 % index).str(), cv::IMREAD_UNCHANGED);

        if ( gray.data==nullptr || depth.data==nullptr )
            continue;

        if (!first_img)
        {
            first_img = true;
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect( gray, kps );
            for ( auto kp:kps )
                keypoints.push_back( kp.pt );
            last_gray = gray;
            continue;
        }
        
        // 对其他帧用LK跟踪特征点
        vector<cv::Point2f> next_keypoints; 
        vector<cv::Point2f> prev_keypoints;
        for ( auto kp:keypoints )
            prev_keypoints.push_back(kp);
        vector<unsigned char> status;
        vector<float> error; 
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK( last_gray, gray, prev_keypoints, next_keypoints, status, error );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
        cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
        // 把跟丢的点删掉
        int i=0; 
        for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
        {
            if ( status[i] == 0 )
            {
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = next_keypoints[i];
            iter++;
        }
        cout<<"tracked keypoints: "<<keypoints.size()<<endl;
        if (keypoints.size() == 0)
        {
            cout<<"all keypoints are lost."<<endl;
            break; 
        }
        // 画出 keypoints

        cv::cvtColor ( gray, color, cv::COLOR_GRAY2BGR);
        cv::Mat img_show = color.clone();
        for ( auto kp:keypoints )
            cv::circle(img_show, kp, 2, cv::Scalar(0, 240, 0), 1);
        cv::imshow("corners", img_show);
        cv::waitKey(0);
        last_color = color;
    }
    return 0;
}