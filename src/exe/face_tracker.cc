///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2010, Jason Mora Saragih, all rights reserved.
//
// This file is part of FaceTracker.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * The software is provided under the terms of this licence stricly for
//       academic, non-commercial, not-for-profit purposes.
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions (licence) and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions (licence) and the following disclaimer
//       in the documentation and/or other materials provided with the
//       distribution.
//     * The name of the author may not be used to endorse or promote products
//       derived from this software without specific prior written permission.
//     * As this software depends on other libraries, the user must adhere to
//       and keep in place any licencing terms of those libraries.
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite the following work:
//
//       J. M. Saragih, S. Lucey, and J. F. Cohn. Face Alignment through
//       Subspace Constrained Mean-Shifts. International Conference of Computer
//       Vision (ICCV), September, 2009.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////
#include <FaceTracker/Tracker.h>
#include <opencv/highgui.h>
#include <opencv2/photo/photo.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <omp.h>

#define N 2 // number of cameras

//=============================================================================
int dst_index(int i){
    i++;
    if(i>N-1) i=0;
    return i;
}

int src_index(int i){
    i--;
    if(i<0) i=N-1;
    return i;
}

//=============================================================================
void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi)
{
    int i,n = shape.rows/2; cv::Point p1,p2; cv::Scalar c;

    //draw triangulation
    c = CV_RGB(0,0,0);
    for(i = 0; i < tri.rows; i++){
        if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
                visi.at<int>(tri.at<int>(i,1),0) == 0 ||
                visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
        p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
                shape.at<double>(tri.at<int>(i,0)+n,0));
        p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
                shape.at<double>(tri.at<int>(i,1)+n,0));
        cv::line(image,p1,p2,c);
        p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
                shape.at<double>(tri.at<int>(i,0)+n,0));
        p2 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
                shape.at<double>(tri.at<int>(i,2)+n,0));
        cv::line(image,p1,p2,c);
        p1 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
                shape.at<double>(tri.at<int>(i,2)+n,0));
        p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
                shape.at<double>(tri.at<int>(i,1)+n,0));
        cv::line(image,p1,p2,c);
    }
    //draw connections
    c = CV_RGB(0,0,255);
    for(i = 0; i < con.cols; i++){
        if(visi.at<int>(con.at<int>(0,i),0) == 0 ||
                visi.at<int>(con.at<int>(1,i),0) == 0)continue;
        p1 = cv::Point(shape.at<double>(con.at<int>(0,i),0),
                shape.at<double>(con.at<int>(0,i)+n,0));
        p2 = cv::Point(shape.at<double>(con.at<int>(1,i),0),
                shape.at<double>(con.at<int>(1,i)+n,0));
        cv::line(image,p1,p2,c,1);
    }
    //draw points
    for(i = 0; i < n; i++){
        if(visi.at<int>(i,0) == 0)continue;
        p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
        c = CV_RGB(255,0,0); cv::circle(image,p1,2,c);
    }return;
}
void ExtractMask(cv::Mat &face_im, cv::Mat &mask, cv::Mat &frame, cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi, cv::Mat &dst_shape, cv::Mat &dst_visi)
{
    int i,n = shape.rows/2; cv::Point p1,p2; cv::Scalar c;
    int n_rows = tri.rows;
#pragma omp parallel for
    for(i = 0; i < n_rows; i++){
        cv::Point2f pts[3];
        cv::Point2f dst_pts[3];
        cv::Point2i pts_i[3];
        cv::Point2i dst_pts_i[3];
        cv::Mat tmp_im, tmp_mask;

        if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
                visi.at<int>(tri.at<int>(i,1),0) == 0 ||
                visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
        if(dst_visi.at<int>(tri.at<int>(i,0),0) == 0 ||
                dst_visi.at<int>(tri.at<int>(i,1),0) == 0 ||
                dst_visi.at<int>(tri.at<int>(i,2),0) == 0)continue;

        tmp_im = frame.clone();
        tmp_mask.create(frame.size(), CV_8UC1);
        tmp_im = cv::Scalar(0);
        tmp_mask = cv::Scalar(0);
        pts[0] = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
                shape.at<double>(tri.at<int>(i,0)+n,0));
        pts[1] = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
                shape.at<double>(tri.at<int>(i,1)+n,0));
        pts[2] = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
                shape.at<double>(tri.at<int>(i,2)+n,0));
        dst_pts[0] = cv::Point(dst_shape.at<double>(tri.at<int>(i,0),0),
                dst_shape.at<double>(tri.at<int>(i,0)+n,0));
        dst_pts[1] = cv::Point(dst_shape.at<double>(tri.at<int>(i,1),0),
                dst_shape.at<double>(tri.at<int>(i,1)+n,0));
        dst_pts[2] = cv::Point(dst_shape.at<double>(tri.at<int>(i,2),0),
                dst_shape.at<double>(tri.at<int>(i,2)+n,0));

        cv::Mat affineTransformM = cv::getAffineTransform(pts, dst_pts);

        for(int j=0; j<3; j++){
            pts_i[j].x = (int)pts[j].x;
            pts_i[j].y = (int)pts[j].y;
            dst_pts_i[j].x = (int)dst_pts[j].x;
            dst_pts_i[j].y = (int)dst_pts[j].y;
        }

        cv::fillConvexPoly(tmp_mask, pts_i, 3, cv::Scalar(255));
        frame.copyTo(tmp_im, tmp_mask);
        cv::warpAffine(tmp_im, tmp_im, affineTransformM, tmp_im.size());
        tmp_mask = cv::Scalar(0);
        cv::fillConvexPoly(tmp_mask, dst_pts_i, 3, cv::Scalar(255));
        // cv::warpAffine(tmp_mask, tmp_mask, affineTransformM, tmp_mask.size());
        // cv::bitwise_or(face_im, tmp_im, face_im);
        cv::bitwise_or(mask, tmp_mask, mask);
        // cv::blur(tmp_mask, tmp_mask,cv::Size(5,5));
        tmp_im.copyTo(face_im, tmp_mask);
    }

    return;
}

//=============================================================================
int parse_cmd(int argc, const char** argv,
        char* ftFile,char* conFile,char* triFile,
        bool &fcheck,double &scale,int &fpd)
{
    int i; fcheck = false; scale = 1; fpd = -1;
    for(i = 1; i < argc; i++){
        if((std::strcmp(argv[i],"-?") == 0) ||
                (std::strcmp(argv[i],"--help") == 0)){
            std::cout << "track_face:- Written by Jason Saragih 2010" << std::endl
                << "Performs automatic face tracking" << std::endl << std::endl
                << "#" << std::endl
                << "# usage: ./face_tracker [options]" << std::endl
                << "#" << std::endl << std::endl
                << "Arguments:" << std::endl
                << "-m <string> -> Tracker model (default: ../model/face2.tracker)"
                << std::endl
                << "-c <string> -> Connectivity (default: ../model/face.con)"
                << std::endl
                << "-t <string> -> Triangulation (default: ../model/face.tri)"
                << std::endl
                << "-s <double> -> Image scaling (default: 1)" << std::endl
                << "-d <int>    -> Frames/detections (default: -1)" << std::endl
                << "--check     -> Check for failure" << std::endl;
            return -1;
        }
    }
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"--check") == 0){fcheck = true; break;}
    }
    if(i >= argc)fcheck = false;
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-s") == 0){
            if(argc > i+1)scale = std::atof(argv[i+1]); else scale = 1;
            break;
        }
    }
    if(i >= argc)scale = 1;
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-d") == 0){
            if(argc > i+1)fpd = std::atoi(argv[i+1]); else fpd = -1;
            break;
        }
    }
    if(i >= argc)fpd = -1;
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-m") == 0){
            if(argc > i+1)std::strcpy(ftFile,argv[i+1]);
            else strcpy(ftFile,"../model/face2.tracker");
            break;
        }
    }
    if(i >= argc)std::strcpy(ftFile,"../model/face2.tracker");
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-c") == 0){
            if(argc > i+1)std::strcpy(conFile,argv[i+1]);
            else strcpy(conFile,"../model/face.con");
            break;
        }
    }
    if(i >= argc)std::strcpy(conFile,"../model/face.con");
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-t") == 0){
            if(argc > i+1)std::strcpy(triFile,argv[i+1]);
            else strcpy(triFile,"../model/face.tri");
            break;
        }
    }
    if(i >= argc)std::strcpy(triFile,"../model/face.tri");
    return 0;
}
//=============================================================================
int main(int argc, const char** argv)
{
    //parse command line arguments
    char ftFile[256],conFile[256],triFile[256];
    bool fcheck = false; double scale = 1; int fpd = -1; bool show = true;
    if(parse_cmd(argc,argv,ftFile,conFile,triFile,fcheck,scale,fpd)<0)return 0;

    //set other tracking parameters
    std::vector<int> wSize1(1); wSize1[0] = 7;
    std::vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
    int nIter = 5; double clamp=3,fTol=0.01;
    // FACETRACKER::Tracker model(ftFile);
    FACETRACKER::Tracker models[N];
    for(int i=0; i<N; i++){
        models[i].Load(ftFile);
    }
    cv::Mat tri=FACETRACKER::IO::LoadTri(triFile);
    cv::Mat con=FACETRACKER::IO::LoadCon(conFile);

    //initialize camera and display window
    // cv::Mat frame,gray,im,tmp_im; double fps=0; char sss[256]; std::string text;
    cv::Mat frames[N],grays[N],ims[N],masks[N],face_ims[N],masked_ims[N],edge_masks[N];
    double fps=0; char sss[256]; std::string text;
    // CvCapture* camera1 = cvCreateCameraCapture(CV_CAP_ANY); if(!camera1)return -1;
    CvCapture* cameras[N];
    for(int i=0; i<N; i++){
        cameras[i] = cvCreateCameraCapture(i); if(!cameras[i]) return -1;
        cvSetCaptureProperty(cameras[i], CV_CAP_PROP_FRAME_WIDTH, 640);
        cvSetCaptureProperty(cameras[i], CV_CAP_PROP_FRAME_HEIGHT, 480);
        std::stringstream ss;
        // sprintf(sss, "Camera %d", i);
        // cvNamedWindow(sss,1);
        sprintf(sss, "Swaped %d", i);
        cvNamedWindow(sss,1);
    }

    int64 t1,t0 = cvGetTickCount(); int fnum=0;

    std::cout << "Hot keys: "        << std::endl
        << "\t ESC - quit"     << std::endl
        << "\t d   - Redetect" << std::endl;

    //loop until quit (i.e user presses ESC)
    bool failed_flags[N];
    for(int i=0; i<N; i++){
        failed_flags[i] = true;
    }
    while(1){
        for(int i=0; i<N; i++){
            //grab image, resize and flip
            IplImage* I = cvQueryFrame(cameras[i]); if(!I)continue; frames[i] = I;
            if(scale == 1){
                ims[i] = frames[i];
                masks[i].create(ims[i].size(), CV_8UC1);
            }else{
                cv::resize(frames[i],ims[i],cv::Size(scale*frames[i].cols,scale*frames[i].rows));
                masks[i].create(cv::Size(scale*frames[i].cols,scale*frames[i].rows), CV_8UC1);
            }
            cv::flip(ims[i],ims[i],1); cv::cvtColor(ims[i],grays[i],CV_BGR2GRAY);
            masks[i] = cv::Scalar(0);
            edge_masks[i] = masks[i].clone();
            face_ims[i] = ims[i].clone();
            face_ims[i] = cv::Scalar(0);
            masked_ims[i] = ims[i].clone();
        }
#pragma omp parallel for
        for(int i=0; i<N; i++){
            //track this image
            std::vector<int> wSize; if(failed_flags[i])wSize = wSize2; else wSize = wSize1;
            if(models[i].Track(grays[i],wSize,fpd,nIter,clamp,fTol,fcheck) == 0){
                int idx = models[i]._clm.GetViewIdx(); failed_flags[i] = false;
                int dst_idx = models[dst_index(i)]._clm.GetViewIdx();
                ExtractMask(face_ims[i], masks[i],ims[i], models[i]._shape,con,tri,models[i]._clm._visi[idx], models[dst_index(i)]._shape,models[dst_index(i)]._clm._visi[dst_idx]);
                // Draw(ims[i],models[i]._shape,con,tri,models[i]._clm._visi[idx]);
                face_ims[i].copyTo(masked_ims[dst_index(i)], masks[i]);
            }else{
                if(show){cv::Mat R(masked_ims[i],cvRect(0,0,150,50)); R = cv::Scalar(0,0,255);}
                models[i].FrameReset(); failed_flags[i] = true;
            }
            //draw framerate on display image
            if(i==0){
                if(fnum >= 9){
                    t1 = cvGetTickCount();
                    fps = 10.0/((double(t1-t0)/cvGetTickFrequency())/1e+6);
                    t0 = t1; fnum = 0;
                }else fnum += 1;
                if(show){
                    sprintf(sss,"%d frames/sec",(int)round(fps)); text = sss;
                    cv::putText(masked_ims[i],text,cv::Point(10,20),
                            CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
                }
            }
            //show image and check for user input
            // sprintf(sss, "Camera %d", i);
            // imshow(sss,face_ims[i]);
            sprintf(sss, "Swaped %d", i);
            imshow(sss,masked_ims[dst_index(i)]);
        }

        int c = cvWaitKey(10);
        if(c == 27){
            break;
        }else if(char(c) == 'd'){
            for(int i=0; i<N; i++){
                models[i].FrameReset();
            }
        }
    }return 0;
}
//=============================================================================
