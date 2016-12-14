//
// Created by prassanna on 14/12/16.
//
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#ifndef PROJECT_IMAGECOLLECTOR_H
#define PROJECT_IMAGECOLLECTOR_H


class imagecollector {

};




#endif //PROJECT_IMAGECOLLECTOR_H

std::vector<std::string> getFilenames(std::string filename)
{
    std::vector<std::string> filenames;
    std::ifstream infile(filename);
    std::string line;
    while (std::getline(infile, line))
        filenames.push_back(line);

    return filenames;


}


std::string getImageName(int index, std::string prefix, std::vector<std::string> &filenames)
{
    return std::string(prefix+filenames[index]);
}

std::vector<std::string> appendList(std::string prefix, std::vector<std::string> &filenames)
{

    std::vector<std::string> fnames;
    for (std::string fname : filenames )
    {
     fnames.push_back(std::string(prefix+fname));
    }
    return fnames;
}
std::vector<cv::Mat> loadImages(std::vector<std::string> &filenames)
{
    cv::Mat img;
    std::vector<cv::Mat> imgs;
    for (std::string fname : filenames )
    {
        img = cv::imread(fname);
        imgs.push_back(img);
    }

    //cv::imshow("Image",img);
    //cv::waitKey();
    return imgs;
}


//TODO : UnModular. Change later. Make General Purpose
cv::Mat parseAnnotationKitti(cv::Mat &ann_img)
{
    cv::Mat ann = cv::Mat(ann_img.rows, ann_img.cols, CV_8U);
    cv::Vec3b color = cv::Vec3b(0,0,255);

    for (int x = 0;x<ann_img.cols;x++)
        for(int y=0;y<ann_img.rows;y++)
        {
            ann.at<uchar> (y,x) = (ann_img.at<cv::Vec3b>(y,x)  == color);
        }

    std::cout<<ann;

    return ann.clone();
}

int main()
{
    std::string train_loc = "/home/prassanna/Development/Datasets/Dataset_Kitti/training/Train.txt";
    std::string image_loc ="/home/prassanna/Development/Datasets/Dataset_Kitti/training/image_2/";
    std::string ann_loc = "/home/prassanna/Development/Datasets/Dataset_Kitti/training/gt_images_2/";
    std::vector<std::string> fnames = getFilenames(train_loc);
    //std::string fname = getImageName(0,image_loc, fnames);
    std::vector<std::string>  image_files = appendList(image_loc,fnames);
    std::vector<std::string>  ann_files = appendList(ann_loc,fnames);

    std::vector<cv::Mat> image_data = loadImages(image_files);
    std::vector<cv::Mat> ann_data = loadImages(ann_files); //Expect Parsed
    parseAnnotationKitti(ann_data[0]);
    return 0;
}