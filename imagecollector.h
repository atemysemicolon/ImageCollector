//
// Created by prassanna on 14/12/16.
//
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include "feature.h"

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
    ann=1-ann;
    return ann.clone();
}




//not radius exactly, but half the edge of the square region
cv::Mat getSelectedRegion(cv::Mat &img, cv::Point p,int radius)
{
    cv::Mat reg;
    int x,y,w,h;
    x = (p.x-radius);
    y = p.y-radius;

    int radius_x=radius;
    int radius_y = radius;
    bool flag_change_radius=false;
    int x_stored;
    int y_stored;

    if(x<0) {
        radius_x += x;
        x = 0;
    }
    if(y<0){
        radius_y+=y;
        y=0;
    }

    w=radius_x*2;
    h=radius_y*2;

    if((x+w)>img.cols)
        w=img.cols-x;
    if((h+y)>img.rows)
        h=img.rows-y;

    cv::Rect roi = cv::Rect(x,y,w,h);
    reg = img(roi);
    return reg.clone();

}



cv::Scalar getAveragePixelValue(cv::Mat &region)
{
    cv::Scalar s = cv::mean(region);
    return s;
}


int  getFrequentPixelValue(cv::Mat &region)
{

    int histSize = 2;
    float range[] = { 0, 1 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    cv::Mat hist;
    calcHist( &region, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );


    double min, max;
    cv::Point min_loc, max_loc;

    cv::minMaxLoc(hist, &min, &max, &min_loc, &max_loc);

    //int ret = (int)max_loc.x;
    return ((int)max_loc.x);
}

std::vector<cv::Mat> parseAllAnnotations(std::vector<cv::Mat> &annimgs)
{
    std::vector<cv::Mat> annParsed;

    for (cv::Mat ann : annimgs)
    {
        annParsed.push_back(parseAnnotationKitti(ann));
    }
    return annParsed;
}

std::vector<cv::Mat> getRegionsFromVector(std::vector<cv::Mat> &imgs, cv::Point p,int radius)
{
    std::vector<cv::Mat> regions;
    for ( cv::Mat img : imgs )
    {
        regions.push_back(getSelectedRegion(img,p,radius));
    }
    return regions;
}


std::vector<int> getRegionGts(std::vector<cv::Mat> &regions)
{
    std::vector<int> gts;
    for ( cv::Mat reg : regions)
    {
        gts.push_back(getFrequentPixelValue(reg));
    }

    return gts;
}


std::vector<cv::Scalar> getRegionDescs(std::vector<cv::Mat> &regions)
{
    std::vector<cv::Scalar> descs;

    for(cv::Mat reg : regions)
    {
        descs.push_back(getAveragePixelValue(reg));
    }
    return descs;

}


//TODO : GEt all feature maps for one image and select the interesting one. Then make a function to do this for all regions
std::vector<cv::Mat> getFeatureMap(cv::Mat &region, std::vector<int> &kernels, CaffeModel &fe)
{
    std::vector<cv::Mat> selected_mats;



    return selected_mats;
}

int main()
{
    std::string train_loc = "/home/prassanna/Development/workspace/NewKerasFramework/SuperDatasets/Dataset_Kitti/training/Train.txt";
    std::string image_loc ="/home/prassanna/Development/workspace/NewKerasFramework/SuperDatasets/Dataset_Kitti/training/image_2/";
    std::string ann_loc = "/home/prassanna/Development/workspace/NewKerasFramework/SuperDatasets/Dataset_Kitti/training/gt_images_2/";
    std::vector<std::string> fnames = getFilenames(train_loc);

    //std::string fname = getImageName(0,image_loc, fnames);

    std::vector<std::string>  image_files = appendList(image_loc,fnames);
    std::vector<std::string>  ann_files = appendList(ann_loc,fnames);

    //Image files
    std::vector<cv::Mat> image_data = loadImages(image_files);
    std::vector<cv::Mat> ann_data = loadImages(ann_files); //Expect Parsed
    //cv::Mat annimg = parseAnnotationKitti(ann_data[0]);
    ann_data = parseAllAnnotations(ann_data);


    //imgs->image_data
    //anns->ann_data
    //regions in regions_img and regions_ann

    std::vector<cv::Mat> regions_img = getRegionsFromVector(image_data,cv::Point(10,10), 100);
    std::vector<cv::Mat> regions_ann = getRegionsFromVector(ann_data,cv::Point(300,300), 100);

    std::vector<int> gt_regions =getRegionGts(regions_ann);
    std::vector<cv::Scalar> desc_regions = getRegionDescs(regions_img);




    //Feature Inits
    int layer_nr = 1;
    std::string model_file="/home/prassanna/Libraries/caffe-master/models/bvlc_alexnet/deploy_small.prototxt";
    std::string trained_file="/home/prassanna/Libraries/caffe-master/models/bvlc_alexnet/bvlc_alexnet.caffemodel";
    std::string mean_file = "";
    std::string label_file="/home/prassanna/Libraries/caffe-master/data/ilsvrc12/synset_words.txt";
    std::vector<int> kernels = {1,2,3};
    CaffeModel classifier(model_file, trained_file, mean_file, label_file);
    std::vector<cv::Mat> hc = classifier.forwardPassRescaleImg(regions_img[0],1);

    //std::cout<<hc[0]<<std::endl;

    cv::imshow("Region",hc[0]*255);
    cv::waitKey();

    //All images and annotations have been selected

    return 0;
}