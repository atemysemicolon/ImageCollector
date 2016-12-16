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


//Currently only works when used as gt. Only works for binary classification
int  getFrequentPixelValue(cv::Mat &region)
{

    int histSize = 2;
    float range[] = { 0, 1 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    cv::Mat hist;
    calcHist( &region, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );


    /*double min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(hist, &min, &max, &min_loc, &max_loc);
    int ret = (int)max_loc.x;

    std::cout<<hist.rows<<","<<hist.cols<<","<<hist.channels()<<std::endl;*/
    if(hist.at<float>(0)>hist.at<float>(1))
        return 0;
    else
        return 1;


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


cv::Mat getRegionGts(std::vector<cv::Mat> &regions)
{
    cv::Mat gts(regions.size(),1,CV_8UC1);
    for (int i = 0;i<regions.size();i++)
        gts.at<uchar>(i) = (uchar)getFrequentPixelValue(regions[i]);

    return gts;
}



cv::Mat getRegionDescs(std::vector<cv::Mat> &regions)
{
    cv::Mat descs;

    for(cv::Mat reg : regions)
    {
        cv::Mat temp_desc = cv::Mat(getAveragePixelValue(reg));
        cv::transpose(temp_desc,temp_desc);
        descs.push_back(temp_desc);
    }
    return descs;

}


cv::Mat getFeatureMap(cv::Mat &region, std::vector<int> &kernels, CaffeModel &fe)
{
    std::vector<cv::Mat> selected_mats;
    std::vector<cv::Mat> hc = fe.forwardPassRescaleImg(region,1);

    for(int i = 0;i<kernels.size();i++)
        selected_mats.push_back(hc[kernels[i]]);


    cv::Mat fmap;
    cv::merge(selected_mats, fmap);
    //std::cout<<fmap.rows<<","<<fmap.cols<<","<<fmap.channels()<<std::endl;
    //cv::imshow("Feature map",selected_mats[0]*255);
    //cv::waitKey();
    return fmap.clone();
}



std::vector<cv::Mat> getAllFeatureMaps(std::vector<cv::Mat> &regions, std::vector<int> &kernels, CaffeModel &fe)
{
    std::vector<cv::Mat> fmaps;
    for( cv::Mat reg : regions)
    {
        fmaps.push_back(getFeatureMap(reg, kernels, fe));
    }

    return fmaps;
}


//Todo :: Change functions getFrequentPixelValue and parseAnnotationKitti to work for general cases

int main()
{
    //general inits
    std::string train_loc = "/home/prassanna/Development/workspace/NewKerasFramework/SuperDatasets/Dataset_Kitti/training/Train.txt";
    std::string image_loc ="/home/prassanna/Development/workspace/NewKerasFramework/SuperDatasets/Dataset_Kitti/training/image_2/";
    std::string ann_loc = "/home/prassanna/Development/workspace/NewKerasFramework/SuperDatasets/Dataset_Kitti/training/gt_images_2/";
    std::vector<std::string> fnames = getFilenames(train_loc);

    //Feature Inits
    int layer_nr = 1;
    std::string model_file="/home/prassanna/Libraries/caffe-master/models/bvlc_alexnet/deploy_small.prototxt";
    std::string trained_file="/home/prassanna/Libraries/caffe-master/models/bvlc_alexnet/bvlc_alexnet.caffemodel";
    std::string mean_file = "";
    std::string label_file="/home/prassanna/Libraries/caffe-master/data/ilsvrc12/synset_words.txt";
    std::vector<int> kernels = {1,2,3};
    CaffeModel classifier(model_file, trained_file, mean_file, label_file);

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

    std::vector<cv::Mat> regions_img = getRegionsFromVector(image_data,cv::Point(1000,300), 10);
    std::vector<cv::Mat> regions_ann = getRegionsFromVector(ann_data,cv::Point(1000,300), 10);

    cv::Mat gt_regions =getRegionGts(regions_ann);
    std::cout<<gt_regions<<gt_regions.rows<<", "<<gt_regions.cols<<std::endl;
    //std::vector<cv::Scalar> desc_regions = getRegionDescs(regions_img);






    std::vector<int> kernels_selected = {1,2,3,32};


    std::cout<<"Feature Computation"<<std::endl;
    std::vector<cv::Mat> fmaps = getAllFeatureMaps(regions_img,kernels_selected, classifier);

    std::cout<<"Pooling into regions"<<std::endl;
    cv::Mat descriptors = getRegionDescs(fmaps);


    //std::cout<<hc[0]<<std::endl;
    //cv::waitKey();

    //All images and annotations have been selected

    return 0;
}