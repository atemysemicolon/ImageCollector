//
// Created by prassanna on 14/12/16.
//
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include "feature.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <cstdlib>
#include <ctime>
namespace cvml = cv::ml;


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
std::vector<cv::Mat> loadImages(std::vector<std::string> &filenames, int &min_width, int &min_height)
{
    cv::Mat img;
    std::vector<cv::Mat> imgs;
    for (std::string fname : filenames )
    {
        img = cv::imread(fname);

        if(img.cols<min_width)
            min_width = img.cols;
        if(img.rows<min_height)
            min_height=img.rows;

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




//Works only with max 4 channel image. TODO: Make general purpose
cv::Scalar getAveragePixelValue(cv::Mat &region)
{
    //std::cout<<region<<std::endl;
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
    {
        gts.at<uchar>(i) = (uchar)getFrequentPixelValue(regions[i]);
        //cv::imshow("regions", regions[i]);
        //cv::waitKey();
        //std::cout<<"."<<std::endl;
    }
    cv::Mat gts_alt;
    gts.convertTo(gts_alt, CV_32SC1);
    return gts_alt.clone();
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
    cv::Mat alt_desc;
    descs.convertTo(alt_desc, CV_32FC1);
    return alt_desc.clone();

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

std::vector<float> trainSvm(cv::Mat &data, cv::Mat &gt)
{
    cv::Ptr<cvml::SVM> svm;
    std::vector<float> weights;
    weights.resize(4+1, 0.0);
    int w = 0;
    svm = cvml::SVM::create();
    svm->setType(cvml::SVM::C_SVC);
    svm->setKernel(cvml::SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 1000, 0.01));
    svm->setC(0.5);

    cv::Ptr<cvml::TrainData> tdata = cvml::TrainData::create(data, cvml::ROW_SAMPLE, cv::Mat(gt));

    svm->train(tdata);

    //Debug and extract weights
    if(svm->isTrained())
    {
        std::cout<<"Trained"<<std::endl;
        cv::Mat alpha, wts;
        std::cout<<"Bias : "<<( -1 * (float)svm->getDecisionFunction(0, alpha, wts))<<std::endl;
        weights[w++] = ( -1 * (float)svm->getDecisionFunction(0, alpha, wts));

        cv::Mat svs = svm->getSupportVectors();
        for(int j=0;j<svs.cols;j++)
        {
            std::cout<< (svs.at<float>(j))<<", ";
            weights[w++] = (svs.at<float>(j));
        }



        std::cout<<std::endl;
        svm->clear();
        tdata.release();
    }

    return weights;

}

struct params
{
    int size;
    cv::Point p;
    std::vector<int> kernels;
    std::vector<float> weights;

};




//Todo :: Change functions getFrequentPixelValue and parseAnnotationKitti to work for general cases

int main()
{

    /////GENERAL OPERATIONS////
    //imgs->image_data
    //anns->ann_data
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
    int min_h = 10000;
    int min_w = 10000;
    std::vector<cv::Mat> image_data = loadImages(image_files,min_w,min_h);
    std::vector<cv::Mat> ann_data = loadImages(ann_files,min_w,min_h); //Expect Parsed
    int min_dim = min_h<min_w?min_h:min_w;
    //cv::Mat annimg = parseAnnotationKitti(ann_data[0]);
    ann_data = parseAllAnnotations(ann_data);

    //Variables go here



    srand (time(NULL));
    /////Per Region Operations OPERATIONS -> Each Node////

    //Node Inits
    //regions in regions_img and regions_ann
    std::vector<params> params_all;
    for(int num_its=0;num_its<1;num_its++)
    {
        //Select Random Parameters
        std::cout<<" ITERATION : "<<num_its<<std::endl;
        int random_size = (rand()%50)/2;

        //TODO : Sample x and y better please
        //int random_x = (min_w/2.0) + rand()%(min_w/2);
        int random_x = rand()%min_w;
        int random_y = rand()%min_h;
        cv::Point random_pt = cv::Point(random_x, random_y);

        int random_kernel_size = 1+rand()%4;
        std::vector<int> kernels_selected;// = {1,2,3,32};
        //select kernels randomly
        for(int j = 0;j<random_kernel_size;j++)
            kernels_selected.push_back(rand()%96);


        std::cout<<" Parameters Selected : -"<<std::endl;
        std::cout<<"Size : "<<random_size<<std::endl;
        std::cout<<"Point: "<<random_pt<<std::endl;
        std::cout<<"Kernels : ";
        for (std::vector<int>::const_iterator i = kernels_selected.begin(); i != kernels_selected.end(); ++i)
            std::cout << *i << ' ';
        std::cout<<std::endl;


        //Node getdata
        std::vector<cv::Mat> regions_img = getRegionsFromVector(image_data,random_pt, random_size);
        std::vector<cv::Mat> regions_ann = getRegionsFromVector(ann_data,random_pt, random_size);

        //Do GT check. to make sure gt is all nice
        cv::Mat annotations =getRegionGts(regions_ann);
        //std::cout<<annotations<<std::endl;
        int nr_zeros = cv::countNonZero(annotations);
        int tot = annotations.rows*annotations.cols;
        float factor = nr_zeros*1.0/tot;
        if ((factor == 0) || (factor ==1))
        {
            num_its--;
            std::cout<<" Bad GT, all are the same : "<<nr_zeros<<" / "<<tot<<std::endl;
            continue;
        }




        //Node GetRegions
        std::cout<<"Feature Computation"<<std::endl;
        std::vector<cv::Mat> fmaps = getAllFeatureMaps(regions_img,kernels_selected, classifier);
        std::cout<<"Pooling into regions"<<std::endl;
        cv::Mat data = getRegionDescs(fmaps);






        //Node Train clf
        std::cout<<"Training SVM"<<std::endl;

        bool flag_continue = true;
        std::vector<float> wts = trainSvm(data,annotations);

        params_all.push_back(params());
        params_all[num_its].size = random_size;
        params_all[num_its].p = random_pt;
        params_all[num_its].kernels= kernels_selected;
        params_all[num_its].weights = wts;



        /*while(flag_continue)
        {
            try
            {
                flag_continue=true;

            }
            catch(cv::Exception& e)
            {
                std::cout<<"Failed";
                flag_continue = true;
            }
        std::cout<<"Trained SVM"<<std::endl;

        }*/

    }

    std::cout<<"Saved Parameters -size :"<<params_all.size()<<std::endl;

    ////Predict For One Image////
    cv::Mat img = image_data[0];
    cv::Mat ann = ann_data[0];
    cv::Mat pred = cv::Mat::zeros(ann.rows,ann.cols, ann.type());


    //First just for one node
    params p = params_all[0];
    int n_blks_x = img.cols / (p.size*2);
    int n_blks_y = img.rows / (p.size*2);

    std::vector<cv::Mat> regions;
    //Dividing into regions -> grids

    std::cout<<"Getting all regions"<<std::endl;
    for(int i = 0;i<img.rows; i++)
    {
        for(int j= 0;j<img.cols;j++)
        {
            regions.push_back(getSelectedRegion(img, p.p, p.size));
        }
    }

    std::cout<<"Getting all fMaps"<<std::endl;
    std::vector<cv::Mat> fmaps = getAllFeatureMaps(regions, p.kernels, classifier);

    std::cout<<"Pooling all descriptors"<<std::endl;
    cv::Mat data = getRegionDescs(fmaps);
    std::vector<double> res;
    for(int row = 0;row<data.rows;row++)
    {
        cv::Mat dataRow = data.row(row);
        double response =p.weights[0];
        for(int i = 0;i<p.weights.size();i++)
        {
            response += p.weights[i+1]*dataRow.at<float>(i);
        }
        res.push_back(response);
    }



    //std::cout<<hc[0]<<std::endl;
    //cv::waitKey();

    //All images and annotations have been selected

    return 0;
}