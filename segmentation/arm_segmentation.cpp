#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/search/kdtree.h>

ros::Publisher pub;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*input, *cloud);

    // Filtro Z per isolare il tavolo di lavoro
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 1.3); 
    pass.filter(*cloud_f);

    if (cloud_f->points.empty()) return;

    // Segmentazione Region Growing
    pcl::search::Search <pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
    reg.setInputCloud(cloud_f);
    reg.setSearchMethod(tree);
    reg.setDistanceThreshold(10);      
    reg.setPointColorThreshold(6);     
    reg.setRegionColorThreshold(5);    
    reg.setMinClusterSize(600);       

    std::vector <pcl::PointIndices> clusters;
    reg.extract(clusters);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*colored_cloud, output);
    output.header.frame_id = input->header.frame_id;
    pub.publish(output);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "arm_segmentation_node");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("/camera/depth/points", 1, cloud_cb);
    pub = nh.advertise<sensor_msgs::PointCloud2>("/camera/segmented_arm", 1);
    ros::spin();
    return 0;
}