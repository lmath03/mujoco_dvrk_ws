#include <ros/ros.h>
#include <iostream>
#include <filesystem>
#include <math.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/search/kdtree.h>

ros::Publisher pub;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // Convertiamo direttamente dal messaggio ROS al formato XYZ
    pcl::fromROSMsg(*input, *cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

// ------ inizio elaborazione dell'immagine ----------------------------------------------------------------------------------------------------------------------------------

    // VOXEL
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f); //più è alta la leaf più saranno distanziati i punti
    voxel_filter.filter(*cloud_filtered); 

    // PLANE SEGMENTATION

    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);                            //andiamo a creare gli indici
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);             //creiamo i coefficienti
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGB>());  //setuppiamo il cloud_plane

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);      //settiamo inoltre il model type, che modello useremo per fare la segmentazione dei nostri punti 
    seg.setMethodType(pcl::SAC_RANSAC);         //setuppiamo il tipo della forma geometrica che vogliamo segmentare e nel nostro caso sarà SACKMODEL_PLANE = piano
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.08);             //definisce quanto distante vogliamo includere il punto dalla segmentazione del piano
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers_plane, *coefficients_plane);   //segment, dobbiamo passare attraverso questa attività di segmentazione gli inliers plane e i coefficienti
    
    // Subito dopo seg.segment:
    std::cout << "Inliers trovati: " << inliers_plane->indices.size() << std::endl;
    
    //estraioamo il piano da seg
    pcl::ExtractIndices<pcl::PointXYZRGB> neg_plane_extracted;
    neg_plane_extracted.setInputCloud(cloud_filtered);
    neg_plane_extracted.setIndices(inliers_plane);
    neg_plane_extracted.setNegative(true);         // TRUE = Toglie il piano, tiene il resto
    neg_plane_extracted.filter(*cloud_filtered);
    
    // PASS THROUG FILTER -> limita lo spazio di osservazione del dataset/della cloud
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-0.1, 2);
    pass.filter(*cloud_filtered);

    pass.setFilterFieldName("y");
    pass.setFilterLimits(-3, 3);
    pass.filter(*cloud_filtered);

    pass.setFilterFieldName("z");
    pass.setFilterLimits(-1, 2.5);
    pass.filter(*cloud_filtered);

    //Remove outliers using a statistical outlier removal filter
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setMeanK(50);               //distanze della media della densità dei punti a cui applicare il filtro
    sor.setStddevMulThresh(1.5);    //definisce quanto è aggressivo il filtro
    sor.filter(*cloud_filtered);
    
    // RANSAC: SEGMENTAZIONE CILINDRI: utile per determinare il centroide della figura. Mettiamo la figura dentro al cilindo e ne calcoliamo il centroide
    //setting delle normali (set the normals)
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointNormal> normals_estimator;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>()); //metodo che ci permette di cercare le normali lungo tutti i punti e le normali che possono fittare dentro al cilindro
    normals_estimator.setSearchMethod(tree);
    normals_estimator.setInputCloud(cloud_filtered);
    normals_estimator.setKSearch(100); //valutiamo 100 punti per vedere se possiamo crare una normale
    normals_estimator.compute(*cloud_normals);

    // segmentazione del cilindro dalle normali
    pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::PointNormal> cylinder_segmentator;
    cylinder_segmentator.setOptimizeCoefficients(true);
    cylinder_segmentator.setModelType(pcl::SACMODEL_CYLINDER);
    cylinder_segmentator.setMethodType(pcl::SAC_RANSAC);
    cylinder_segmentator.setNormalDistanceWeight(0.1);
    cylinder_segmentator.setMaxIterations(10000);
    cylinder_segmentator.setDistanceThreshold(0.03);    //tolleranza
    cylinder_segmentator.setRadiusLimits(0.01, 0.25);   //intervallo raggio del cilindro
    cylinder_segmentator.setInputCloud(cloud_filtered);
    cylinder_segmentator.setInputNormals(cloud_normals);

    //prendiamo gli inliers e i coefficienti che descrivono il nostro cilindro in 3D
    pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
    cylinder_segmentator.segment(*inliers_cylinder,*coefficients_cylinder);

    //estraiamo il cilindro
    pcl::ExtractIndices<pcl::PointXYZRGB> cylinder_extracted;
    cylinder_extracted.setInputCloud(cloud_filtered);
    cylinder_extracted.setIndices(inliers_cylinder);
    cylinder_extracted.setNegative(false);      

    //filtriamo
    cylinder_extracted.filter(*cloud_filtered);

    //calcolo del centroide del cilindro estratto
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_filtered, centroid);


    // Pubblicazione per RViz
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud_filtered, output_msg); // Trasforma i punti XYZ in messaggio ROS
    output_msg.header = input->header;          // Fondamentale: copia il "DNA" (timestamp e coordinate) della camera
    
    //stampa a terminale vairie info
    //std::cout << "Punti finali inviati a RViz: " << cloud_filtered->points.size() << std::endl;
    std::cout << "centroid of cylinder in camera frame: ( "
              << centroid[0] << ","
              << centroid[1] << ","
              << centroid[2] << " )" << std::endl;

    pub.publish(output_msg); // Spedisce a RViz

}

int main(int argc, char** argv) {
    ros::init(argc, argv, "my_pcl_tutorial");
    ros::NodeHandle nh;

    // ASCOLTA: dalla tua camera di Coppelia
    ros::Subscriber sub = nh.subscribe("/camera/depth_registered/points", 1, cloud_cb);

    // PUBBLICA: la nuvola alleggerita
    pub = nh.advertise<sensor_msgs::PointCloud2>("/camera/segmented_arm", 1);

    ros::spin();
}