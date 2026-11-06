#include <ros/ros.h>                             // La base per far parlare il codice con ROS
#include <sensor_msgs/PointCloud2.h>             // Il tipo di messaggio "nuvola di punti" di ROS
#include <pcl_conversions/pcl_conversions.h>     // Il "traduttore" tra il linguaggio ROS e PCL
#include <pcl/point_cloud.h>                     // Strutture dati PCL
#include <pcl/point_types.h>                     // Definizioni dei tipi di punti (XYZ, RGB, ecc.)
#include <pcl/filters/voxel_grid.h>              // Il filtro che riduce la densità dei punti
ros::Publisher pub;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input) {
    // 1. Convertire il messaggio ROS in un oggetto PCL
    pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
    pcl_conversions::toPCL(*input, *cloud);

    // 2. Creare l'oggetto del filtro (VoxelGrid)
    pcl::PCLPointCloud2 cloud_filtered;
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud(boost::shared_ptr<pcl::PCLPointCloud2>(cloud));
    // Dimensione del "cubetto" (voxel) - 1cm x 1cm x 1cm
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(cloud_filtered);

    //stampo a terminale la variazione dei punti
    ROS_INFO("Punti originali: %d - Punti filtrati: %d", 
         input->width * input->height, 
         cloud_filtered.width * cloud_filtered.height);

    // 3. Convertire di nuovo in messaggio ROS
    sensor_msgs::PointCloud2 output;
    pcl_conversions::moveFromPCL(cloud_filtered, output);

    // 4. Pubblicare il risultato
    pub.publish(output);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "my_pcl_tutorial");
    ros::NodeHandle nh;

    // ASCOLTA: dalla tua camera di Coppelia
    ros::Subscriber sub = nh.subscribe("/camera/depth_registered/points", 1, cloud_cb);

    // PUBBLICA: la nuvola alleggerita
    pub = nh.advertise<sensor_msgs::PointCloud2>("output_alleggerito", 1);

    ros::spin();
}