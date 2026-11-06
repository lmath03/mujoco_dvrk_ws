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
#include <pcl/common/common.h>
#include <pcl_ros/transforms.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/PointStamped.h>

#include <pcl/segmentation/extract_clusters.h> //cluster (euclideo)

#include <pcl/features/moment_of_inertia_estimation.h> //bounfing box OBB

#include <visualization_msgs/Marker.h>  //per la bounding box

class DVRKVisionNode {
private:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Publisher marker_pub_;
    ros::Subscriber sub_;

    //publisher della bounding box
    ros::Publisher obstacle_pose_pub_;
    ros::Publisher obstacle_dims_pub_;

    //creazione delle variabili globali per ascoltare le informazioni
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    void VoxelF(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered){
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
        voxel_filter.setInputCloud(cloud_filtered);
        //voxel_filter.setLeafSize(0.0025f, 0.0025f, 0.0025f); //più è alta la leaf più saranno distanziati i punti
        voxel_filter.setLeafSize(0.007f, 0.007f, 0.007f);
        voxel_filter.filter(*cloud_filtered); 
    }
    
    void PassThroughF(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered){
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud(cloud_filtered);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-1, 3);
        pass.filter(*cloud_filtered);

        pass.setFilterFieldName("y");
        pass.setFilterLimits(-1, 0.4);
        pass.filter(*cloud_filtered);

        pass.setFilterFieldName("z");
        pass.setFilterLimits(0, 1.4);
        pass.filter(*cloud_filtered);
    }

    void SORF(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered){
        //Remove outliers using a statistical outlier removal filter - serve a ripulire la nuvola di punti da tutto quel "rumore" o "pulviscolo" di punti isolati
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud_filtered);
        sor.setMeanK(20);               //distanze della media della densità dei punti a cui applicare il filtro
        sor.setStddevMulThresh(2.5);    //definisce quanto è aggressivo il filtro ()
        sor.filter(*cloud_filtered);
    }

    void ColorSegmentationF(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered){
        pcl::PointIndices::Ptr inliers_table_color(new pcl::PointIndices());

        // Range Tavolo
        int r_min = 140, r_max = 190;
        int g_min = 140, g_max = 185;
        int b_min = 130, b_max = 180;

        // Range Supporto Anelli (Marrone - valori stimati)
        int r_min_m = 40, r_max_m = 200;
        int g_min_m = 20,  g_max_m = 120;
        int b_min_m = 10,  b_max_m = 100;

        //range Bianco (per evitare il Panda)
        int white_min = 200;

        for (int i = 0; i < cloud_filtered->points.size(); ++i) {
            int r = cloud_filtered->points[i].r;
            int g = cloud_filtered->points[i].g;
            int b = cloud_filtered->points[i].b;

            bool is_table = (r >= r_min && r <= r_max &&
                            g >= g_min && g <= g_max &&
                            b >= b_min && b <= b_max);

            bool is_support = (r >= r_min_m && r <= r_max_m &&
                            g >= g_min_m && g <= g_max_m &&
                            b >= b_min_m && b <= b_max_m);

            bool is_white = (r >= white_min && g >= white_min && b >= white_min);

            // Se il colore del punto rientra nel range del tavolo e nel range del supporto anelli...
            if (is_table || is_support || is_white) {
                // ...salviamo il suo "numero di serie" (indice)
                inliers_table_color->indices.push_back(i); 
            }

        }

        //std::cout << "Punti tavolo trovati in base al colore: " << inliers_table_color->indices.size() << std::endl;

        // 4. Estraiamo (rimuoviamo) i punti trovati usando ExtractIndices (che già conosci)
        pcl::ExtractIndices<pcl::PointXYZRGB> extract_color;
        extract_color.setInputCloud(cloud_filtered);
        extract_color.setIndices(inliers_table_color);
        extract_color.setNegative(true);  // TRUE = elimina i punti trovati (il tavolo), tiene il resto (il robot)
        extract_color.filter(*cloud_filtered);
    }
    
    void ClusterF(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered){
        // Creating the KdTree object for the search method of the extraction
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
        tree->setInputCloud (cloud_filtered);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
        ec.setClusterTolerance (0.02); // 2cm
        ec.setMinClusterSize (100);
        ec.setMaxClusterSize (25000);
        ec.setSearchMethod (tree);
        ec.setInputCloud (cloud_filtered);
        ec.extract (cluster_indices);

        // MAX(PCD): Tieni solo il cluster più grande (il Robot, rimuovendo eventualmente i cluster più piccoli) ---
        if (!cluster_indices.empty()) 
        {
            // Creiamo un puntatore agli indici del PRIMO cluster (che PCL ordina per dimensione)
            pcl::PointIndices::Ptr max_indices(new pcl::PointIndices);
            *max_indices = cluster_indices[0]; 

            std::cout << "Robot isolato! Punti nel cluster principale: " << max_indices->indices.size() << std::endl;

            // Estraiamo il cluster dalla nuvola originale
            pcl::ExtractIndices<pcl::PointXYZRGB> extract_robot;
            extract_robot.setInputCloud(cloud_filtered);
            extract_robot.setIndices(max_indices);
            extract_robot.setNegative(false); // FALSE = Tieni solo il robot, cancella i puntini del tavolo residui
            extract_robot.filter(*cloud_filtered);
        } 
        else {
            ROS_WARN("Nessun cluster trovato! La nuvola potrebbe essere vuota.");
        }

        Eigen::Vector4f centroid_world;

        // DEBUG E GESTIONE ERRORI
        if (cloud_filtered->points.empty()) {
            ROS_WARN("Cilindro NON trovato");
            //*cloud_filtered = *cloud_filtered_backup; 
        } 
        else {

            // Creiamo una nuova "scatola" vuota per i punti trasformati
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_world(new pcl::PointCloud<pcl::PointXYZRGB>);
            try {
                cloud_filtered->header.stamp = 0; // Ignora il ritardo dei millisecondi e prende l'ultima TF disponibile
                
                if (!cloud_filtered->header.frame_id.empty() && cloud_filtered->header.frame_id[0] == '/') {
                    cloud_filtered->header.frame_id.erase(0, 1);
                }
    
                // TRASFORMAZIONE: Copiamo i punti della camera in cloud_world facendoli "ruotare" verso il mondo
                //pcl_ros::transformPointCloud("world", *cloud_filtered, *cloud_world, *tf_buffer_ptr);
                pcl_ros::transformPointCloud("world", *cloud_filtered, *cloud_world, tf_buffer_);                
                
                // ------------- ESTRAZIONE MARKER OBB ----------------------------
                //https://github.com/PointCloudLibrary/pcl/blob/master/doc/tutorials/content/sources/moment_of_inertia/moment_of_inertia.cpp
                pcl::MomentOfInertiaEstimation <pcl::PointXYZRGB> feature_extractor;
                feature_extractor.setInputCloud (cloud_world);
                feature_extractor.compute ();

                pcl::PointXYZRGB min_pt_OBB, max_pt_OBB, pos_OBB;
                Eigen::Matrix3f rot_OBB;
                feature_extractor.getOBB(min_pt_OBB, max_pt_OBB, pos_OBB, rot_OBB);

                visualization_msgs::Marker obb_marker;              //creiamo l'oggetto "messaggio" (con tutte le sue informazioni)
                obb_marker.header.frame_id = "world";               //disegnamo il cubo sulle coordinate mondo
                obb_marker.header.stamp = ros::Time::now();         //sincronizziamo il tempo
                obb_marker.ns = "obb_box"; // Namespace diverso
                obb_marker.id = 1;         // ID diverso
                obb_marker.type = visualization_msgs::Marker::CUBE; //forma geometrica del marker
                obb_marker.action = visualization_msgs::Marker::ADD;//dice ad RViz di aggiungere questo marker nella scena

                // posizione marker = posizione pont cloud boxxata
                obb_marker.pose.position.x = pos_OBB.x;
                obb_marker.pose.position.y = pos_OBB.y;
                obb_marker.pose.position.z = pos_OBB.z;

                // quaternioni per l'orientamento della bounding box
                Eigen::Quaternionf quat(rot_OBB);
                obb_marker.pose.orientation.x = quat.x();
                obb_marker.pose.orientation.y = quat.y();
                obb_marker.pose.orientation.z = quat.z();
                obb_marker.pose.orientation.w = quat.w();

                obb_marker.scale.x = max_pt_OBB.x - min_pt_OBB.x;
                obb_marker.scale.y = max_pt_OBB.y - min_pt_OBB.y;
                obb_marker.scale.z = max_pt_OBB.z - min_pt_OBB.z;

                // colore e opacità del marker
                obb_marker.color.r = 1.0f; 
                obb_marker.color.g = 0.0f; 
                obb_marker.color.b = 0.0f; 
                obb_marker.color.a = 0.5f;
                obb_marker.lifetime = ros::Duration(1);
                marker_pub_.publish(obb_marker); // Pubblica la rossa

                //preleviamo le info per il publish della bounding box
                geometry_msgs::Pose pose_msg;
                pose_msg.position.x = pos_OBB.x;
                pose_msg.position.y = pos_OBB.y;
                pose_msg.position.z = pos_OBB.z;

                Eigen::Quaternionf q(rot_OBB);
                pose_msg.orientation.x = q.x();
                pose_msg.orientation.y = q.y();
                pose_msg.orientation.z = q.z();
                pose_msg.orientation.w = q.w();

                obstacle_pose_pub_.publish(pose_msg);

                geometry_msgs::Vector3 dims_msg;
                dims_msg.x = max_pt_OBB.x - min_pt_OBB.x;
                dims_msg.y = max_pt_OBB.y - min_pt_OBB.y;
                dims_msg.z = max_pt_OBB.z - min_pt_OBB.z;

                obstacle_dims_pub_.publish(dims_msg);
            } 
            catch (tf2::TransformException &ex) {
                    ROS_WARN("TF non ancora pronta: %s", ex.what());
            }    
        }
    }
    
    void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*input, *cloud);

        //applichiamo i filtri
        VoxelF(cloud);
        PassThroughF(cloud);
        SORF(cloud);
        ColorSegmentationF(cloud);
        ClusterF(cloud);

        // Pubblicazione per RViz
        sensor_msgs::PointCloud2 output_msg;
        pcl::toROSMsg(*cloud, output_msg); // Trasforma i punti XYZ in messaggio ROS
        //output_msg.header = input->header;          // Fondamentale: copia il "DNA" (timestamp e coordinate) della camera
        output_msg.header.frame_id = input->header.frame_id;
        output_msg.header.stamp = ros::Time::now();

        pub_.publish(output_msg); // Spedisce a RViz
    }

public:
    DVRKVisionNode() : tf_listener_(tf_buffer_) {
        // Inizializziamo publisher e subscriber 
        pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/camera/segmented_arm", 1);
        marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/camera/bounding_box", 1);
        
        obstacle_pose_pub_ = nh_.advertise<geometry_msgs::Pose>("/vision/obstacle_pose", 1);
        obstacle_dims_pub_ = nh_.advertise<geometry_msgs::Vector3>("/vision/obstacle_dims", 1);

        sub_ = nh_.subscribe("/camera/depth_registered/points", 1, &DVRKVisionNode::cloud_cb, this);
        ROS_INFO("DVRK Vision Node Inizializzato");
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "davinci_vision_node");
    
    // Creiamo l'oggetto della nostra classe (questo fa partire tutto!)
    DVRKVisionNode vision_node;

    ros::spin();
    return 0;
}