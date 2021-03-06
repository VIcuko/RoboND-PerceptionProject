#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # TODO: Filter noise
    outlier_filter = cloud.make_statistical_outlier_filter()
    
    outlier_filter.set_mean_k(20)
    outlier_filter.set_std_dev_mul_thresh(0.3)
    cloud_filtered = outlier_filter.filter()

    # TODO: Voxel Grid Downsampling:
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    cloud_filtered = vox.filter()
   
    # TODO: PassThrough Filter:
    passthrough = cloud_filtered.make_passthrough_filter()

    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.3
    passthrough.set_filter_limits(axis_min, axis_max)

    cloud_filtered = passthrough.filter()

    # Now we filter y axis to avoid viewing the containers on the sides

    passthrough2 = cloud_filtered.make_passthrough_filter()

    filter_axis = 'y'
    passthrough2.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough2.set_filter_limits(axis_min, axis_max)

    cloud_filtered = passthrough2.filter()

    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()

    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    inliers, coefficients = seg.segment()

     # TODO: Extract inliers and outliers
    extracted_inliers_table = cloud_filtered.extract(inliers, negative=False)
    extracted_outliers_objects = cloud_filtered.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers_objects)
    tree = white_cloud.make_kdtree()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    ec = white_cloud.make_EuclideanClusterExtraction()

    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(1050)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

# Exercise-3 TODOs:

# Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = extracted_outliers_objects.extract(pts_list)

        # TODO: convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # TODO: complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        
        # Compute the associated feature vector
        feature = np.concatenate((chists, nhists))
        #labeled_features.append([feature, model_name])

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # TODO: Convert PCL data to ROS messages 
    ros_cloud_table = pcl_to_ros(extracted_inliers_table)
    ros_cloud_objects = pcl_to_ros(extracted_outliers_objects)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_table_pub.publish(ros_cloud_table)
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    test_case = 3
    pcl.save(extracted_inliers_table, "table"+ str(test_case) + ".pcd")
    pcl.save(extracted_outliers_objects, "objects"+ str(test_case) + ".pcd")
    pcl.save(cluster_cloud, "cluster_cloud"+ str(test_case) + ".pcd")
    
    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    # Create list of objects sorted as of the pick list
    sorted_objects = []
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    dropbox = []
    yaml_params = []

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # Check if list corresponds to current scenery
    if not len(object_list) == len(object_list_param):
        rospy.loginfo("List of detected objects does not match pick list.")
        return

    # Define test case
    test_scene_num = Int32()
    test_scene_num.data = 3

    # Define position of each dropbox
    red_dropbox = dropbox_param[0]['position']
    green_dropbox = dropbox_param[1]['position']

    # TODO: Loop through the pick list
    for i in range(len(object_list_param)):

        # Get the label for of each item in the pick list
        object_label = object_list_param[i]['name']

        # Iterate over object until a match is found between list and environment
        for current_object in object_list:
            if current_object.label == object_label:
                sorted_objects.append(current_object)
                object_list.remove(current_object)
                break

    for object_sorted in sorted_objects:

        labels.append(object_sorted.label)
    
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        points_arr = ros_to_pcl(object_sorted.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])

        for current_object in object_list_param:

            if current_object['name'] == object_sorted.label:
                dropbox.append(current_object['group'])
                break

    for i in range(len(sorted_objects)):

        object_name = String()
        object_name.data = object_list_param[i]['name']

        object_box = dropbox[i]

        # Convert <numpy.float64> data type to native float as expected by ROS
        np_centroid = centroids[i]
        s_centroid = [np.asscalar(element) for element in np_centroid]

        # Create 'pick_pose' message with centroid as the position data
        pick_pose = Pose()
        pick_pose.position.x = s_centroid[0]
        pick_pose.position.y = s_centroid[1]
        pick_pose.position.z = s_centroid[2]

        # TODO: Create 'place_pose' for the object
        place_pose = Pose()
        dropbox_pos = green_dropbox if object_box == 'green' else red_dropbox
        place_pose.position.x = dropbox_pos[0]
        place_pose.position.y = dropbox_pos[1]
        place_pose.position.z = dropbox_pos[2]

        # TODO: Assign the arm to be used for pick_place
        arm_name = String()
        arm_name.data = 'right' if object_box == 'green' else 'left'

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_params.append(make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose))

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    yaml_filename = "output_" + str(test_scene_num.data) + ".yaml"
    send_to_yaml(yaml_filename, yaml_params)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_objects_nonoise_pub = rospy.Publisher("/pcl_objects_nonoise", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
