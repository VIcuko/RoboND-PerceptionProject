## Project: Perception Pick & Place

# Required Steps for a Passing Submission:

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.  

You're reading it!

[layout1-pointcloud]: ./writeup_images/layout1-pointcloud.png
[layout2-pointcloud]: ./writeup_images/layout2-pointcloud.png
[layout3-pointcloud]: ./writeup_images/layout3-pointcloud.png
[layout1-clusters]: ./writeup_images/layout1-clusters.png
[layout2-clusters]: ./writeup_images/layout2-clusters.png
[layout3-clusters]: ./writeup_images/layout3-clusters.png
[layout1-tags]: ./writeup_images/layout3-tags.png
[layout2-tags]: ./writeup_images/layout3-tags.png
[layout3-tags]: ./writeup_images/layout3-tags.png
[table]: ./writeup_images/table.png
[training_data]: ./writeup_images/training_data.png
[normalized_training_data]: ./writeup_images/normalized_training_data.png
[pr2]: ./writeup_images/pr2.png

1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify).

As shown during the Robotics course, the robot was trained using the sensor_stick package from the exercises previous to the project. Using this, the robot was trained considering the following parameters:

Number of bins: 32
Number of positions viewed per object: 30
Classifier used: svm.SVC(kernel='linear')
Modifying the models to include all the objects in the 3 scenarios.

Once having defined the parameters, I carried out the capture of all the objects and then executed the training.

The training data results were the following:

![training data][training_data]

And the normalized results were:

![training data][normalized_training_data]

2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.

As it may be observed in the file "project_template" in line 323 is subscribed to "/pr2/world/points".

```python
pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
```

3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.

As it may be seen in the code file, the process carried out is the following:

First of all, before beginning any further processing, as indicated within the project, the scenario contains some noise in the form of points distributed all over the place. For this reason, the first filter to apply would be an outlier filter to reduce the amount of noise in order to better process the image afterwards.

After some testing, I used the following values in order to initially filter the image:

```python
outlier_filter.set_mean_k(20)
outlier_filter.set_std_dev_mul_thresh(0.3)
cloud_filtered = outlier_filter.filter()
```

Then I applied a downsize of the voxel grid with a size of 0.01:

```python
vox = cloud_filtered.make_voxel_grid_filter()
LEAF_SIZE = 0.01
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
```

For the resulting cloud I then apply a filter in order to ommit irrelevant information from the point cloud, and therefore enabling a faster processing of the information. In this case, I cut everything below the table and above the objects, filtering along the z axis from position 0.6 to position 1.3:

```python
filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.3
    passthrough.set_filter_limits(axis_min, axis_max)
```

Then, in order to avoid including the image from the containers on the sides, I filter along the y axis from position -0.5 to position 0.5:

```python
filter_axis = 'y'
    passthrough2.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough2.set_filter_limits(axis_min, axis_max)
```

Once having filtered the part of the image to process, the next step would be to apply a RANSAC segmentation for the table element, being its opposite value the remaining items without the table. For this segmentation I used the available methods from the pcl library:

```python
eg = cloud_filtered.make_segmenter()

seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

max_distance = 0.01
seg.set_distance_threshold(max_distance)

inliers, coefficients = seg.segment()
```

Once having done the segmentation, I obtained the point cloud for the table using the inliers obtained:

```python
extracted_inliers_table = cloud_filtered.extract(inliers, negative=False)
```

Being the resulting cloud the following for all 3 scenarios:

![Table][Table]

And the point cloud for the objects without the table:

```python
extracted_outliers_objects = cloud_filtered.extract(inliers, negative=True)
```

Obtaining the following point clouds for each of the 3 scenarios:

![Layout 1 point cloud][layout1-pointcloud]
![Layout 2 point cloud][layout2-pointcloud]
![Layout 3 point cloud][layout3-pointcloud]

4. Apply Euclidean clustering to create separate clusters for individual items.

Once I had obtained the point cloud for all the objects on the table, the next step would be to identify each object on the table as an individual object. For doing so, I used an Euclidean clustering, using after some testing a tolerance of 0.03, a minimum cluster of 20 (had to reduce this one compared to the previous exercises mainly because the object of smaller size in the list) and a maximum cluster size of 1050, as you may view in the code:

```python
white_cloud = XYZRGB_to_XYZ(extracted_outliers_objects)
tree = white_cloud.make_kdtree()
ec = white_cloud.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance(0.03)
ec.set_MinClusterSize(20)
ec.set_MaxClusterSize(1050)
ec.set_SearchMethod(tree)
cluster_indices = ec.Extract()
```

Once the clusters have been identified, as shown in the course, I randomly assigned a colour to each cluster, in order to differentiate them from each other:

```python
cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
```

Creating at last, a new point cloud containing all clusters with their corresponding colour. Being the result of each cluster for each scenario the ones shown in the following images:

![Layout 1 clusters][layout1-clusters]
![Layout 2 clusters][layout2-clusters]
![Layout 3 clusters][layout3-clusters]

5. Perform object recognition on these objects and assign them labels (markers in RViz).

Once the image has been processed, the parallel process (as inicated at the beginning) is to train the robot in order to recognize each figure. For doing so, 

As shown during the Robotics course, the robot was trained using the sensor_stick package from the exercises previous to the project. Using this, the robot was trained considering the following parameters:

Number of bins: 32
Number of positions viewed per object: 30
Classifier used: svm.SVC(kernel='linear')
Modifying the models to include all the objects in the 3 scenarios.

Once having defined the parameters, I carried out the capture of all the objects and then executed the training.

The training data results were the following:

![training data][training_data]

And the normalized results were:

![training data][normalized_training_data]

Giving quite a high accuracy.

Once the training was done, I needed to implement the logic for the point cloud obtained from the RGBD camera after filtering, in order for the robot to identify the object infront with the one trained. For doing this, I had to analyse the image before clustering (since the colours in the clustered image aren't the real ones) using the positions for the clusters detected previously to obtain a histogram for colours in the point cloud and normals for each point on the point cloud:

```python
chists = compute_color_histograms(ros_cluster, using_hsv=True)
normals = get_normals(ros_cluster)
nhists = compute_normal_histograms(normals)
```

Once both histograms are obtained, a feature vector is computed:

```python
feature = np.concatenate((chists, nhists))
```

And then, with this vector, I used it to compare it to the trained data mentioned before in order to detect which object is the one being processed:

```python
prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
label = encoder.inverse_transform(prediction)[0]
detected_objects_labels.append(label)
```

Once the prediction has been carried out, the label of the object is then published to RViz in order to be viewed in the environment:

```python
label_pos = list(white_cloud[pts_list[0]])
label_pos[2] += .4
object_markers_pub.publish(make_label(label,label_pos, index))
```

After all this processing, the object identification result for all the scenarios are the following:

![1st scenario tags][layout1-tags]
![2nd scenario tags][layout2-tags]
![3rd scenario tags][layout3-tags]

6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.

Once I have confirmed that the robot identifies all of the objects appropriately, the next step was to identify the centroids of each object in order for the robot to be able to pick each one of them and the position them in the corresponding box on the sides.

I carried this out in the pr2_mover function as indicated. In order to locate the centroid, I used the code indicated in the project obtaining the centroid of each object as indicated by its position in the 3 axes:

```python
points_arr = ros_to_pcl(object_sorted.cloud).to_array()
centroids.append(np.mean(points_arr, axis=0)[:3])
```

where object_sorted.cloud is the point cloud of each object, considering this code is run inside a loop iterating over each object in the scenario.

Once having done this, the next step was to indicate the robot this position as the position to pick up and then indicate the position of the corresponding box where it should be dropped according to the yaml file for each scenario.

Pick location:

```python
pick_pose = Pose()
        pick_pose.position.x = s_centroid[0]
        pick_pose.position.y = s_centroid[1]
        pick_pose.position.z = s_centroid[2]
```

Drop location:

```python
place_pose = Pose()
dropbox_pos = green_dropbox if object_box == 'green' else red_dropbox
place_pose.position.x = dropbox_pos[0]
place_pose.position.y = dropbox_pos[1]
place_pose.position.z = dropbox_pos[2]
```

Once having defined the locations, it is required to indicate the robot which arm should be picking up the object depending on the drop location. For doing so, a simple logic indicating that if the dropbox is the red one, the arm should be the left one, and if it is the green box, the arm should be the right one:

```python
arm_name = String()
arm_name.data = 'right' if object_box == 'green' else 'left'
```

7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)

After having indicated the required information for the robot to pick up each object and drop it in the corresponding box, the next step is the save the results of the process in a yaml file containing all the relevant information as indicated by the project rubric. For doing so I implemented the following code at the end of the function, where the name of the file varies depending on the scenario being executed, and "yaml_params" are the information required to be saved in the yaml file:

```python
yaml_filename = "output_" + str(test_scene_num.data) + ".yaml"
send_to_yaml(yaml_filename, yaml_params)
```
(The resulting files may be viewed in the following folder: pr2_robot/scripts/, with the names of "output_1.yaml","output_2.yaml" and "output_3.yaml")

8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world). You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.

As it may be viewed in the yaml files, the robot picked up and dropped all of the items in the 3 scenarios. Although I had several issues not related to the content to be developed in the project, being the main one the grip. I tried to modified its functioning several times since it seemed the right gripper didn't pick up items, while the left one did, being the resulting gripping quite random, something which I believe has been very discussed in the slack forum. Apart from this, the results seemd to be quite satisfactory.

9. Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further. 

Perhaps using a non linear model for the training would increase in some cases the accuracy of the object identification. In my case it was high, but perhaps it could be increased. Also, increasing the number of bins for the histograms increases the accuracy of the results, but the main issue with this part is the fact that the processing becomes slower in RViz, taking longer to carry out the picking process.
Also the collision process could be improved, since there were some cases when after picking up an object, the robot got stuck, leading to several errors.
Apart from this, I think the robot did quite a good job at identifying each element. :+1:

10. Congratulations!  Your Done!



