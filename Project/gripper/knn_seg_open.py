import numpy as np
import open3d as o3d
from sklearn.neighbors import KNeighborsClassifier

# Load point clouds
ply_path = 'gripper/gt_gripper_open.ply'
pcd = o3d.io.read_point_cloud(ply_path)

gt_ply = 'gripper/gt_gripper_open.ply'
gt_pcd = o3d.io.read_point_cloud(gt_ply)
# Ground truth links
gt_link0 = 'gripper/gripper_links/link_0.ply'
link0_pcd = o3d.io.read_point_cloud(gt_link0)
gt_link1 = 'gripper/gripper_links/link_1.ply'
link1_pcd = o3d.io.read_point_cloud(gt_link1)
gt_link2 = 'gripper/gripper_links/link_2.ply'
link2_pcd = o3d.io.read_point_cloud(gt_link2)
gt_link3 = 'gripper/gripper_links/link_3.ply'
link3_pcd = o3d.io.read_point_cloud(gt_link3)
gt_link4 = 'gripper/gripper_links/link_4.ply'
link4_pcd = o3d.io.read_point_cloud(gt_link4)
gt_link5 = 'gripper/gripper_links/link_5.ply'
link5_pcd = o3d.io.read_point_cloud(gt_link5)
gt_link6 = 'gripper/gripper_links/link_6.ply'
link6_pcd = o3d.io.read_point_cloud(gt_link6)
gt_link7 = 'gripper/gripper_links/link_7.ply'
link7_pcd = o3d.io.read_point_cloud(gt_link7)
gt_link8 = 'gripper/gripper_links/link_8.ply'
link8_pcd = o3d.io.read_point_cloud(gt_link8)
gt_link9 = 'gripper/gripper_links/link_9.ply'
link9_pcd = o3d.io.read_point_cloud(gt_link9)
gt_link10 = 'gripper/gripper_links/link_10.ply'
link10_pcd = o3d.io.read_point_cloud(gt_link10)

# Convert point clouds to numpy arrays
gt_points = np.asarray(gt_pcd.points)
link1_points = np.asarray(link1_pcd.points)
link2_points = np.asarray(link2_pcd.points)
link3_points = np.asarray(link3_pcd.points)
link4_points = np.asarray(link4_pcd.points)
link5_points = np.asarray(link5_pcd.points)
link6_points = np.asarray(link6_pcd.points)
link7_points = np.asarray(link7_pcd.points)
link8_points = np.asarray(link8_pcd.points)
link9_points = np.asarray(link9_pcd.points)
link10_points = np.asarray(link10_pcd.points)
pcd_points = np.asarray(pcd.points)

# Create labels for the ground truth points
gt_labels = np.zeros(gt_points.shape[0])
link1_labels = np.ones(link1_points.shape[0])
link2_labels = 2 * np.ones(link2_points.shape[0])
link3_labels = 3 * np.ones(link3_points.shape[0])
link4_labels = 4 * np.ones(link4_points.shape[0])
link5_labels = 5 * np.ones(link5_points.shape[0])
link6_labels = 6 * np.ones(link6_points.shape[0])
link7_labels = 7 * np.ones(link7_points.shape[0])
link8_labels = 8 * np.ones(link8_points.shape[0])
link9_labels = 9 * np.ones(link9_points.shape[0])
link10_labels = 10 * np.ones(link10_points.shape[0])

# Concatenate the points and labels
train_points = np.concatenate((gt_points, link1_points, link2_points, link3_points, link4_points, link5_points, link6_points, link7_points, link8_points, link9_points, link10_points))
train_labels = np.concatenate((gt_labels, link1_labels, link2_labels, link3_labels, link4_labels, link5_labels, link6_labels, link7_labels, link8_labels, link9_labels, link10_labels))

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors
knn.fit(train_points, train_labels)

# Predict the labels of the point cloud
pred_labels = knn.predict(pcd_points)

# Save the trained model
import joblib
joblib.dump(knn, 'gripper/knn_model.pkl')

# Visualize the predicted labels
pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(pcd_points))
for i in range(len(pcd_points)):
    
    if pred_labels[i] == 1:
        pcd.colors[i] = [0, 1, 0]  # Green for link 1
    elif pred_labels[i] == 2:
        pcd.colors[i] = [0, 0, 1]  # Blue for link 2
    elif pred_labels[i] == 3:
        pcd.colors[i] = [1, 1, 0]  # Yellow for link 3
    elif pred_labels[i] == 4:
        pcd.colors[i] = [1, 0, 1]  # Magenta for link 4
    elif pred_labels[i] == 5:
        pcd.colors[i] = [0, 1, 1]  # Cyan for link 5
    elif pred_labels[i] == 6:
        pcd.colors[i] = [0.5, 0, 0]  # Brown for link 6
    elif pred_labels[i] == 7:
        pcd.colors[i] = [0, 0.5, 0]  # Dark Green for link 7
    elif pred_labels[i] == 8:
        pcd.colors[i] = [0, 0, 0.5]  # Dark Blue for link 8
    elif pred_labels[i] == 9:
        pcd.colors[i] = [0.5, 0, 0.5]  # Dark Magenta for link 9
    elif pred_labels[i] == 10:
        pcd.colors[i] = [0, 0.5, 0.5]  # Dark Cyan for link 10
        
# Show only those points which are classified; don't show the rest
for i in range(len(pcd_points)):
    if pred_labels[i] == 0:
        pcd.colors[i] = [1, 1, 1]  # Black for unclassified points
        
        
o3d.visualization.draw_geometries([pcd])