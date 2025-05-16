import numpy as np
import cv2
import open3d as o3d

# ply_path = "ur5e_gt.ply"
ply_path = "ur5e_gt.ply"

gt_pcd = o3d.io.read_point_cloud(ply_path)

points = np.asarray(gt_pcd.points)

# insert coordinate frame in pcd
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

# box condition
box_condition = ((points[:, 0] > -0.3) * (points[:, 0] < 0.2) * (points[:, 1] > -0.3) * (points[:, 1] < 0.6) * (points[:, 2] > 0.0) * (points[:, 2] < 0.6))


# Segment Base
centers = np.asarray([[0, 0, 0.09752], [-0.07282, 0 , .1625], [-0.06109, 0, .5873], [-0.05988, 0.39225, .5873], [-0.1333, 0.05335+0.39225, .5873], [-0.1333, 0.04655+0.39225+0.05335, .5873 -0.05328]]) # length = 6

points = np.asarray(gt_pcd.points)
colors = np.zeros_like(points)

colors[np.where(points[:, 2] < centers[0, 2])[0]] = [1, 0, 0]

gt_pcd.colors = o3d.utility.Vector3dVector(colors)


#Segment Link 1

condition = np.where((points[:, 2] > centers[0, 2])*(points[:, 0] > centers[1, 0])* (points[:, 2] < 0.3))[0]


colors[condition] = [0, 1, 0]

gt_pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 2

condition1 = np.where((points[:,0] < centers[1,0]) * (points[:,2] > centers[0,2]) * (points[:,2] < 0.3))[0]
condition2 = np.where((points[:,0] < centers[2,0]) * (points[:, 2] >= 0.3) * (points[:, 1] < 0.1))[0]
condition = np.concatenate([condition1, condition2])
colors[condition] = [0, 0, 1]

gt_pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 3
condition1 = np.where((points[:,0] > centers[2,0]) * (points[:,1] > (centers[2,1] - 0.1)) * (points[:,1] < 0.3) * (points[:,2] > 0.3))[0]
condition2 = np.where((points[:, 0] > centers[3, 0]) * (points[:, 1] >= 0.3) * (points[:, 2] > 0.3))[0]
condition = np.concatenate([condition1, condition2])
colors[condition] = [1, 1, 0]

gt_pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 4
condition = np.where((points[:, 0] < centers[3, 0]) * (points[:, 1] > 0.25) * (points[:,1] < centers[4, 1]) * (points[:,2] > 0.3))[0]

colors[condition] = [1, 0, 1]

gt_pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 5
condition = np.where((points[:, 0] < centers[3, 0]) * (points[:,1] > centers[4, 1]) * (points[:, 2] > centers[5, 2]))[0]

colors[condition] = [0, 1, 1]

gt_pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 6
condition = np.where(((points[:, 0] < centers[3, 0]) * (points[:,1] > centers[4, 1]) * (points[:, 2] < centers[5, 2]))*box_condition)[0]

colors[condition] = [0, 0, 0]

gt_pcd.colors = o3d.utility.Vector3dVector(colors)


# visualize point cloud
o3d.visualization.draw_geometries([gt_pcd, mesh_frame])