import numpy as np
import cv2
import open3d as o3d

pcd_path = "gripper/gt_gripper_open.ply"
pcd = o3d.io.read_point_cloud(pcd_path)

# Load indices
indices = np.load('gripper/indices.npy')

pcd_points_torch = np.load('gripper/pcd_points_torch.npy')
print('pcd_points_torch:', pcd_points_torch.shape)

pcd_2 = o3d.geometry.PointCloud()
pcd_2.points = o3d.utility.Vector3dVector(pcd_points_torch)

# convert the points to numpy array
points = np.asarray(pcd.points) # Nx3
print('points:', points.shape)

# index the points
# seg_points = points[indices]
seg_points = pcd_points_torch[indices]
print('seg_points:', seg_points.shape)

# create a point cloud from the segmented points
seg_pcd = o3d.geometry.PointCloud()
seg_pcd.points = o3d.utility.Vector3dVector(seg_points)

# color the segmented points
# colors = np.zeros_like(seg_points)
# colors[:, 0] = 1
# seg_pcd.colors = o3d.utility.Vector3dVector(colors)

# visualize the segmented point cloud
o3d.visualization.draw_geometries([seg_pcd])
# o3d.visualization.draw_geometries([pcd])
