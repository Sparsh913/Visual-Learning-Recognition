import numpy as np
import cv2
import open3d as o3d

ply_path = "gt_pcd.ply"

gt_pcd = o3d.io.read_point_cloud(ply_path)

# insert coordinate frame in pcd
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

# Segment Base
centers = np.asarray([[0, 0, 0.0213], [-0.0663-0.00785, 0 , .0892], [-0.0743, 0, .5142], [-0.0743 +0.0174 -0.00785, 0.39225, .5142], [-0.0743 +0.0174-0.0531, 0.04165+0.39225+0.00785, .5142], [-0.0743 +0.0174-0.0531, 0.04165+0.39225+0.0531 , .5142 -0.04165-0.00785]]) # length = 6

points = np.asarray(gt_pcd.points)
colors = np.zeros_like(points)

colors[np.where(points[:, 2] < centers[0, 2])[0]] = [1, 0, 0]

gt_pcd.colors = o3d.utility.Vector3dVector(colors)


#Segment Link 1

condition = np.where((points[:, 2] > centers[0, 2])*(points[:, 0] > centers[1, 0])* (points[:, 2] < 0.2)
                     )[0]


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


# visualize point cloud
o3d.visualization.draw_geometries([gt_pcd, mesh_frame])