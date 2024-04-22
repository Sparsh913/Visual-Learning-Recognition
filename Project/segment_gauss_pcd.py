import numpy as np
import cv2
import open3d as o3d
from copy import deepcopy

ply_path = "pcd2.ply"

pcd = o3d.io.read_point_cloud(ply_path)
points11 = np.asarray(pcd.points)
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points11)

# insert coordinate frame in pcd
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

# T = -0.272776812315 -0.074060574174 0.179010376334 -0.366493761539
# -0.193020626903 0.130256026983 -0.240235880017 0.399390488863
# -0.016514312476 -0.299140989780 -0.148925781250 0.708794176579
# 0.000000000000 0.000000000000 0.000000000000 1.000000000000

T = np.array([-0.272776812315, -0.074060574174, 0.179010376334, -0.366493761539,
                -0.193020626903, 0.130256026983, -0.240235880017, 0.399390488863,
                -0.016514312476, -0.299140989780, -0.148925781250, 0.708794176579,
                0.0, 0.0, 0.0, 1.0])

T = T.reshape(4, 4)
R = T[:3, :3]
t = T[:3, 3]
scale = 0.33456

print("determinant of R:", np.linalg.det(R))


# transform points
pcd = pcd.transform(T)

points = np.asarray(pcd.points)
colors = np.zeros_like(points)

centers = np.asarray([[0, 0, 0.0213], [-0.0663-0.00785, 0 , .0892], [-0.0743, 0, .5142], [-0.0743 +0.0174 -0.00785, 0.39225, .5142], [-0.0743 +0.0174-0.0531, 0.04165+0.39225+0.00785, .5142], [-0.0743 +0.0174-0.0531, 0.04165+0.39225+0.0531 , .5142 -0.04165-0.00785]]) # length = 6



# Segment the base
colors[np.where(points[:, 2] < centers[0, 2])[0]] = [1, 0, 0]

pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 1
condition = np.where((points[:, 2] > centers[0, 2])*(points[:, 0] > centers[1, 0])* (points[:, 2] < 0.2)
                     )[0]

colors[condition] = [0, 1, 0]

pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 2
condition1 = np.where((points[:,0] < centers[1,0]) * (points[:,2] > centers[0,2]) * (points[:,2] < 0.3))[0]
condition2 = np.where((points[:,0] < centers[2,0]) * (points[:, 2] >= 0.3) * (points[:, 1] < 0.1))[0]
condition = np.concatenate([condition1, condition2])
colors[condition] = [0, 0, 1]

pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 3
condition1 = np.where((points[:,0] > centers[2,0]) * (points[:,1] > (centers[2,1] - 0.1)) * (points[:,1] < 0.3) * (points[:,2] > 0.3))[0]
condition2 = np.where((points[:, 0] > centers[3, 0]) * (points[:, 1] >= 0.3) * (points[:, 2] > 0.3))[0]
condition = np.concatenate([condition1, condition2])
colors[condition] = [1, 1, 0]

pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 4
condition = np.where((points[:, 0] < centers[3, 0]) * (points[:, 1] > 0.25) * (points[:,1] < centers[4, 1]) * (points[:,2] > 0.3))[0]

colors[condition] = [1, 0, 1]

pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 5
condition = np.where((points[:, 0] < centers[3, 0]) * (points[:,1] > centers[4, 1]) * (points[:, 2] > centers[5, 2]))[0]
print('condition:', condition.shape)
colors[condition] = [0, 1, 1]
pcd.colors = o3d.utility.Vector3dVector(colors)


#Segment cubooid around robot
condition = np.where(np.logical_not((points[:, 0] > -0.3) * (points[:, 0] < 0.2) * (points[:, 1] > -0.3) * (points[:, 1] < 0.6) * (points[:, 2] > 0.0) * (points[:, 2] < 0.6)))[0]

colors[condition] = [0, 0, 0]

pcd.colors = o3d.utility.Vector3dVector(colors)




# Transform the points back
T_inv = np.linalg.inv(T)

pcd = pcd.transform(T_inv)

# visualize point cloud
o3d.visualization.draw_geometries([pcd])