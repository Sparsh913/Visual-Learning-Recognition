import numpy as np
import cv2
import open3d as o3d
from copy import deepcopy

ply_path = "ur_pusher/pcd_1.ply"

pcd = o3d.io.read_point_cloud(ply_path)
points11 = np.asarray(pcd.points)
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points11)

# insert coordinate frame in pcd
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

# T = 0.144820347428 0.019638715312 -0.143120646477 -0.240426957607
# 0.141510024667 0.021471565589 0.146136879921 0.408296585083
# 0.029053352773 -0.202473223209 0.001615444897 0.492784976959
# 0.000000000000 0.000000000000 0.000000000000 1.000000000000

T = np.array([0.144820347428, 0.019638715312, -0.143120646477, -0.240426957607,
                0.141510024667, 0.021471565589, 0.146136879921, 0.408296585083,
                0.029053352773, -0.202473223209, 0.001615444897, 0.492784976959,
                0.0, 0.0, 0.0, 1.0])

T = T.reshape(4, 4)
R = T[:3, :3]
t = T[:3, 3]
scale = np.linalg.inv(R)**(1/3)
print("scale:", scale)

print("determinant of R:", np.linalg.det(R))


# transform points
pcd = pcd.transform(T)

points = np.asarray(pcd.points)
colors = np.zeros_like(points)
box_condition = ((points[:, 0] > -0.3) * (points[:, 0] < 0.2) * (points[:, 1] > -0.15) * (points[:, 1] < 0.6) * (points[:, 2] > 0.0) * (points[:, 2] < 0.7))
# box_condition = ((points[:, 0] > -0.4) * (points[:, 0] < 0.3) * (points[:, 1] > -0.4) * (points[:, 1] < 0.7) * (points[:, 2] > 0.0) * (points[:, 2] < 0.7))



# Segment Base
centers = np.asarray([[0, 0, 0.09752], [-0.07282, 0 , .1625], [-0.06109, 0, .5873], [-0.05988, 0.39225, .5873], [-0.1333, 0.05335+0.39225, .5873], [-0.1333, 0.04655+0.39225+0.05335, .5873 -0.05328]]) # length = 6

points = np.asarray(pcd.points)
colors = np.zeros_like(points)

condition = np.where((points[:, 2] < centers[0, 2])*(points[:,1] > -0.1)*(points[:,1] < 0.1)*(points[:,0] > -0.1)*(points[:,0] < 0.1)*box_condition)[0]

colors[condition] = [1, 0, 0]

pcd.colors = o3d.utility.Vector3dVector(colors)


#Segment Link 1

condition = np.where((points[:, 2] > centers[0, 2])*(points[:, 0] > centers[1, 0])* (points[:, 2] < 0.3)*(points[:,1] < 0.1)*(points[:,1] > -0.1)*box_condition)[0]


colors[condition] = [0, 1, 0]

pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 2

condition1 = np.where((points[:,0] < centers[1,0]) * (points[:,2] > centers[0,2]) * (points[:,2] < 0.3)*box_condition)[0]
condition2 = np.where((points[:,0] < centers[2,0]) * (points[:, 2] >= 0.3) * (points[:, 1] < 0.1)*box_condition)[0]
condition = np.concatenate([condition1, condition2])
colors[condition] = [0, 0, 1]

pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 3
condition1 = np.where((points[:,0] > centers[2,0]) * (points[:,1] > (centers[2,1] - 0.1)) * (points[:,1] < 0.3) * (points[:,2] > 0.3)*box_condition)[0]
condition2 = np.where((points[:, 0] > centers[3, 0]) * (points[:, 1] >= 0.3) * (points[:, 2] > 0.3)*box_condition)[0]
condition = np.concatenate([condition1, condition2])
colors[condition] = [1, 1, 0]

pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 4
condition = np.where((points[:, 0] < centers[3, 0]) * (points[:, 1] > 0.25) * (points[:,1] < centers[4, 1]) * (points[:,2] > 0.3)*box_condition)[0]

colors[condition] = [1, 0, 1]

pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 5
condition = np.where((points[:, 0] < centers[3, 0]) * (points[:,1] > centers[4, 1]) * (points[:, 2] > centers[5, 2])*box_condition)[0]

colors[condition] = [0, 1, 1]

pcd.colors = o3d.utility.Vector3dVector(colors)

# Segment Link 6
condition = np.where(((points[:, 0] < centers[3, 0]) * (points[:,1] > centers[4, 1]) * (points[:, 2] < centers[5, 2]))*box_condition)[0]

colors[condition] = [0, 0, 0]

pcd.colors = o3d.utility.Vector3dVector(colors)


#Segment cubooid around robot
# condition = np.where(np.logical_not((points[:, 0] > -0.3) * (points[:, 0] < 0.2) * (points[:, 1] > -0.3) * (points[:, 1] < 0.6) * (points[:, 2] > 0.0) * (points[:, 2] < 0.6))*box_condition)[0]

# colors[condition] = [0, 0, 0]

# pcd.colors = o3d.utility.Vector3dVector(colors)




# Transform the points back
T_inv = np.linalg.inv(T)

pcd = pcd.transform(T_inv)

# visualize point cloud
o3d.visualization.draw_geometries([pcd])