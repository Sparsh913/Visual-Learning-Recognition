import numpy as np
import cv2
import open3d as o3d
import torch

ply_path = "/home/uas-laptop/Visual-Learning-Recognition/Project/pcd2.ply"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pcd = o3d.io.read_point_cloud(ply_path)

# Apply a transform to the pcd
# T = torch.tensor([-0.122496815889771, -0.190901099205025, 0.246965895011582, -0.329810908630861,
#                  -0.302414055868664, 0.006435588695706, -0.145030531875939, 0.295867956019945,
#                  0.077896919593464, -0.275448483064183, -0.174602357521242, 0.773293581992212,
#                  0.0, 0.0, 0.0, 1.0])

# T = -0.272776812315 -0.074060574174 0.179010376334 -0.366493761539
# -0.193020626903 0.130256026983 -0.240235880017 0.399390488863
# -0.016514312476 -0.299140989780 -0.148925781250 0.708794176579
# 0.000000000000 0.000000000000 0.000000000000 1.000000000000

T = np.array([-0.272776812315, -0.074060574174, 0.179010376334, -0.366493761539,
                -0.193020626903, 0.130256026983, -0.240235880017, 0.399390488863,
                -0.016514312476, -0.299140989780, -0.148925781250, 0.708794176579,
                0.0, 0.0, 0.0, 1.0])

T = T.reshape(4, 4)
print('Transformation:', T)

# pcd_points_torch = torch.tensor(np.asarray(pcd.points)).float().to(device)

pcd = pcd.transform(T)

#save pcd
# o3d.io.write_point_cloud("/home/uas-laptop/Visual-Learning-Recognition/Project/raw_transform.ply", pcd)

# insert coordinate frame in pcd
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

gt_ply = "/home/uas-laptop/Visual-Learning-Recognition/Project/gt_pcd.ply"

gt_pcd = o3d.io.read_point_cloud(gt_ply)

# ICP on pcd and gt_pcd
# icp = o3d.pipelines.registration.registration_icp(pcd, gt_pcd, 0.003, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint())

# transformation_matrix = icp.transformation
# print('ICP Transformation:', transformation_matrix)

# pcd = pcd.transform(transformation_matrix)

# visualize the pcd
o3d.visualization.draw_geometries([pcd, mesh_frame])