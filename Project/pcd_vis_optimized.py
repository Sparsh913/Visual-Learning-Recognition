import numpy as np
import cv2
import open3d as o3d

# ply_path = "/home/uas-laptop/Visual-Learning-Recognition/Project/pcd2.ply"
ply_path = "/home/uas-laptop/Visual-Learning-Recognition/Project/oriented_pcd.ply"
pcd = o3d.io.read_point_cloud(ply_path)


outlier_nb = 80
outlier_std = 0.9
# pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=outlier_nb, std_ratio=outlier_std)


# remove points with x > 0.5
pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,0] < 0.5)[0])

# insert coordinate frame in pcd
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
# pcd = pcd + mesh_frame

gt_ply = "/home/uas-laptop/Visual-Learning-Recognition/Project/gt_pcd.ply"

gt_pcd = o3d.io.read_point_cloud(gt_ply)

o3d.visualization.draw_geometries([pcd, mesh_frame])