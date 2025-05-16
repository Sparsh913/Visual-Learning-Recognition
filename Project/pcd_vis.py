import numpy as np
import cv2
import open3d as o3d

# ply_path = "/home/uas-laptop/Visual-Learning-Recognition/Project/pcd1.ply"
# ply_path = "/home/uas-laptop/Visual-Learning-Recognition/Project/transformed_pcd.ply"
ply_path = "power_drill/drill_pcd_1.ply"
pcd = o3d.io.read_point_cloud(ply_path)


outlier_nb = 80
outlier_std = 0.3
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=outlier_nb, std_ratio=outlier_std)


# remove points with x > 0.5
pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,0] > -3)[0])
pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,0] < 2)[0])

# remove points with z < 0
pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,2] > 0)[0])

# remove points with y >4
pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,1] < 4)[0])

# remove points with y < -4
pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,1] > 0.2)[0])

# scale down the pcd
# pcd.scale(0.07, center=(0, 0, 0))

#save pcd
# o3d.io.write_point_cloud("/home/uas-laptop/Visual-Learning-Recognition/Project/pcd2.ply", pcd)
o3d.io.write_point_cloud("power_drill/drill_pcd_2.ply", pcd)


# insert coordinate frame in pcd
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
# pcd = pcd + mesh_frame

gt_ply = "power_drill/drill_point_cloud.ply"

# shift 0.1 in z

gt_pcd = o3d.io.read_point_cloud(gt_ply)
#rotate 90 degrees around x-axis
# R = gt_pcd.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
# gt_pcd.rotate(R, center=(0, 0, 0))
# R = gt_pcd.get_rotation_matrix_from_xyz((0, -np.pi/2, 0))
# gt_pcd.rotate(R, center=(0, 0, 0))
    

# gt_pcd = gt_pcd.translate([0, 0, -0.1])

# save gt_pcd
# o3d.io.write_point_cloud("/home/uas-laptop/Visual-Learning-Recognition/Project/gt_pcd.ply", gt_pcd)

# Remove points with z > 0.223
# gt_pcd = gt_pcd.select_by_index(np.where(np.asarray(gt_pcd.points)[:,2] < 0.16)[0])

# Insert a plane at z = 0.16 in gt_pcd
planes = []

plane = o3d.geometry.TriangleMesh.create_box(width=0.25, height=0.25, depth=0.001)
plane.translate([0-0.125, 0-0.125, 0.0213])
plane.paint_uniform_color([1, 0, 0])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.001 , height=0.25, depth=0.25)
plane.translate([-0.0663-0.00785, 0 -0.125, .0892-0.125])
plane.paint_uniform_color([0, 1, 1])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.001 , height=0.25, depth=0.25)
plane.translate([-0.0743, 0 -0.125, .5142-0.125])
plane.paint_uniform_color([0, 0, 1])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.001 , height=0.25, depth=0.25)
plane.translate([-0.0743 +0.0174 -0.00785, 0.39225 -0.125, .5142-0.125])
plane.paint_uniform_color([0, 0, 1])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.25 , height=0.001, depth=0.25)
plane.translate([-0.0743 +0.0174-0.0531 -0.125, 0.04165+0.39225+0.00785, .5142-0.125])
plane.paint_uniform_color([0, 0, 1])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.25 , height=0.25, depth=0.001)
plane.translate([-0.0743 +0.0174-0.0531 -0.125, 0.04165+0.39225+0.0531-0.125 , .5142 -0.04165-0.00785])
plane.paint_uniform_color([0, 0, 1])
planes.append(plane)

# plane = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.001)
# plane.translate([-0.25, -0.25, 0.0295])
# plane.paint_uniform_color([0.1, 0.1, 0.7])

# Mesh coordinate frame
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

o3d.visualization.draw_geometries([mesh_frame, gt_pcd]+planes)
# o3d.visualization.draw_geometries([gt_pcd, pcd, mesh_frame])