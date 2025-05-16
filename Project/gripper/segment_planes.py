import numpy as np
import cv2
import open3d as o3d

# ply_path = "/home/uas-laptop/Visual-Learning-Recognition/Project/pcd1.ply"
# ply_path = "/home/uas-laptop/Visual-Learning-Recognition/Project/transformed_pcd.ply"
ply_path = "gripper/2f85_open_gt.ply"
pcd = o3d.io.read_point_cloud(ply_path)


# insert coordinate frame in pcd
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
# pcd = pcd + mesh_frame

gt_ply = "gripper/2f85_open_gt.ply"

gt_pcd = o3d.io.read_point_cloud(gt_ply)

# Insert a plane at z = 0.16 in gt_pcd
planes = []

plane = o3d.geometry.TriangleMesh.create_box(width=0.15, height=0.15, depth=0.001)
plane.translate([0-0.075, 0-0.075, 0.055503])
plane.paint_uniform_color([1, 0, 0])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.001 , height=0.15, depth=0.15)
plane.translate([0.0123, 0 -0.075, 0.061178-0.075])
plane.paint_uniform_color([0, 1, 1])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.001 , height=0.15, depth=0.15)
plane.translate([-0.0123, 0 -0.075, 0.061178-0.075])
plane.paint_uniform_color([0, 1, 1])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.15, height=0.15, depth=0.001)
plane.translate([0-0.075, 0-0.075, 0.109517])
plane.paint_uniform_color([1, 0, 0])
planes.append(plane)

# create an inclined plane at an angle of 49.445 degrees with the y axis and 0.1 distance from the origin


# plane = o3d.geometry.TriangleMesh.create_box(width=0.001 , height=0.25, depth=0.25)
# plane.translate([-0.0743, 0 -0.125, .5142-0.125])
# plane.paint_uniform_color([0, 0, 1])
# planes.append(plane)

# plane = o3d.geometry.TriangleMesh.create_box(width=0.001 , height=0.25, depth=0.25)
# plane.translate([-0.0743 +0.0174 -0.00785, 0.39225 -0.125, .5142-0.125])
# plane.paint_uniform_color([0, 0, 1])
# planes.append(plane)

# plane = o3d.geometry.TriangleMesh.create_box(width=0.25 , height=0.001, depth=0.25)
# plane.translate([-0.0743 +0.0174-0.0531 -0.125, 0.04165+0.39225+0.00785, .5142-0.125])
# plane.paint_uniform_color([0, 0, 1])
# planes.append(plane)

# plane = o3d.geometry.TriangleMesh.create_box(width=0.25 , height=0.25, depth=0.001)
# plane.translate([-0.0743 +0.0174-0.0531 -0.125, 0.04165+0.39225+0.0531-0.125 , .5142 -0.04165-0.00785])
# plane.paint_uniform_color([0, 0, 1])
# planes.append(plane)

# plane = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.001)
# plane.translate([-0.25, -0.25, 0.0295])
# plane.paint_uniform_color([0.1, 0.1, 0.7])

# Mesh coordinate frame
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

o3d.visualization.draw_geometries([mesh_frame, gt_pcd]+planes)
# o3d.visualization.draw_geometries([gt_pcd, pcd, mesh_frame])