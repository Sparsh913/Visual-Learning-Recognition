import numpy as np
import cv2
import open3d as o3d

# insert coordinate frame in pcd
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
# pcd = pcd + mesh_frame

gt_ply = "ur5e_gt.ply"

# shift 0.1 in z

gt_pcd = o3d.io.read_point_cloud(gt_ply)

# Insert a plane at z = 0.16 in gt_pcd
planes = []

plane = o3d.geometry.TriangleMesh.create_box(width=0.25, height=0.25, depth=0.001)
plane.translate([0-0.125, 0-0.125, 0.09752])
plane.paint_uniform_color([1, 0, 0])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.001 , height=0.25, depth=0.25)
plane.translate([-0.07282, 0 -0.125, .1625-0.125])
plane.paint_uniform_color([0, 1, 1])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.001 , height=0.25, depth=0.25)
plane.translate([-0.06109, 0 -0.125, .5873-0.125])
plane.paint_uniform_color([0, 0, 1])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.001 , height=0.25, depth=0.25)
plane.translate([-0.05988, 0.39225 -0.125, .5873-0.125])
plane.paint_uniform_color([0, 0, 1])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.25 , height=0.001, depth=0.25)
plane.translate([-0.1333 -0.125, 0.05335+0.39225, .5873-0.125])
plane.paint_uniform_color([0, 0, 1])
planes.append(plane)

plane = o3d.geometry.TriangleMesh.create_box(width=0.25 , height=0.25, depth=0.001)
plane.translate([-0.1333 -0.125, 0.04655+0.39225+0.05335-0.125 , .5873 -0.05328])
plane.paint_uniform_color([0, 0, 1])
planes.append(plane)

# plane = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.001)
# plane.translate([-0.25, -0.25, 0.0295])
# plane.paint_uniform_color([0.1, 0.1, 0.7])

# Mesh coordinate frame
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

o3d.visualization.draw_geometries([mesh_frame, gt_pcd]+planes)
# o3d.visualization.draw_geometries([gt_pcd, pcd, mesh_frame])