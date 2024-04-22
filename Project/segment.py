import open3d as o3d
import numpy as np
import torch

#load point cloud
pcd = o3d.io.read_point_cloud("colored_gt.ply")


#cooridnate frame
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
#visualize point cloud
# o3d.visualization.draw_geometries([pcd, mesh_frame])
color = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
#round colors to 2 decimal places
colors = np.round(colors, 0)
#find unique colors in the point cloud
unique_colors = np.unique(colors, axis=0)
print('unique_colors:', unique_colors)
print('unique_colors:', unique_colors.shape)
#find index of points with unique colors
idxs = []
for color in unique_colors:
    idx = np.where(np.all(colors == color, axis=1))[0]
    idxs.append(idx)
    print('idx:', idx.shape)
segmented_points = [points[idx][::10] for idx in idxs]

#convert to torch tensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
segmented_points = [torch.tensor(segmented_points).float().to(device) for segmented_points in segmented_points]

#find distance of between segmented points [0] and [1]

dist = torch.cdist(segmented_points[0], segmented_points[1])
print('dist:', dist.shape)
#find 3 points with minimum distance
min_idx = torch.argsort(dist, dim=1)  
print('min_idx:', min_idx.shape)


print('min_points:', min_points)
