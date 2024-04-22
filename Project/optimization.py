import numpy as np
import cv2
import open3d as o3d
import torch 

ply_path = "/home/uas-laptop/Visual-Learning-Recognition/Project/raw_transform.ply"
pcd = o3d.io.read_point_cloud(ply_path)


gt_ply = "/home/uas-laptop/Visual-Learning-Recognition/Project/gt_pcd.ply"
gt_pcd = o3d.io.read_point_cloud(gt_ply)

gt_points_torch = torch.tensor(np.asarray(gt_pcd.points)).float()
pcd_points_torch = torch.tensor(np.asarray(pcd.points)).float()

print('gt_points_torch:', gt_points_torch.shape)
print('pcd_points_torch:', pcd_points_torch.shape)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gt_points_torch = gt_points_torch.to(device)
pcd_points_torch = pcd_points_torch.to(device)

#define optimization variables R, T, scale
euler_angles = torch.randn(3, requires_grad=True, device=device)

T = torch.zeros(3, requires_grad=True, device=device)
scale = torch.tensor(1.0, requires_grad=True, device=device)


# define optimizer
parameters = [
        {'params': [euler_angles], 'lr': 0.05, "name": "opacities"},
        {'params': [T], 'lr': 0.01, "name": "scales"},
        {'params': [scale], 'lr': 0.1, "name": "colours"},
    ]
    
optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)
# optimizer = torch.optim.Adam([euler_angles, T, scale], lr=1e-3)

# define loss function as chamfer distance
import pytorch3d
from pytorch3d.loss import chamfer_distance

for i in range(1000):
    optimizer.zero_grad()
    rotation_matrix = pytorch3d.transforms.euler_angles_to_matrix(euler_angles, convention="XYZ")
    transformed_pcd = torch.matmul( torch.sigmoid(scale)*pcd_points_torch, rotation_matrix.T) + T
    loss = chamfer_distance(transformed_pcd.unsqueeze(0), gt_points_torch.unsqueeze(0))[0]
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print('Loss:', loss.item())
        #save the transformed pcd
        pcd.points = o3d.utility.Vector3dVector(transformed_pcd.cpu().detach().numpy())
        o3d.io.write_point_cloud('/home/uas-laptop/Visual-Learning-Recognition/Project/transformed_pcd.ply', pcd)
        
        #save angles, T, scale
        np.save('/home/uas-laptop/Visual-Learning-Recognition/Project/angles.npy', euler_angles.cpu().detach().numpy())
        np.save('/home/uas-laptop/Visual-Learning-Recognition/Project/T.npy', T.cpu().detach().numpy())
        np.save('/home/uas-laptop/Visual-Learning-Recognition/Project/scale.npy', scale.cpu().detach().numpy())


