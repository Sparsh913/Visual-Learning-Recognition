# import numpy as np
# import cv2
# import open3d as o3d
# import torch 
# import pytorch3d
# from pytorch3d.loss import chamfer_distance

# ply_path = 'gripper/gt_gripper_open.ply'
# pcd = o3d.io.read_point_cloud(ply_path)

# gt_ply = 'gripper/gt_gripper_open.ply'
# gt_pcd = o3d.io.read_point_cloud(gt_ply)
# gt_link = 'gripper/transformed_link.ply'
# link_pcd = o3d.io.read_point_cloud(gt_link)

# # get the ratio of number of points in link_pcd to gt_pcd
# points_ratio = len(link_pcd.points)/len(gt_pcd.points)

# # no. of points in splat to optimize upon
# n_points = int(points_ratio * len(pcd.points))

# link_points_torch = torch.tensor(np.asarray(link_pcd.points)).float()
# pcd_points_torch = torch.tensor(np.asarray(pcd.points)).float()

# print('link_points_torch:', link_points_torch.shape)
# print('pcd_points_torch:', pcd_points_torch.shape)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# link_points_torch = link_points_torch.to(device)
# pcd_points_torch = pcd_points_torch.to(device)
# pcd_points_torch.requires_grad_(True)

# transformation = torch.tensor([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.71050543e-20],
#  [ 0.00000000e+00,  8.77582562e-01, -4.79425539e-01, -6.71543213e-03],
#  [ 0.00000000e+00,  4.79425539e-01,  8.77582562e-01,  2.42203907e-02],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]).float()
# transformation = transformation.to(device)

# def compute_chamfer_distance(indices):
#     selected_points = pcd_points_torch[indices]
#     transformed_points = torch.matmul(selected_points, transformation[:3, :3].T) + transformation[:3, 3]
#     loss = chamfer_distance(transformed_points.unsqueeze(0), link_points_torch.unsqueeze(0))[0]
#     return loss

# # task is to optimize the splat pcd points itself that if underwent the transformation, it should yield least chamfer distance with link_pcd
# # So the optimization parameters are the inidices of n_points from pcd_points_torch
# indices = torch.randperm(len(pcd_points_torch))[:n_points]
# indices = indices.to(device)
# optimizer = torch.optim.Adam([pcd_points_torch], lr=1e-3)
# temperature = 1.0
# cooling_rate = 0.99

# # define loss function as chamfer distance
# for i in range(1000000):
#     optimizer.zero_grad()
    
#     # update indices at each iteration
#     # indices = torch.randperm(len(pcd_points_torch))[:n_points].to(device)
#     # transformed_pcd = torch.matmul(pcd_points_torch[indices], transformation[:3,:3].T) + transformation[:3,3]
#     # loss = chamfer_distance(transformed_pcd.unsqueeze(0), link_points_torch.unsqueeze(0))[0]
#     # Randomly swap 5 indices
#     current_loss = compute_chamfer_distance(indices).to(device)
    
#     new_indices = indices.clone()
#     idx_to_replace = torch.randperm(n_points)[:20].to(device)
#     new_indices[idx_to_replace] = torch.randperm(len(pcd_points_torch))[:20].to(device)
    
#     new_loss = compute_chamfer_distance(new_indices).to(device)
#     if (new_loss < current_loss) or (torch.rand(1).item() < torch.exp((current_loss - new_loss) / temperature)):
#         indices = new_indices
#         current_loss = new_loss
    
#     # Cool down the temperature
#     temperature *= cooling_rate
    
#     current_loss.backward()
#     optimizer.step()
#     if i % 10 == 0:
#         print('Loss:', current_loss.item())
#         #save the transformed pcd
#         with torch.no_grad():
#             selected_points = pcd_points_torch[indices]
#             transformed_points = torch.matmul(selected_points, transformation[:3, :3].T) + transformation[:3, 3]
#         pcd.points = o3d.utility.Vector3dVector(pcd_points_torch.cpu().detach().numpy())
#         o3d.io.write_point_cloud('splat_link_trans.ply', pcd)
#         # save the indices
#         np.save('indices.npy', indices.cpu().detach().numpy())
        
        
        
        
        
        
        
        
import numpy as np
import open3d as o3d
import torch
from pytorch3d.loss import chamfer_distance

# Load point clouds
ply_path = 'gripper/gt_gripper_open.ply'
pcd = o3d.io.read_point_cloud(ply_path)

gt_ply = 'gripper/gt_gripper_open.ply'
gt_pcd = o3d.io.read_point_cloud(gt_ply)
gt_link = 'gripper/transformed_link.ply'
link_pcd = o3d.io.read_point_cloud(gt_link)

# Get the ratio of number of points in link_pcd to gt_pcd
points_ratio = len(link_pcd.points) / len(gt_pcd.points)
n_points = int(points_ratio * len(pcd.points))

# Convert point clouds to PyTorch tensors
link_points_torch = torch.tensor(np.asarray(link_pcd.points)).float()
pcd_points_torch = torch.tensor(np.asarray(pcd.points)).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
link_points_torch = link_points_torch.to(device)
pcd_points_torch = pcd_points_torch.to(device)
# pcd_points_torch.requires_grad_(True)

# Define the transformation matrix
transformation = torch.tensor([[1.0, 0.0, 0.0, -2.71050543e-20],
                               [0.0, 0.877582562, -0.479425539, -0.00671543213],
                               [0.0, 0.479425539, 0.877582562, 0.0242203907],
                               [0.0, 0.0, 0.0, 1.0]]).float()
transformation = transformation.to(device)

# Initialize index weights for soft selection
index_weights = torch.nn.Parameter(torch.randn(len(pcd_points_torch)))

# Function to compute Chamfer distance
def compute_chamfer_distance(selected_points):
    transformed_points = torch.matmul(selected_points, transformation[:3, :3].T) + transformation[:3, 3]
    loss = chamfer_distance(transformed_points.unsqueeze(0), link_points_torch.unsqueeze(0))[0]
    return loss

# Optimizer
optimizer = torch.optim.Adam([index_weights], lr=1e-2)

# Optimization loop
for i in range(100000):
    optimizer.zero_grad()
    
    # Softmax to get probabilities and sample points based on probabilities
    prob = torch.softmax(index_weights, dim=0)
    # print("index_weights:", index_weights.shape)
    # print(index_weights[:10])
    # print("prob:", prob.shape)
    # print(prob[:10])
    sampled_indices = torch.multinomial(prob, n_points, replacement=False)
    
    # check if everything is differentiable till now
    if not prob.requires_grad:
        raise RuntimeError("Probabilities do not require gradients")
    if not sampled_indices.requires_grad:
        raise RuntimeError("Sampled indices do not require gradients")
    
    # print("sampled_indices:", sampled_indices.shape)
    # print(sampled_indices[:10])
    
    selected_points = pcd_points_torch[sampled_indices]
    
    loss = compute_chamfer_distance(selected_points)
    
    # Check that loss requires gradients
    if not loss.requires_grad:
        raise RuntimeError("Loss does not require gradients")
    
    # Backpropagation and optimization step
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print('Loss:', loss.item())
        
        # Save the transformed point cloud
        with torch.no_grad():
            # transformed_points = torch.matmul(selected_points, transformation[:3, :3].T) + transformation[:3, 3]
            transformed_points = selected_points
            pcd.points = o3d.utility.Vector3dVector(transformed_points.cpu().detach().numpy())
            o3d.io.write_point_cloud('splat_link_trans.ply', pcd)
            
            # Save the indices
            np.save('indices.npy', sampled_indices.cpu().detach().numpy())
            
            # also save pcd_points_torch as npy
            np.save('pcd_points_torch.npy', pcd_points_torch.cpu().detach().numpy())
