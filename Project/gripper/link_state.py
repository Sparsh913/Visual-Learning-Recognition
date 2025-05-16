import pybullet as p
import time
import pybullet_data
import numpy as np
from urdf_models import models_data

p.connect(p.GUI)
pandaUid = p.loadURDF("pybullet-playground-20240518T073454Z-001/pybullet-playground/urdf/robotiq_2f_85.urdf", [0, 0, 0])
p.resetBasePositionAndOrientation(pandaUid, [0, 0, 0.03], [0, 0, 0, 1])

# get link state
link_states = p.getLinkState(pandaUid, 9)
print("Link states", link_states)

# total number of links
num_links = p.getNumJoints(pandaUid)
print("Number of links:", num_links) # 11

# color all the separate links of the gripper
# for i in range(num_links):
#     p.changeVisualShape(pandaUid, i, rgbaColor=[np.random.rand(), np.random.rand(), np.random.rand(), 1])
    
# color the link 9 to red
p.changeVisualShape(pandaUid, 9, rgbaColor=[1, 0, 0, 1])

# change the position of the link 9 slightly
p.resetBasePositionAndOrientation(pandaUid, [0, 0, 0.03], [0, 0, 0, 1])
p.resetJointState(pandaUid, 9, 0.5)

# from the initial position of link 9 to the final position, get the 4x4 transformation matrix for link 9
link_states_new = p.getLinkState(pandaUid, 9)
print("Link states", link_states_new)

# get the transformation matrix from link_states to link_states_new
link_transform = p.getMatrixFromQuaternion(link_states[1])
link_transform = np.array(link_transform).reshape(3, 3) # 3 x 3
link_transform = np.vstack([link_transform, [0, 0, 0]]) # 4 x 3
link_transform = np.hstack([link_transform, [[link_states[0][0]], [link_states[0][1]], [link_states[0][2]], [1]]]) # 4 x 4
print("Link transform:", link_transform)

# get the transformation matrix from link_states to link_states_new
link_transform_new = p.getMatrixFromQuaternion(link_states_new[1])
link_transform_new = np.array(link_transform_new).reshape(3, 3) # 3 x 3
link_transform_new = np.vstack([link_transform_new, [0, 0, 0]]) # 4 x 3
link_transform_new = np.hstack([link_transform_new, [[link_states_new[0][0]], [link_states_new[0][1]], [link_states_new[0][2]], [1]]]) # 4 x 4
print("Link transform:", link_transform_new)

# transformation between link_states and link_states_new
transformation = np.matmul(np.linalg.inv(link_transform), link_transform_new)
print("Transformation:", transformation)

# get the transformation matrix
# link_transform = p.getMatrixFromQuaternion(link_states_new[1])
# link_transform = np.array(link_transform).reshape(3, 3) # 3 x 3
# link_transform = np.vstack([link_transform, [0, 0, 0]]) # 4 x 3
# link_transform = np.hstack([link_transform, [[link_states_new[0][0]], [link_states_new[0][1]], [link_states[0][2]], [1]]]) # 4 x 4
# print("Link transform:", link_transform)

# move the gripper with all the constraints
# p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, targetPosition=0.5, force=5)

while True:
    p.stepSimulation()
    time.sleep(1./240.)