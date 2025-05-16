import pybullet as p
import time
import pybullet_data
import numpy as np
from urdf_models import models_data

p.connect(p.GUI)
# p.loadURDF(pybullet_data.getDataPath() + "/plane.urdf")

# p.loadURDF("T_object/urdf_T.urdf", [0, 0, 0.5])
pandaUid = p.loadURDF("gripper/2f85_closed.urdf", [0, 0, 0])
p.resetBasePositionAndOrientation(pandaUid, [0, 0, 0], [1, 0, 0, 1])

# joint_poses = [0, np.deg2rad(124) , -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]

# for i in range(len(joint_poses)):
#     p.resetJointState(pandaUid, i, joint_poses[i])

while True:
    p.stepSimulation()
    time.sleep(1./240.)