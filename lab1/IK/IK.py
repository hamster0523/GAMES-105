from BVH_Reader.bvh_read import Bone_Tree
from torch import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_joint_offset_along_path(bone_tree: Bone_Tree, path_name: list):
    path_offset_along_path = []

    """
    path_name is like : ['lToeJoint_end', 'lToeJoint', 'lAnkle', 'lKnee', 'lHip', 'RootJoint']
              is from end to root joint
    """
    all_joint_name_index = [
        bone_tree.joint_name.index(joint_name) for joint_name in path_name
    ]

    path_offset_along_path = [
        bone_tree.joint_offset[joint_name_index]
        for joint_name_index in all_joint_name_index
    ]
    # so the offset is from end to root joint
    path_offset_along_path.reverse()
    return path_offset_along_path


def forward_kinematics(
    root_joint_pos: list, all_joint_offset: list, rotation_param_matrix: torch.Tensor
) -> list:
    # do forward kinematics from a root joint to a end joint
    # return -> end joint position

    # first calculate all needed rotations
    Q = []
    Q.append(R.from_euler("XYZ", [0, 0, 0], degrees=True).as_matrix())
    for i in range(len(rotation_param_matrix)):
        Q.append(R.from_quat(rotation_param_matrix[i].tolist()).as_matrix() @ Q[-1])
    # exclude the identity matrix
    Q = Q[1:]

    # second calculate the end joint position
    """
    if the joint nums is 5, the final position is calculated by :
    X = p0 + R0l0 + R0R1l1 + R0R1R2l2 + R0R1R2R3l3 + R0R1R2R3R4l4
    Here p0 is root_joint_pos
         R0 - R0R1R2R3R4 is calculated in Q
         l0 - l4 - x is all offset from root joint to end joint
    """
    final_position = np.array(root_joint_pos)
    for i in range(len(Q)):
        final_position += np.array(Q[i] @ all_joint_offset[i])
    return final_position.tolist()


def inverse_kinematics(
    bone_tree: Bone_Tree,
    path_info: list,
    joint_positions: list,
    joint_orientations: list,
    target_pos: list,
) -> list:
    """
    input : path_info : list -> from AJoint To BJoint path info
            joint_positions : list -> no IK joint positions
            joint_orientations : list -> no IK joint orientations
            target_pos : list -> target position
    output :
            joint_positions : list -> IK joint positions
            joint_orientations : list -> IK joint orientations
    """
    # in gradient descent -> update the rotate parameter of each joint
    # then calculate the new joint position
    # the primart position for root joint is all the same -> not change
    # every iterate -> do forward kinamatics to calculate the end joint position

    """
    all needed info :
    thresold : 0.001
    learning_rate : 0.01
    """
    thresold = 0.001
    learning_rate = 0.01
    epochs = 1000

    try:
        path_index = path_info[0]
        path_name = path_info[1]
    except:
        if len(path_index) == 2:
            raise ValueError("more path here not supported")
    # this only process one path case

    primary_positions, primary_rotations = (
        bone_tree.get_bone_tree_primary_positions_rotations_from_AJoint_To_BJoint(
            path_info=path_info
        )
    )

    # offset from root joint to end joint
    root_to_end_offset = get_joint_offset_along_path(bone_tree, path_name)

    # reverse -> from RootJoint to end joint
    primary_positions.reverse()
    primary_rotations.reverse()

    # type : np.ndarray
    primary_positions_root = primary_positions[0]

    # create a rotation parameter matrix
    rotation_matrix = torch.from_numpy(
        np.array([x.tolist() for x in primary_rotations])
    ).requires_grad_(True)
    """
    torch array like :
    tensor([[ 0.0065, -0.0204, -0.0107,  0.9997],
        [ 0.0065, -0.0204, -0.0107,  0.9997],
        [ 0.0755, -0.0239,  0.0039,  0.9969],
        [ 0.2270,  0.0226, -0.0644,  0.9715],
        [ 0.1131,  0.0359, -0.0237,  0.9927],
        [ 0.0170,  0.0664, -0.0058,  0.9976]], dtype=torch.float64)
    """
    final_position = forward_kinematics(
        root_joint_pos=primary_positions_root,
        all_joint_offset=root_to_end_offset,
        rotation_param_matrix=rotation_matrix,
    )
    x = torch.from_numpy(np.array(final_position - target_pos)).requires_grad_(True)
    y = 0.5 * torch.dot(x, x)

    optimizer = torch.optim.SGD([rotation_matrix], lr=learning_rate)
    for epoch in epochs:
        optimizer.zero_grad()
        final_position = forward_kinematics(
            root_joint_pos=primary_positions_root,
            all_joint_offset=root_to_end_offset,
            rotation_param_matrix=rotation_matrix,
        )
        x = torch.tensor(final_position, dtype=torch.float64) - torch.tensor(
            target_pos, dtype=torch.float64
        )
        y = 0.5 * torch.dot(x, x)
        y.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"epoch {epoch} loss {y.item()}")

        if y.item() < thresold:
            break

    final_joint_positions = forward_kinematics(
        root_joint_pos=primary_positions_root,
        all_joint_offset=root_to_end_offset,
        rotation_param_matrix=rotation_matrix,
    )
    final_orientation = rotation_matrix.detach().numpy()

    return final_joint_positions, final_orientation


def inverse_kinematic_test(meta_data, joint_positions, joint_orientations, target_pos) -> list:
   joint_parent = meta_data.joint_parent
   joint_offset = [
       meta_data.joint_inital_positions[i] - meta_data.joint_inital_positions[joint_parent[i]]
       for i in range(len(joint_positions))
   ]
   joint_offset = np.array([0, 0, 0])
   joint_ik_path, _, _, _ = meta_data.get_path_from_root_to_end()
   # evety joint rotations
   local_rotation = [
       R.from_quat(joint_orientations[joint_parent[i]]).inv() @ R.from_quat(joint_orientations[i])
       for i in range(len(joint_orientations))
   ]
   local_rotation[0] = R.from_quat(joint_orientations[0])
   #gradient descent
   joint_offset_tensor = [torch.tensor(data) for data in joint_offset]
   joint_positions_tensor = [torch.tensor(data) for data in joint_positions]
   joint_orientations_tensor = [
       torch.tensor(R.from_quat(data).as_matrix(), requires_grad = True)
       for data in joint_orientations
   ]
   local_rotation_tensor = [
       torch.tensor(data.as_matrix(), requires_grad = True)
       for data in local_rotation
   ]
   target_pos_tensor = torch.tensor(target_pos, requires_grad = True)
   
   eppchs = 300
   learning_rate = 0.01
   for _ in range(eppchs):
       for j in len(joint_ik_path):
           # Update every joint position
           a = chain_current = joint_ik_path[j]
           b = chain_parent  = joint_ik_path[j - 1]
           if j == 0:
               local_rotation_tensor[a] = local_rotation_tensor[a]
               joint_positions_tensor[a] = joint_positions_tensor[a]
           elif b == joint_parent[a]:
               # forward kinematics
               local_rotation_tensor[a] = joint_orientations_tensor[b] @ \
                   local_rotation_tensor[a]
               joint_positions_tensor[a] = joint_positions_tensor[b] + \
                   joint_offset_tensor[a] @ torch.transpose(joint_orientations_tensor[b], 0, 1)
           else:
               # inverse kinematics
               