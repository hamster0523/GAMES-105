import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


def test_root_to_end_path(meta_data, joint_positions, joint_orientations, target_pose):
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    return path, path_name, path1, path2


def part1_inverse_kinematics(
    meta_data, joint_positions, joint_orientations, target_pose
):
    """
    完成函数，计算逆运动学
    输入:
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        ----
        一些固定信息，其中joint_initial_position是T-pose下的关节位置，可以用于计算关节相互的offset
        root_joint是固定节点的索引，并不是RootJoint节点
        self.joint_name = joint_name
        self.joint_parent = joint_parent
        self.joint_initial_position = joint_initial_position
        self.root_joint = root_joint
        self.end_joint = end_joint
        ----
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    # now the data is character_model.npy
    # not the raw data of in data file
    # get the info about the path from root to end
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    # we need the path is from RootJoint to EndJoint -> so we need to sort the path
    # ------ depend on the path order, we assume the path is from RootJoint to A EndJoint
    # a joint positions = parent_joint_position + offset * parent_orientation
    joint_offset = [
        meta_data.joint_initial_position[i]
        - meta_data.joint_initial_position[meta_data.joint_parent[i]]
        for i in range(len(joint_positions))
    ]
    joint_offset[0] = np.array([0, 0, 0])
    # print("joint_offset: ", joint_offset)
    # current joint_orientation = parent_joint_orientation * current_joint_rotation
    # current_joint_rotation = paretn_joint_orientation.inv() * current_joint_orientation
    joint_rotation = [
        R.from_quat(joint_orientations[meta_data.joint_parent[i]]).inv()
        * R.from_quat(joint_orientations[i])
        for i in range(len(joint_orientations))
    ]
    joint_rotation[0] = R.from_euler("XYZ", [0, 0, 0], degrees=True)
    # print("joint_rotation: ", joint_rotation)

    # gradient descent
    joint_offset_tensor = [torch.tensor(data) for data in joint_offset]
    # matrix tensor data
    joint_rotation_tensor = [
        torch.tensor(data.as_matrix(), requires_grad=True) for data in joint_rotation
    ]
    joint_positions_tensor = [torch.tensor(data) for data in joint_positions]
    # matrix tensor data
    joint_orientations_tensor = [
        torch.tensor(R.from_quat(data).as_matrix(), requires_grad=True)
        for data in joint_orientations
    ]
    target_pos_tensor = torch.tensor(target_pose)

    # just need do rootjoint to endjoint forward
    epoch = 300
    alpha = 0.01
    for _ in range(epoch):
        for j in range(len(path)):
            joint_current = path[j]
            if joint_current == 0:
                # this is root joint
                joint_positions_tensor[joint_current] = joint_positions_tensor[
                    joint_current
                ]
                joint_orientations_tensor[joint_current] = joint_orientations_tensor[
                    joint_current
                ]
            else:
                # print("joint_current: ", joint_current, " joint_parent: ", meta_data.joint_parent[joint_current])
                joint_parent = meta_data.joint_parent[joint_current]
                # not root joint -> do forward kinematics -> position_parent + orientation_parent @ this_joint_offset
                joint_positions_tensor[joint_current] = (
                    joint_positions_tensor[joint_parent]
                    + joint_orientations_tensor[joint_parent]
                    @ joint_offset_tensor[joint_current]
                )
                joint_orientations_tensor[joint_current] = (
                    joint_orientations_tensor[joint_parent]
                    @ joint_rotation_tensor[joint_current]
                )
                # torch.tensor(
                #     (R.from_quat(joint_orientations_tensor[joint_parent].numpy()) * \
                #         R.from_quat(joint_rotation_tensor[joint_current].numpy())).as_quat()
                # )

        optimizer_target = torch.norm(
            joint_positions_tensor[path[-1]] - target_pos_tensor
        )
        optimizer_target.backward()
        for num in path:
            if joint_rotation_tensor[num].grad is not None:
                temp = (
                    joint_rotation_tensor[num] - alpha * joint_rotation_tensor[num].grad
                )
                joint_rotation_tensor[num] = torch.tensor(temp, requires_grad=True)

    # # update complete -> do once all joint forward kinematic
    # for j in range(len(path)):
    #     joint_current = path[j]
    #     if joint_current == 0 :
    #         # this is root joint
    #         joint_positions_tensor[joint_current] = joint_positions_tensor[joint_current]
    #         joint_orientations_tensor[joint_current] = joint_orientations_tensor[joint_current]
    #     else:
    #         # print("joint_current: ", joint_current, " joint_parent: ", meta_data.joint_parent[joint_current])
    #         joint_parent  = meta_data.joint_parent[joint_current]
    #         # not root joint -> do forward kinematics -> position_parent + orientation_parent @ this_joint_offset
    #         joint_positions_tensor[joint_current] = joint_positions_tensor[joint_parent] + \
    #                 joint_orientations_tensor[joint_parent] @ joint_offset_tensor[joint_current]
    #         joint_orientations_tensor[joint_current] = joint_orientations_tensor[joint_parent] @ joint_rotation_tensor[joint_current]
    #         # torch.tensor(
    #         #     (R.from_quat(joint_orientations_tensor[joint_parent].numpy()) * \
    #         #         R.from_quat(joint_rotation_tensor[joint_current].numpy())).as_quat()
    #         # )

    # do once forward and translate to numpy
    for j in range(len(path)):
        joint_current = path[j]
        joint_parent = meta_data.joint_parent[joint_current]
        if joint_current == 0:
            joint_positions[joint_current] = (
                joint_positions_tensor[joint_current].detach().numpy()
            )
            joint_orientations[joint_current] = R.from_matrix(
                joint_orientations_tensor[joint_current].detach().numpy()
            ).as_quat()
        else:
            joint_positions[joint_current] = (
                joint_positions_tensor[joint_parent].detach().numpy()
                + joint_orientations_tensor[joint_parent].detach().numpy()
                @ joint_offset[joint_current]
            )
            joint_orientations[joint_current] = (
                R.from_matrix(joint_orientations_tensor[joint_parent].detach().numpy())
                * R.from_matrix(joint_rotation_tensor[joint_current].detach().numpy())
            ).as_quat()

    # just update none path joint
    path_set = set(path)
    for i in range(len(joint_positions)):
        if i in path_set:
            continue
        else:
            joint_positions[i] = (
                joint_positions[meta_data.joint_parent[i]]
                + np.asmatrix(
                    R.from_quat(
                        joint_orientations[meta_data.joint_parent[i]]
                    ).as_matrix()
                )
                @ joint_offset[i]
            )
            joint_orientations[i] = R.from_matrix(
                R.from_quat(joint_orientations[meta_data.joint_parent[i]]).as_matrix()
                @ R.from_quat(joint_rotation[i]).as_matrix()
            ).as_quat()

    # # translate back to numpy
    # joint_positions = [
    #     joint_positions_tensor[i].detach().numpy()
    #     for i in range(len(joint_positions))
    # ]
    # joint_orientations = [
    #     R.from_matrix(joint_orientations_tensor[i].detach().numpy()).as_quat()
    #     for i in range(len(joint_orientations))
    # ]

    return np.array(joint_positions), np.array(joint_orientations)

def inverse_kinematics_3(meta_data, joint_positions, joint_orientations, target_pose):
    joint_parents = meta_data.joint_parent
    joint_offsets = [
        meta_data.joint_initial_position[i]
        - meta_data.joint_initial_position[joint_parents[i]]
        for i in range(len(meta_data.joint_initial_position))
    ]
    joint_offsets[0] = np.array([0, 0, 0])
    local_rotations = [
        R.from_quat(joint_orientations[joint_parents[i]]).inv()
        * R.from_quat(joint_orientations[i])
        for i in range(len(joint_orientations))
    ]
    local_rotations[0] = R.from_quat(joint_orientations[0])

    joint_ik_path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    joint_offsets_t = [torch.tensor(data) for data in joint_offsets]
    joint_positions_t = [torch.tensor(data) for data in joint_positions]
    joint_orientations_t = [
        torch.tensor(R.from_quat(data).as_matrix(), requires_grad=True)
        for data in joint_orientations
    ]
    local_rotations_t = [
        torch.tensor(data.as_matrix(), requires_grad=True) for data in local_rotations
    ]
    target_pose_t = torch.tensor(target_pose)

    epochs = 1000
    alpha = 0.001

    for _ in range(epochs):
        for j in range(len(joint_ik_path)):
            chain_current = joint_ik_path[j]
            chain_parent = joint_parents[chain_current]
            if j == 0:
                local_rotations_t[chain_current] = local_rotations_t[chain_current]
                joint_positions_t[chain_current] = joint_positions_t[chain_current]
            else:
                joint_orientations_t[chain_current] = (
                    joint_orientations_t[chain_parent]
                    @ local_rotations_t[chain_current]
                )
                # joint_positions_t[chain_current] = joint_positions_t[chain_parent] + joint_orientations_t[chain_parent] @ joint_offsets_t[chain_current]
                joint_positions_t[chain_current] = joint_positions_t[
                    chain_parent
                ] + joint_offsets_t[chain_current] @ torch.transpose(
                    joint_orientations_t[chain_parent], 0, 1
                )

        # we just update the local rotations parameters
        optimizer_target = torch.norm(
            joint_positions_t[joint_ik_path[-1]] - target_pose_t
        )
        optimizer_target.backward()
        for index in joint_ik_path:
            if local_rotations_t[index].grad is not None:
                temp = local_rotations_t[index] - alpha * local_rotations_t[index].grad
                local_rotations_t[index] = torch.tensor(temp, requires_grad=True)

    # all we use tensor is just local_rotations_t -> attention : except that do not use tensor
    for j in range(len(joint_ik_path)):
        chain_current = joint_ik_path[j]
        chain_parent = joint_parents[chain_parent]
        if j == 0:
            local_rotations[chain_current] = R.from_matrix(
                local_rotations_t[chain_current].detach().numpy()
            )
            joint_positions[chain_current] = joint_positions[chain_current]
        else:
            joint_orientations[chain_current] = (
                R.from_quat(joint_orientations[chain_parent])
                * R.from_matrix(local_rotations_t[chain_current].detach().numpy())
            ).as_quat()
            # joint_positions[chain_current] = joint_positions[chain_parent] + \
            #     R.from_quat(joint_orientations[chain_parent]).as_matrix() @ joint_offsets[chain_current]
            joint_positions[chain_current] = (
                joint_positions[chain_parent]
                + joint_offsets[chain_current]
                * np.asmatrix(
                    R.from_quat(joint_orientations[chain_parent]).as_matrix()
                ).transpose()
            )

    ik_path_set = set(joint_ik_path)
    for i in range(len(joint_positions)):
        if i in ik_path_set:
            joint_orientations[i] = R.from_matrix(
                joint_orientations_t[i].detach().numpy()
            ).as_quat()
        else:
            joint_orientations[i] = (
                R.from_quat(joint_orientations[joint_parents[i]]) * local_rotations[i]
            ).as_quat()
            # joint_positions[i] = joint_positions[joint_parents[i]] + \
            #     R.from_quat(joint_orientations[joint_parents[i]]).as_matrix() @ joint_offsets[i]
            joint_positions[i] = (
                joint_positions[joint_parents[i]]
                + joint_offsets[i]
                * np.asmatrix(
                    R.from_quat(joint_orientations[joint_parents[i]]).as_matrix()
                ).transpose()
            )

    return joint_positions, joint_orientations

def inverse_kinematics_final(meta_data, joint_positions, joint_orientations, target_pose):
    joint_parents = meta_data.joint_parent
    joint_offsets = [
        meta_data.joint_initial_position[i] - meta_data.joint_initial_position[joint_parents[i]]
        for i in range(len(meta_data.joint_initial_position))
    ]
    joint_offsets[0] = np.array([0, 0, 0])
    local_rotations = [
        R.from_quat(joint_orientations[joint_parents[i]]).inv() * \
            R.from_quat(joint_orientations[i])
        for i in range(len(joint_orientations))
    ]
    local_rotations[0] = R.from_quat(joint_orientations[0])
    
    joint_ik_path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    
    joint_offset_t = [torch.tensor(data) for data in joint_offsets]
    joint_positions_t = [torch.tensor(data) for data in joint_positions]
    joint_orientations_t = [
        torch.tensor(R.from_quat(data).as_matrix(), requires_grad=True)
        for data in joint_orientations
    ]
    local_rotations_t = [
        torch.tensor(data.as_matrix(), requires_grad=True) for data in local_rotations
    ]
    target_pose_t = torch.tensor(target_pose,requires_grad=True)
    
    epochs = 1000
    alpha = 0.001
    
    for _ in range(epochs):
        for j in range(len(joint_ik_path)):
            chain_current = joint_ik_path[j]
            chain_parent = joint_ik_path[j - 1]
            if j == 0 :
                local_rotations_t[chain_current] = local_rotations_t[chain_current]
                joint_positions_t[chain_current] = joint_positions_t[chain_current]
            elif chain_parent == joint_parents[chain_current]: 
                joint_orientations_t[chain_current] = joint_orientations_t[chain_parent] @ local_rotations_t[chain_current]
                joint_positions_t[chain_current] = joint_positions_t[chain_parent] + \
                    joint_offset_t[chain_current] @ torch.transpose(joint_orientations_t[chain_parent], 0, 1)
            elif chain_current == joint_parents[chain_current]:
                joint_orientations_t[chain_current] = joint_orientations_t[chain_parent] @ torch.transpose(
                    local_rotations_t[chain_parent], 0, 1
                )
                joint_positions_t[chain_current] = joint_positions_t[chain_parent] - \
                    joint_offset_t[chain_current] @ torch.transpose(joint_orientations_t[chain_current], 0, 1)
    
        optimizer_target = torch.norm(
            joint_positions_t[joint_ik_path[-1]] -  target_pose_t
        )
        optimizer_target.backward()
        for index in joint_ik_path:
            if local_rotations_t[index].grad is not None:
                temp = local_rotations_t[index] - alpha * local_rotations_t[index].grad
                local_rotations_t[index] = torch.tensor(temp, requires_grad=True)
        
    for j in range(len(joint_ik_path)):
        chain_current = joint_ik_path[j]
        chain_parent = joint_ik_path[j - 1]
        if j == 0:
            local_rotations[chain_current] = R.from_matrix(local_rotations_t[chain_current].detach().numpy())
            joint_positions[chain_current] = joint_positions[chain_current]
        elif chain_parent == joint_parents[chain_current]:
            joint_orientations[chain_current] = (
                R.from_quat(joint_orientations[chain_parent]) * \
                    R.from_matrix(local_rotations_t[chain_current].detach().numpy())
            ).as_quat()
            joint_positions[chain_current] = (
                joint_positions[chain_parent] + \
                    joint_offsets[chain_current] * \
                        np.asmatrix(R.from_quat(joint_orientations[chain_parent]).as_matrix()).transpose()
            )
        else:
            joint_orientations[chain_current] = (
                R.from_quat(joint_orientations[chain_parent]) * \
                    R.from_matrix(local_rotations_t[chain_parent].detach().numpy()).inv()
            ).as_quat()
            joint_positions_t[chain_current] = (
                joint_positions[chain_parent] - \
                    joint_offsets[chain_parent] * \
                        np.asmatrix(R.from_quat(joint_orientations[chain_current]).as_matrix()).transpose()
            )
    
    ik_path_set = set(joint_ik_path)
    for i in range(len(joint_positions)):
        if i in ik_path_set:
            joint_orientations[i] = R.from_matrix(
                joint_orientations_t[i].detach().numpy()
            ).as_quat()
        else:
            joint_orientations[i] = (
                R.from_quat(joint_orientations[joint_parents[i]]) * local_rotations[i]
            ).as_quat()
            joint_positions[i] = (
                joint_positions[joint_parents[i]] + \
                    joint_offsets[i] * \
                        np.asmatrix(R.from_quat(joint_orientations[joint_parents[i]]).as_matrix()).transpose()
            )
    
    return joint_positions, joint_orientations
                
        

def part2_inverse_kinematics(
    meta_data,
    joint_positions,
    joint_orientations,
    relative_x,
    relative_z,
    target_height,
):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """

    return joint_positions, joint_orientations


def bonus_inverse_kinematics(
    meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose
):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    return joint_positions, joint_orientations
