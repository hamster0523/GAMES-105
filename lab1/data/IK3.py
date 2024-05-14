from read_bvh import Bone_Tree
from torch import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_joint_offset_along_path(bone_tree : Bone_Tree, path_name : list):
    path_offset_along_path = []
    
    '''
    path_name is like : ['lToeJoint_end', 'lToeJoint', 'lAnkle', 'lKnee', 'lHip', 'RootJoint']
              is from end to root joint
    '''
    all_joint_name_index = [
        bone_tree.joint_name.index(joint_name)
        for joint_name in path_name
    ]
    
    path_offset_along_path = [
        bone_tree.joint_offset[joint_name_index]
        for joint_name_index in all_joint_name_index
    ]
    # so the offset is from end to root joint
    path_offset_along_path.reverse()
    return path_offset_along_path

def forward_kinematics(
    root_positions_tensor : torch.Tensor,
    root_to_end_offset_tensor : torch.Tensor,
    root_to_end_rotations_tensor : torch.Tensor
) -> torch.Tensor:

    # # calculate every joint positions
    # for i in range(len(num_joints)):
    #     current_joint_name = path_names[i]
    #     current_joint_index = path_index[i]
    #     current_joint_offset = root_to_end_offset_tensor[i]
    #     current_Node = joint_list[current_joint_index]
    #     parent_joint_index = joint_parent[current_joint_index]
    #     parent_joint_name =  joint_names[parent_joint_index]
    #     if current_Node.type == 'root':
    #       chains.append(
    #           {
    #               "name" : current_Node.name,
    #               "position" : root_positions_tensor,
    #               "orientation" : root_to_end_rotations_tensor[0]
    #           }
    #       )  
    #     elif current_Node.type == 'joint':
    #         parent_position = find_parent_positions(chains, parent_joint_name)
    
    # calculate all joint orientations
    Q = []
    Q.append(
        R.from_euler("XYZ", [0, 0, 0], degrees=True).as_matrix()
    )
    for i in range(len(root_to_end_rotations_tensor)):
        Q.append(
            R.from_quat(root_to_end_rotations_tensor[i].tolist()).as_matrix() \
                @ Q[-1]
        )
    '''
    all needed orientations are calculated , here is list
    R0 R0R1 R0R1R2 ...
    '''    
    # exclude the identity matrix
    Q = Q[1:]
    
    # calculate all joint positions
    joint_position_list = []
    joint_position_list.append(root_positions_tensor.tolist())
    for i in range(len(Q)):
        joint_position_list.append(
            Q[i] @ root_to_end_offset_tensor[i].tolist() + joint_position_list[-1]
        )
    
    # convert to tensor
    joint_position_tensor = [
        torch.tensor(data, requires_grad = True)
        for data in joint_position_list
    ]
    joint_orientation_tensor = [
        torch.tensor(data, requires_grad = True)
        for data in Q
    ]
    
    return joint_position_tensor, joint_orientation_tensor
    


def inverse_kinematics_2(
    bone_tree : Bone_Tree,
    path_info : list,
    target_pos : list
) -> list:
    threshold = 0.001
    learning_rate = 0.01
    epochs = 1000
    
    try:
        path_index = path_info[0]
        path_name  = path_info[1]
        num_joints = len(path_index)
    except:
        if(len(path_index) == 2):
            raise ValueError("two path not supported yet")
    
    # primary joint positions and  ! local ! rotations
    primary_positions, primary_rotations = bone_tree.get_bone_tree_primary_positions_rotations_from_AJoint_To_BJoint(
        path_info = path_info
    )
    #print("primary_positions : ", primary_positions)
    #print("primary_rotations : ", primary_rotations)
    # root position ->  change every iteration
    primary_positions.reverse()
    #print("root_to_end_positions : ", primary_positions)
    root_position_not_change = primary_positions[0]
    # rotations     ->  change every iteration -> calculate grad -> update
    primary_rotations.reverse()
    #print("root_to_end_rotations : ", primary_rotations)
    # offset        ->  not change every iteration
    root_to_end_offset = get_joint_offset_along_path(bone_tree = bone_tree, path_name = path_name)
    #print("root_to_end_offset : ", root_to_end_offset)
    
    
    root_to_end_positions_tensor = [
        torch.tensor(data, requires_grad = True)
        for data in primary_positions
    ]
    root_position_not_change_tensor =torch.tensor(root_position_not_change, requires_grad = True)
    root_to_end_rotations_tensor = [
        torch.tensor(data, requires_grad = True)
        for data in primary_rotations
    ]
    root_to_end_offset_tensor = [
        torch.tensor(data, requires_grad = True)
        for data in root_to_end_offset
    ]
    target_pos_tensor = torch.tensor(target_pos, requires_grad = True)
    
    for epcoh in range(epochs):
        # forward kinematics
        root_to_end_positions_tensor, root_to_end_orientations_tensor = forward_kinematics(
            root_positions_tensor = root_position_not_change_tensor,
            root_to_end_offset_tensor = root_to_end_offset_tensor,
            root_to_end_rotations_tensor = root_to_end_rotations_tensor
        )
        optimizer_target = torch.norm(
            root_to_end_positions_tensor[-1] - target_pos_tensor
        )
        optimizer_target.backward()
        
        print("epoch : {}, optimizer_target : {}", epcoh, optimizer_target.item())
        
        # calculate every rotations grad
        for i in range(num_joints):
            if root_to_end_rotations_tensor[i].grad is not None:
                print("grad of root_to_end_rotations_tensor[{}] : {}".format(i, root_to_end_rotations_tensor[i].grad))
                temp = root_to_end_rotations_tensor[i] - learning_rate * \
                    root_to_end_rotations_tensor[i].grad
                root_to_end_rotations_tensor[i] = temp
        
        if optimizer_target.item() < threshold:
            break
    
    # calculate the final position and orientation
    final_positions, final_orientations = forward_kinematics(
            root_positions_tensor = root_position_not_change_tensor,
            root_to_end_offset_tensor = root_to_end_offset_tensor,
            root_to_end_rotations_tensor = root_to_end_rotations_tensor
        )
    
    # return the final position and orientation -> list
    return final_positions.detach().numpy().tolist(), final_orientations.detach().numpy().tolist()
           