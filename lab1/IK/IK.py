from BVH_Reader.bvh_read import Bone_Tree
from torch import torch
import numpy as np


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

    try:
        path_index = path_info[0]
        path_name = path_info[1]
    except:
        if len(path_index) == 2:
            raise ValueError("more path here not supporte")
    # this only process one path case

    primary_positions, primary_rotations = (
        bone_tree.get_bone_tree_primary_positions_rotations_from_AJoint_To_BJoint(
            path_info=path_info
        )
    )

    # reverse -> from RootJoint to end joint
    primary_positions.reverse()
    primary_rotations.reverse()

    # type : np.ndarray
    primary_positions_root = primary_positions[0]

    # create a rotation parameter matrix
    rotation_matrix = torch.from_numpy(
        np.array([x.tolist() for x in primary_rotations])
    )
    """
    torch array like :
    tensor([[ 0.0065, -0.0204, -0.0107,  0.9997],
        [ 0.0065, -0.0204, -0.0107,  0.9997],
        [ 0.0755, -0.0239,  0.0039,  0.9969],
        [ 0.2270,  0.0226, -0.0644,  0.9715],
        [ 0.1131,  0.0359, -0.0237,  0.9927],
        [ 0.0170,  0.0664, -0.0058,  0.9976]], dtype=torch.float64)
    """
