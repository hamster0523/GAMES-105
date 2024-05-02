from dataclasses import dataclass
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Node:
    name: str
    offset: list
    channels: dict
    child: list
    parent: dict
    ID: int
    channel_index: int
    type: str


class Bone_Tree:
    def __init__(self, bvh_file_path) -> None:
        self.root = None
        self.joint_List = None
        self.node_count = 0
        self.num_frames = 0
        self.frame_time = 0
        self.frames = None
        self.all_channels = []
        self.read_bvh(bvh_file_path)
        self.root = self.Find_Root(self.joint_List)
        self.sorted_joint_list = self.sort_primary_joint_list(self.joint_List)
        self.channels = self.process_channels(self.all_channels)
        self.joint_name = self.process_joint_name(self.sorted_joint_list)
        self.joint_offset = np.array(self.process_joint_offset(self.sorted_joint_list))
        self.joint_parent = self.process_joint_parent(self.sorted_joint_list)
        self.all_frame_rotation_location_dic = self.forward_kinematics(
            self.sorted_joint_list, self.frames
        )
        self.all_frame_rotation = self.Process_all_frames_rotation(
            self.all_frame_rotation_location_dic
        )
        self.all_frame_location = self.Process_all_frames_location(
            self.all_frame_rotation_location_dic
        )

    def read_bvh(self, file_path) -> None:
        bvh_data = []
        with open(file_path, "r") as f:
            for line in f:
                bvh_data.append(line.split())

        num_frames = 0
        frame_time = 0
        frames = []
        info_List = []

        for line in bvh_data:
            this_line = line
            if this_line[0] == "Hierarchy":
                continue
            if this_line[0] == "HIERARCHY":
                continue
            if this_line[0] == "MOTION":
                continue
            if this_line[0] == "Frames:":
                num_frames = int(this_line[1])
                continue
            if this_line[0] == "Frame":
                frame_time = float(this_line[2])
                continue
            if (
                this_line[0] == "{"
                or this_line[0] == "ROOT"
                or this_line[0] == "}"
                or this_line[0] == "End"
                or this_line[0] == "JOINT"
                or this_line[0] == "OFFSET"
            ):
                info_List.append(this_line)
                continue
            if this_line[0] == "CHANNELS":
                info_List.append(this_line)
                continue
            frames.append(this_line)

        self.num_frames = num_frames
        self.frame_time = frame_time
        self.frames = frames

        offset_stack = deque()
        channel_stack = deque()
        node_stack = deque()
        joint_List = []
        node_id = 0
        channel_index = 0

        for info in info_List:
            if info[0] == "ROOT":
                node_stack.append(
                    Node(
                        name="RootJoint",
                        offset=None,
                        channels=None,
                        child=[],
                        parent=None,
                        ID=node_id,
                        channel_index=channel_index,
                        type="root",
                    )
                )
                channel_index += 6
                node_id += 1
                continue
            elif info[0] == "{":
                continue
            elif info[0] == "OFFSET":
                offset_stack.append(list(map(float, info[1:])))
                continue
            elif info[0] == "CHANNELS":
                channel_stack.append(
                    {"channel_num": int(info[1]), "channels": info[2:]}
                )
                self.all_channels.append(info[2:])
                continue
            elif info[0] == "JOINT":
                node_stack.append(
                    Node(
                        name=info[1],
                        offset=None,
                        channels=None,
                        child=[],
                        parent=None,
                        ID=node_id,
                        channel_index=channel_index,
                        type="joint",
                    )
                )
                node_id += 1
                channel_index += 3
                continue
            elif info[0] == "End":
                node_stack.append(
                    Node(
                        name=info[0] + info[1],
                        offset=None,
                        channels=None,
                        parent=None,
                        child=None,
                        type="End",
                        ID=node_id,
                        channel_index=None,
                    )
                )
                node_id += 1
                continue
            elif info[0] == "}":
                pop_node = node_stack.pop()
                if pop_node.type == "End":
                    top_node = node_stack[-1]
                    pop_node.name = top_node.name + "_end"
                    # pop_node.parent = top_node.name
                    pop_node.parent = {
                        "parent_name": top_node.name,
                        "parent_id": top_node.ID,
                    }
                    pop_node.child = None
                    offset_top = offset_stack.pop()
                    pop_node.offset = offset_top
                    pop_node.channels = None
                    top_node.child.append(pop_node)
                    joint_List.append(pop_node)
                elif pop_node.type == "joint":
                    top_node = node_stack[-1]
                    pop_offset = offset_stack.pop()
                    pop_channels = channel_stack.pop()
                    # print(pop_channels)
                    pop_node.offset = pop_offset
                    pop_node.channels = pop_channels
                    # pop_node.parent = top_node.name
                    pop_node.parent = {
                        "parent_name": top_node.name,
                        "parent_id": top_node.ID,
                    }
                    top_node.child.append(pop_node)
                    joint_List.append(pop_node)
                elif pop_node.type == "root":
                    pop_offset = offset_stack.pop()
                    pop_channels = channel_stack.pop()
                    # print(pop_channels)
                    pop_node.offset = pop_offset
                    pop_node.channels = pop_channels
                    pop_node.parent = {"parent_name": None, "parent_id": -1}
                    joint_List.append(pop_node)
                continue
        self.joint_List = joint_List
        self.node_count = node_id

    def Find_Root(self, joint_list) -> Node:
        for node in joint_list:
            if node.name == "RootJoint":
                return node

    def Get_JointNode_by_name(self, name) -> Node:
        for node in self.joint_List:
            if node.name == name:
                return node

    def process_channels(self, channels_list):
        channels = []
        for subchannel in channels_list:
            for channel in subchannel:
                channels.append(channel)
        return channels

    def process_joint_name(self, joint_list):
        name_list = []
        for joint in joint_list:
            name_list.append(joint.name)
        return name_list

    def process_joint_offset(self, joint_list):
        joint_offset_list = []
        for joint in joint_list:
            joint_offset_list.append(joint.offset)
        return joint_offset_list

    def process_joint_parent(self, joint_list):
        parent_id_list = []
        for joint in joint_list:
            parent_id_list.append(joint.parent["parent_id"])
        return parent_id_list

    def sort_primary_joint_list(self, joint_list):
        new_joint_list = []
        for i in range(self.node_count):
            for joint in joint_list:
                if joint.ID == i:
                    new_joint_list.append(joint)
        return new_joint_list

    def find_parent_rotation(self, Q, parent_index):
        for joint in Q:
            if joint["id"] == parent_index:
                return joint["rotation"]

    def find_parent_location(self, Q, parent_index):
        for joint in Q:
            if joint["id"] == parent_index:
                return joint["location"]

    def forward_kinematics(
        self,
        joint_list: list,
        motion_data: list,
    ) -> list:
        # Q4 = Q3R4 = Q2R3R4 = Q1R2R3R4 = Q0R1R2R3R4 = R0R1R2R3R4
        # DP accumulate
        # joint_all_frames_info_rotation_local : shape -> num_frames * num_joint
        joint_all_frames_info_rotation_local = []
        for frame in motion_data:
            # Q : return list of dict in this frame
            # each dict contains id, rotation accumulate from root to this joint
            #                        location of this join
            # Q's id length is num_joint
            # Q's rotation shape is (num_joint, 4)
            # Q's location shape is (num_joint, 3)
            Q = []
            # Only root joint has translation
            T0 = None
            # joint_list is sorted by ID : the sequence follow the bvh hierarchy
            for joint in joint_list:
                if joint.type == "root":
                    # root channels is 6 : first 3 is translate , second 3 is rotation
                    translate = frame[:3]
                    rotation = frame[3:6]
                    # translate_euler = R.from_euler('XYZ', translate, degrees=True)
                    T0 = np.array(translate)
                    rotation_euler = R.from_euler("XYZ", rotation, degrees=True)
                    rotation_quaternion = rotation_euler.as_quat()
                    # root's location is (0,0,0)
                    Q.append(
                        {
                            "id": 0,
                            "rotation": rotation_quaternion,
                            "location": np.array([0, 0, 0]),
                        }
                    )
                elif joint.type == "joint":
                    joint_index = joint.ID - 1
                    parent_index = joint.parent["parent_id"]
                    joint_rotation = frame[
                        joint.channel_index : joint.channel_index + 3
                    ]
                    # print(joint.ID)
                    # print(joint_index)
                    joint_rotation = np.array(list(map(float, joint_rotation)))
                    # parent_roation is quaternion
                    parent_rotation = self.find_parent_rotation(Q, parent_index)
                    # print(joint_rotation)
                    # print(type(joint_rotation))
                    # print(R.from_euler("xyz", joint_rotation, degrees=True))
                    # print(R.from_euler("xyz", joint_rotation, degrees=True).as_matrix())
                    this_joint_rotation = R.from_matrix(
                        R.from_quat(parent_rotation).as_matrix()
                        @ R.from_euler("xyz", joint_rotation, degrees=True).as_matrix()
                    ).as_quat()
                    # print(this_joint_rotation)
                    # print("--------------")
                    # parent_location is np.array -> 3 dim vec
                    parent_location = self.find_parent_location(Q, parent_index)
                    # joint_offset is np.array -> 3 dim vec
                    joint_offset = joint.offset
                    # ID == 1 -> need translate
                    if joint.ID == 1:
                        T0 = list(map(float, T0))
                        this_joint_location = (
                            parent_location
                            + R.from_quat(parent_rotation).apply(np.array(joint_offset))
                            + T0
                        )
                    else:
                        this_joint_location = parent_location + R.from_quat(
                            parent_rotation
                        ).apply(np.array(joint_offset))
                    Q.append(
                        {
                            "id": joint.ID,
                            "rotation": this_joint_rotation,
                            "location": this_joint_location,
                        }
                    )
                elif joint.type == "End":
                    parent_index = joint.parent["parent_id"]
                    parent_rotation = self.find_parent_rotation(Q, parent_index)
                    parent_location = self.find_parent_location(Q, parent_index)
                    this_offset = joint.offset
                    this_joint_location = parent_location + R.from_quat(
                        parent_rotation
                    ).apply(np.array(this_offset))
                    Q.append(
                        {
                            "id": joint.ID,
                            "rotation": None,
                            "location": this_joint_location,
                        }
                    )
            joint_all_frames_info_rotation_local.append(Q)
        return joint_all_frames_info_rotation_local

    def Process_all_frames_rotation(self, joint_all_frames_info_rotation_location):
        all_frames_rotation = []
        for frame in joint_all_frames_info_rotation_location:
            one_frame_roation = []
            for joint in frame:
                one_frame_roation.append(joint["rotation"])
            all_frames_rotation.append(one_frame_roation)
        return all_frames_rotation

    def Process_all_frames_location(self, joint_all_frames_info_rotation_location):
        all_frames_location = []
        for frame in joint_all_frames_info_rotation_location:
            one_frame_location = []
            for joint in frame:
                one_frame_location.append(joint["location"])
            all_frames_location.append(one_frame_location)
        return all_frames_location


def __main__():
    pass
