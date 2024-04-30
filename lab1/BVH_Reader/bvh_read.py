from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class Node:
    name: str
    offset: list
    channels: dict
    child: list
    parent: dict
    ID: int
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
                        type="root",
                    )
                )
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
                        type="joint",
                    )
                )
                node_id += 1
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


def __main__():
    pass
