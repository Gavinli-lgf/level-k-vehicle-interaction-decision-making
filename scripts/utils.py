'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-04-27 16:17:27
LastEditTime: 2024-10-31 01:01:25
FilePath: /vehicle-interaction-decision-making/scripts/utils.py
Copyright 2024 puyu, All Rights Reserved.
'''

import math
import random
import numpy as np
from enum import Enum
from typing import List, Optional, Tuple


# 预定义6个行为的行为列表
class Action(Enum):
    """Enum of action sets for vehicle."""
    MAINTAIN = [0, 0]              # maintain
    TURNLEFT = [0, math.pi / 4]    # turn left
    TURNRIGHT = [0, -math.pi / 4]  # turn right
    ACCELERATE = [2.5, 0]          # accelerate
    DECELERATE = [-2.5, 0]         # decelerate
    BRAKE = [-5, 0]                # brake

ActionList = [Action.MAINTAIN, Action.TURNLEFT, Action.TURNRIGHT,
              Action.ACCELERATE, Action.DECELERATE, Action.BRAKE]


# 车辆状态(x,y,yaw,v)
class State:
    def __init__(self, x=0, y=0, yaw=0, v=0) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def to_list(self) -> List:
        return [self.x, self.y, self.yaw, self.v]


# 管理和操作车辆状态的列表。它提供了添加状态、反转列表、扩展列表、转换为普通列表等功能
class StateList:
    def __init__(self, state_list = None) -> None:
        self.state_list: List[State] = state_list if state_list is not None else []

    def append(self, state: State) -> None:
        self.state_list.append(state)

    # 反转状态列表 state_list 的顺序，并返回自身self
    def reverse(self) -> 'StateList':
        self.state_list = self.state_list[::-1]

        return self

    """
    输入: excepted_len 希望将self.state_list拓展到的长度; expand_state 希望拓展的self.state_list的末尾状态.
    功能: 当前列表长度大于等于期望长度，则直接返回; 如果小于期望长度则将不足的状态个数全部补齐为expand_state(如果为none就重复原末尾状态)。
    输出: 空
    """
    def expand(self, excepted_len: int, expand_state: Optional[State] = None) -> None:
        cur_size = len(self.state_list)
        if cur_size >= excepted_len:
            return
        else:
            if expand_state is None:
                expand_state = self.state_list[-1]
            for _ in range(excepted_len - cur_size):
                self.state_list.append(expand_state)

    """
    输入: is_vertical: 当个状态是垂直的列,还是水平的行.
    功能: 改变self.state_list的排列格式(即不同的二维数组方式). 
         如果is_vertical为True,转换为"4*n"的形式,每一列是一个状态; 否则转换为"n*4"的形式,每一行是一个状态.
    输出: 改变格式后的状态列表states.
    """
    def to_list(self, is_vertical: bool = True) -> List:
        if is_vertical is True:
            states = [[],[],[],[]]
            for state in self.state_list:
                states[0].append(state.x)
                states[1].append(state.y)
                states[2].append(state.yaw)
                states[3].append(state.v)
        else:
            states = []
            for state in self.state_list:
                states.append([state.x, state.y, state.yaw, state.v])

        return states

    # 返回状态列表中共有几个状态(每个状态是一个(x,y,yaw,v))
    def __len__(self) -> int:
        return len(self.state_list)

    # 获取状态列表中的第 key 个状态( key 是索引)
    def __getitem__(self, key: int) -> State:
        return self.state_list[key]

    # 更改状态列表中第 key 个状态为 value
    def __setitem__(self, key: int, value: State) -> None:
        self.state_list[key] = value


"""
定义了蒙特卡洛树搜索(MCTS)中的节点。它包含了节点的状态、奖励、访问次数、父节点、子节点、当前层级、目标位置等信息,
并提供了添加子节点、生成下一个节点等功能。
"""
class Node:
    MAX_LEVEL: int = 6          # 搜索树的最大深度(层级).当节点达到 MAX_LEVEL 时,被认为是终端节点。
    calc_value_callback = None  # 回调函数,用于计算节点的值(value).这个函数通常由外部传入,用于自定义节点的值计算逻辑。

    # Node构造函数(输入数据: state, cur_level, parent, action, other_agent_state, goal_pos)
    def __init__(self, state = State(), level = 0, p: Optional["Node"] = None,
                 action: Optional[Action] = None, others: StateList = StateList(),
                 goal: State = State()) -> None:
        self.state: State = state           # Node的当前节点的状态

        self.value: float = 0               # 从根节点到当前节点总的价值(giving back), (不是该点的及时奖励)
        self.reward: float = 0              # 即时奖励值
        self.visits: int = 0                # 节点的访问次数
        self.action: Action = action        # 导致当前节点的动作
        self.parent: Node = p               # 父节点
        self.cur_level: int = level         # 当前节点的层级，表示从根节点到当前节点的深度
        self.goal_pos: State = goal         # 目标状态(车辆想达到的终点状态)

        self.children: List[Node] = []      # 子节点列表(子节点个数最多等于动作序列个数,且区别于孙节点等)
        self.actions: List[Action] = []     # 动作列表(从根节点到当前节点的动作序列)
        self.other_agent_state: StateList = others  # 其他车辆的当前状态列表(以列表的方式记录其它所有车辆当前状态,用于判碰,区别预测轨迹)

    # 如果当前节点的层级 cur_level 大于或等于 MAX_LEVEL,则认为当前节点是终端节点
    @property
    def is_terminal(self) -> bool:
        return self.cur_level >= Node.MAX_LEVEL

    # 如果当前节点的子节点个数大于等于动作序列个数,则认为该节点已完全拓展
    @property
    def is_fully_expanded(self) -> bool:
        return len(self.children) >= len(Action)

    # 初始化 Node 类的全局属性(最大层数 max_level; 计算节点价值的回调函数 callback,即为MCTS中的stage reward function)
    @staticmethod
    def initialize(max_level, callback) -> None:
        Node.MAX_LEVEL = max_level
        Node.calc_value_callback = callback

    # 根据输入next_action,delta_t,others; 使用状态转移方程计算下一个节点状态,更新其与父节点的连接信息; 返回新建的子节点.
    def add_child(self, next_action: Action, delta_t: float, others: List[State] = []) -> "Node":
        new_state = kinematic_propagate(self.state, next_action.value, delta_t)
        node = Node(new_state, self.cur_level + 1, self, next_action, others, self.goal_pos)
        node.actions = self.actions + [next_action]
        Node.calc_value_callback(node, self.value)
        self.children.append(node)

        return node

    # 随机选择一个动作，从self.state生成一个节点.(与add_child()区别在于,该节点不记录父节点,即不生成树,用于MCTS的Simulation过程)
    def next_node(self, delta_t: float, others: StateList = StateList()) -> "Node":
        next_action = random.choice(ActionList)
        new_state = kinematic_propagate(self.state, next_action.value, delta_t)
        node = Node(new_state, self.cur_level + 1, None, next_action, others, self.goal_pos)
        Node.calc_value_callback(node, self.value)

        return node

    # 定义节点的字符串表示形式，方便调试和日志记录
    def __repr__(self):
        return (f"children: {len(self.children)}, visits: {self.visits}, "
                f"reward: {self.reward}, actions: {self.actions}")


# 使用分离轴定理（Separating Axis Theorem, SAT）检测两个多边形box2d_0, box2d_1是否相交.
def has_overlap(box2d_0, box2d_1) -> bool:
    total_sides = []
    for i in range(1, len(box2d_0[0])):
        vec_x = box2d_0[0][i] - box2d_0[0][i - 1]
        vec_y = box2d_0[1][i] - box2d_0[1][i - 1]
        total_sides.append([vec_x, vec_y])
    for i in range(1, len(box2d_1[0])):
        vec_x = box2d_1[0][i] - box2d_1[0][i - 1]
        vec_y = box2d_1[1][i] - box2d_1[1][i - 1]
        total_sides.append([vec_x, vec_y])

    for i in range(len(total_sides)):
        separating_axis = [-total_sides[i][1], total_sides[i][0]]

        vehicle_min = np.inf
        vehicle_max = -np.inf
        for j in range(0, len(box2d_0[0])):
            project = separating_axis[0] * box2d_0[0][j] + separating_axis[1] * box2d_0[1][j]
            vehicle_min = min(vehicle_min, project)
            vehicle_max = max(vehicle_max, project)

        box2d_min = np.inf
        box2d_max = -np.inf
        for j in range(0, len(box2d_1[0])):
            project = separating_axis[0] * box2d_1[0][j] + separating_axis[1] * box2d_1[1][j]
            box2d_min = min(box2d_min, project)
            box2d_max = max(box2d_max, project)

        if vehicle_min > box2d_max or box2d_min > vehicle_max:
            return False

    return True


# 状态转移方程(只取第1个输入，计算下一个状态。并对yaw限幅(0~2*pi),v限幅(-20~20))
def kinematic_propagate(state: State, act: List[float], dt: float) -> State:
    next_state = State()
    acc, omega = act[0], act[1]

    next_state.x = state.x + state.v * np.cos(state.yaw) * dt
    next_state.y = state.y + state.v * np.sin(state.yaw) * dt
    next_state.v = state.v + acc * dt
    next_state.yaw = state.yaw + omega * dt

    while next_state.yaw > 2 * np.pi:
        next_state.yaw -= 2 * np.pi
    while next_state.yaw < 0:
        next_state.yaw += 2 * np.pi

    if next_state.v > 20:
        next_state.v = 20
    elif next_state.v < -20:
        next_state.v = -20

    return next_state
