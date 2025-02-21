'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-05-13 23:24:21
LastEditTime: 2024-10-31 01:01:03
FilePath: /vehicle-interaction-decision-making/scripts/planner.py
Copyright 2024 puyu, All Rights Reserved.
'''

import math
import random
import logging
import numpy as np
from typing import Tuple, List

import utils
from utils import Node, StateList
from vehicle_base import VehicleBase


"""
MCTS是一个方法框架,用在什么场景中的关键点是:动作的定义; 状态价值的计算. (需要根据不同的应用场景更改; 也可作为接口对外开放,让外部更改.)
"""
class MonteCarloTreeSearch:
    EXPLORATE_RATE = 1 / ( 2 * math.sqrt(2.0))  # 探索率，用于平衡探索和利用
    LAMDA = 0.9             # 折扣因子，用于计算累积奖励。
    WEIGHT_AVOID = 10       # collision avoidance
    WEIGHT_SAFE = 0.2       # safe distance
    WEIGHT_OFFROAD = 2      # off road
    WEIGHT_DIRECTION = 1    # lane for opposite direction traffic
    WEIGHT_DISTANCE = 0.1   # distance to objective
    WEIGHT_VELOCITY = 0.05  # 目标速度

    # 初始化函数1 ego_vehicle, other_vehicle(其它车的Vehicle对象), other_predict_traj(MCTS中每个level其它agents的状态), cfg(配置信息)
    def __init__(self, ego: VehicleBase, others: List[VehicleBase],
                 other_traj: List[StateList], cfg: dict = {}):
        self.ego_vehicle: VehicleBase = ego
        self.other_vehicle: VehicleBase = others
        self.other_predict_traj: StateList = other_traj #类型为List[StateList]:记录MCTS中每个level其它agents的状态(区别于传统的预测轨迹)

        self.computation_budget = cfg['computation_budget'] # 15000 (完成一次MCTS搜索的迭代次数)
        self.dt = cfg['delta_t']                            # 0.25 (时间步长)

    # 对根节点root完成一次完整的MCTS搜索，并返回root的最佳子节点。
    def excute(self, root: Node) -> Node:
        for _ in range(self.computation_budget):
            # 1. Find the best node to expand (Selection,Expansion)
            expand_node = self.tree_policy(root)
            # 2. Random run to add node and get reward (Simulation)
            reward = self.default_policy(expand_node)
            # 3. Update all passing nodes with reward (Backpropagation)
            self.update(expand_node, reward)

        return self.get_best_child(root, 0)

    # Selection,Expansion: 选择要扩展的节点并拓展，返回新拓展的节点
    def tree_policy(self, node: Node) -> Node:
        while node.is_terminal == False:
            if len(node.children) == 0:
                return self.expand(node)
            elif random.uniform(0, 1) < .5:
                node = self.get_best_child(node, MonteCarloTreeSearch.EXPLORATE_RATE)
            else:
                if node.is_fully_expanded == False:    
                    return self.expand(node)
                else:
                    node = self.get_best_child(node, MonteCarloTreeSearch.EXPLORATE_RATE)

        return node

    """
    Simulation: 从新拓展节点node开始,随机选择动作生成next_node,直到达到终端节点。返回终端节点的奖励值。
    (注: simulation生成next_node不是为了生成树,因此不用记录父子节点的连接关系.)
    """
    def default_policy(self, node: Node) -> float:
        # terminal的条件是到达了MCTS的最大层MAX_LEVEL
        while node.is_terminal == False:
            # 其他所有agents在MCTS树中第"node.cur_level + 1"层的状态(以列表的方式记录,区别传统预测轨迹)
            cur_other_state = self.other_predict_traj[node.cur_level + 1] 
            next_node = node.next_node(self.dt, cur_other_state)
            node = next_node

        return node.value

    # Backpropagation: 更新从给定节点到根节点的路径上所有节点的访问次数和奖励值
    def update(self, node: Node, r: float) -> None:
        while node != None:
            node.visits += 1
            node.reward += r
            node = node.parent

    # 生成node的一个新的子节点(与原有子节点的 action 不相同)，并返回该子节点。(注:记录父子关系,生成MCTS树)
    def expand(self, node: Node) -> Node:
        tried_actions = [c.action for c in node.children]
        next_action = random.choice(utils.ActionList)
        while node.is_terminal == False and next_action in tried_actions:
            next_action = random.choice(utils.ActionList)
        cur_other_state = self.other_predict_traj[node.cur_level + 1]
        node.add_child(next_action, self.dt, cur_other_state)

        return node.children[-1]

    # 基于UCB1公式(探索率 scalar)选择当前节点 node 的最佳子节点,并返回该最佳子节点.
    def get_best_child(self, node: Node, scalar: float) -> Node:
        bestscore = -math.inf
        bestchildren = []
        for child in node.children:
            exploit = child.reward / child.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / child.visits)
            score = exploit + scalar * explore
            if score == bestscore:
                bestchildren.append(child)
            if score > bestscore:
                bestchildren = [child]
                bestscore = score
        if len(bestchildren) == 0:
            logging.debug("No best child found, probably fatal !")
            return node

        return random.choice(bestchildren)

    """
    输入: node 当前节点, last_node_value 父节点的状态价值(giving back);
    功能: 参照论文中"stage reward function(式3)"的定义, 与MDP中episode状态价值的定义, 
          计算从根节点到当前节点总的状态价值, 并返回.
    """
    @staticmethod
    def calc_cur_value(node: Node, last_node_value: float) -> float:
        # 计算自车的box2d与safezone
        x, y, yaw = node.state.x, node.state.y, node.state.yaw
        step = node.cur_level
        ego_box2d = VehicleBase.get_box2d(node.state)
        ego_safezone = VehicleBase.get_safezone(node.state)

        # 计算 collision avoidance 和 safe distance 的 reward
        avoid = 0
        safe = 0
        for cur_other_state in node.other_agent_state:
            if utils.has_overlap(ego_box2d, VehicleBase.get_box2d(cur_other_state)):
                avoid = -1
            if utils.has_overlap(ego_safezone, VehicleBase.get_safezone(cur_other_state)):
                safe = -1

        # 计算 off road 的 reward(与道路外区域是否overlap)
        offroad = 0
        for rect in VehicleBase.env.rect:
            if utils.has_overlap(ego_box2d, rect):
                offroad = -1
                break

        # 计算 lane for opposite direction traffic 的 reward
        direction = 0
        if MonteCarloTreeSearch.is_opposite_direction(node.state, ego_box2d):
            direction = -1

        # 计算 distance to objective 的 reward
        delta_yaw = abs(yaw - node.goal_pos.yaw) % (np.pi * 2)
        delta_yaw = min(delta_yaw, np.pi * 2 - delta_yaw)
        distance = -(abs(x - node.goal_pos.x) + abs(y - node.goal_pos.y) + 1.5 * delta_yaw)

        # 当前状态总的即使奖励(reward)
        cur_reward = MonteCarloTreeSearch.WEIGHT_AVOID * avoid + \
                     MonteCarloTreeSearch.WEIGHT_SAFE * safe + \
                     MonteCarloTreeSearch.WEIGHT_OFFROAD * offroad + \
                     MonteCarloTreeSearch.WEIGHT_DISTANCE * distance + \
                     MonteCarloTreeSearch.WEIGHT_DIRECTION * direction + \
                     MonteCarloTreeSearch.WEIGHT_VELOCITY * node.state.v

        # 根据MDP中episode状态价值的定义, 计算从根节点到当前节点总的状态价值(giving back)
        total_reward = last_node_value + (MonteCarloTreeSearch.LAMDA ** (step - 1)) * cur_reward
        node.value = total_reward

        return total_reward

    # 判断自车是否与道路方向相反。如果自车在道路上且方向与道路方向相反，则返回 True
    @staticmethod
    def is_opposite_direction(pos: utils.State, ego_box2d = None) -> bool:
        x, y, yaw = pos.x, pos.y, pos.yaw
        if ego_box2d is None:
            ego_box2d = VehicleBase.get_box2d(pos)

        for laneline in VehicleBase.env.laneline:
            if utils.has_overlap(ego_box2d, laneline):
                return True

        lanewidth = VehicleBase.env.lanewidth

        # down lane
        if x > -lanewidth and x < 0 and (y < -lanewidth or y > lanewidth):
            if yaw > 0 and yaw < np.pi:
                return True
        # up lane
        elif x > 0 and x < lanewidth and (y < -lanewidth or y > lanewidth):
            if not (yaw > 0 and yaw < np.pi):
                return True
        # right lane
        elif y > -lanewidth and y < 0 and (x < -lanewidth or x > lanewidth):
            if yaw > 0.5 * np.pi and yaw < 1.5 * np.pi:
                return True
        # left lane
        elif y > 0 and y < lanewidth and (x < -lanewidth or x > lanewidth):
            if not (yaw > 0.5 * np.pi and yaw < 1.5 * np.pi):
                return True

        return False

    # 初始化函数2: 输入配置文件cfg进行初始化
    @staticmethod
    def initialize(cfg: dict = {}) -> None:
        MonteCarloTreeSearch.LAMDA = cfg['lamda']
        MonteCarloTreeSearch.WEIGHT_AVOID = cfg['weight_avoid']
        MonteCarloTreeSearch.WEIGHT_SAFE = cfg['weight_safe']
        MonteCarloTreeSearch.WEIGHT_OFFROAD = cfg['weight_offroad']
        MonteCarloTreeSearch.WEIGHT_DIRECTION = cfg['weight_direction']
        MonteCarloTreeSearch.WEIGHT_DISTANCE = cfg['weight_distance']
        MonteCarloTreeSearch.WEIGHT_VELOCITY = cfg['weight_velocity']


# KLevelPlanner用于车辆交互决策的规划器类。它通过level-k进行预测和仿真，生成车辆的最优动作和预期轨迹。
class KLevelPlanner:
    def __init__(self, cfg: dict = {}):
        self.steps = cfg['max_step']    # 配置文件中为 8
        self.dt = cfg['delta_t']        # 配置文件中为 0.25
        self.config = cfg               # unprotected_left_turn.yaml

    # 输入: ego 自车(单个); 他车 others(list多个)。 输出: 自车当前待执行最优动作; 自车最优状态序列(即预测轨迹)。
    def planning(self, ego: VehicleBase, others: List[VehicleBase]) -> Tuple[utils.Action, StateList]:
        # other_prediction 类型为 List[StateList] :记录MCTS中每个 level 时其它 agents 的状态(区别于传统的预测轨迹)
        other_prediction = self.get_prediction(ego, others)
        actions, traj = self.forward_simulate(ego, others, other_prediction) # 使用MCTS前向仿真，生成自车最优动作和最优状态序列(即预期轨迹)

        return actions[0], traj

    """
    输入: ego,others的vehicle对象(ego,others); others在MCTS中每个level的预测状态(traj); 
    功能: 从当前节点进行一次完整的MCTS搜索.只考虑利用,不考虑探索的情况下,一直找到mcts最后一层的最优节点.
         顺序记录从根节点到该节点的状态,即为自车的 expected_traj; 同样顺序记录的动作序列为自车的最优动作序列.
    输出: 经过本次MCTS得到的自车的最优动作序列 actions 与最优状态序列 expected_traj(即自车的预测轨迹) 。
    """
    def forward_simulate(self, ego: VehicleBase, others: List[VehicleBase],
                         traj: List[StateList]) -> Tuple[List[utils.Action], StateList]:
        # 输入ego,others的vehicle对象(ego,others);others在MCTS中每个level的预测状态(traj); 进行一次完整的MCTS搜索
        mcts = MonteCarloTreeSearch(ego, others, traj, self.config)
        current_node = Node(state = ego.state, goal = ego.target)
        current_node = mcts.excute(current_node)

        # scalar置0(只考虑利用,不考虑探索)的情况下，一直找到mcts最后一层的最优节点，再顺序记录从根节点到该节点的状态，即为自车的 expected_traj
        for _ in range(Node.MAX_LEVEL - 1):
            current_node = mcts.get_best_child(current_node, 0)

        actions = current_node.actions  #最后一层的最优子节点上，已经记录了从根节点到该节点所有的动作列表了
        state_list = StateList()
        while current_node != None:
            state_list.append(current_node.state)
            current_node = current_node.parent
        expected_traj = state_list.reverse()

        # 如果最优状态序列的长度不足"self.steps + 1",就用最后一个状态补足。
        if len(expected_traj) < self.steps + 1:
            logging.debug(f"The max level of the node is not enough({len(expected_traj)}),"
                          f"using the last value to complete it.")
            expected_traj.expand(self.steps + 1)

        return actions, expected_traj

    """
    输入: ego,others的vehicle对象(ego,others); 
    功能: 基于 Level-K Thinking与交换自车和其他车辆的角色,通过递归的方式，预测其他车辆在未来时间步长内的状态轨迹。
         重点是递归结束条件2个:到达level-0; 辆车已经到达目标点。
    输出: others在MCTS中每个level的预测状态(即按时间步长组织的others的预测轨迹.每个步长也就对应了MCTS中的每个层.); 
    """
    def get_prediction(self, ego: VehicleBase, others: List[VehicleBase]) -> List[StateList]:
        pred_trajectory = []        # 用于存储最终的预测轨迹，按时间步长组织。
        pred_trajectory_trans = []  # 用于临时存储每辆车的预测轨迹。

        if ego.level == 0:
            #递归结束条件: 当ego为level-0时,其它agents都认为是静态的.因此MCTS中每一层others的状态预测结果,都是他们当前的状态.获取到 pred_trajectory 并return.(由level-k理论可得前述过程)
            for i in range(self.steps + 1):
                pred_traj: StateList = StateList()
                for other in others:
                    pred_traj.append(other.state)
                pred_trajectory.append(pred_traj)
            return pred_trajectory
        elif ego.level > 0:
            for idx in range(len(others)):      # 遍历每辆 others
                if others[idx].is_get_target:
                    # 如果该other车辆已经到达终点，则其在MCTS中每层的预测状态，就是其当前状态保持不变。获取到 pred_trajectory_trans 中保存.
                    pred_traj: StateList = StateList()
                    for i in range(self.steps + 1):
                        pred_traj.append(others[idx].state)
                    pred_trajectory_trans.append(pred_traj)
                    continue
                # 交换角色,降低层级,递归预测,前向仿真(递归模拟其他车辆在level-k-1的交互,最终到level-0时结果才是确定的,才能往>0的level反推。)
                exchanged_ego: VehicleBase = others[idx]
                exchanged_ego.level = ego.level - 1
                exchanged_others: List[VehicleBase] = [ego]
                for i in range(len(others)):
                    if i != idx:
                        exchanged_others.append(others[i])
                exchage_pred_others = self.get_prediction(exchanged_ego, exchanged_others)
                _, pred_idx_vechicle = self.forward_simulate(exchanged_ego, exchanged_others, exchage_pred_others)
                pred_trajectory_trans.append(pred_idx_vechicle)
        else:
            # level-k的level异常
            logging.error("get_prediction() excute error, the level must be >= 0 and > 3 !")
            return pred_trajectory

        # 遍历每个时间步长(self.steps + 1);对于每个时间步长,遍历所有车辆的预测轨迹,将对应时间步长上的状态添加到state中。最后将每个步长上的others的state加入到pred_trajectory中
        for i in range(self.steps + 1):
            state = StateList()
            for states in pred_trajectory_trans:
                state.append(states[i])
            pred_trajectory.append(state)

        return pred_trajectory
