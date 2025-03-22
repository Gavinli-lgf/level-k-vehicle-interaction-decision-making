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

    """
    输入: ego 当前agent; others 除ego外其他所有对手agents; 
         other_traj 除ego外其他所有对手agents的预测轨迹(类型List[StateList],二维.第1维(每一行)表示第i个step时所有others的状态,第2维(每一列)表示一个obs的预测线); 
         cfg 配置信息(map类型);
    """
    def __init__(self, ego: VehicleBase, others: List[VehicleBase],
                 other_traj: List[StateList], cfg: dict = {}):
        self.ego_vehicle: VehicleBase = ego
        self.other_vehicle: VehicleBase = others
        #数据结构见上面描述;作用:当对某个agent进行MCTS到tree的第i层时,其实就是agent的第i步预测.因此可以通过self.other_predict_traj[i]获取第i步其它所有对手agents的状态
        self.other_predict_traj: StateList = other_traj 

        self.computation_budget = cfg['computation_budget'] # 15000 (完成一次MCTS搜索的迭代次数)
        self.dt = cfg['delta_t']                            # 0.25 (时间步长)

    """
    输入: root 该root.state为ego.state, root.goal为ego.target
    输出: root的最优子节点
    功能: MCTS的调用接口,完成从根节点root开始的整个MCTS过程,并返回root的最优子节点(即下一个时间步的最优节点)
         MCTS的Selection,Expansion,Simulation,Backpropagation最多迭代computation_budget次.
    """
    def excute(self, root: Node) -> Node:
        for _ in range(self.computation_budget):
            # 1. Find the best node to expand (Selection,Expansion)(见论文中的tree policy部分)
            expand_node = self.tree_policy(root)
            # 2. Random run to add node and get reward (Simulation)(见论文中的default policy部分)
            reward = self.default_policy(expand_node)
            # 3. Update all passing nodes with reward (Backpropagation)
            self.update(expand_node, reward)

        return self.get_best_child(root, 0)

    """
    输入: node 根节点(每次tree_policy都是从根节点开始)
    输出: node 待simulation节点(该节点只有2中情况:新expand的节点;tree最后一层(is_terminal)的最优叶子结点)
    功能: 通过UCB与0.5的概率来Selection,Expansion.获取最终需要simulation的节点并返回(见论文中的tree policy部分)。
         (注:除了用UCB来平衡exploitation,exploration之外;还对not fully expanded的节点用0.5的概率来拓展或者UCB选择。)
    """
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
    输入: node 由tree policy处理得到的待simulation节点
    输出: 从待simulation节点到simulation结束节点的总代价(用于下一步Backpropagation)
    功能: Simulation,从node开始,随机选择动作生成next_node,直到触发simulation结束。
    (注: simulation生成next_node不是为了生成树,因此不用记录父子节点的连接关系.)
    """
    def default_policy(self, node: Node) -> float:
        # terminal的条件是到达了MCTS的最大层MAX_LEVEL
        while node.is_terminal == False:
            # 获取其他所有agents在MCTS树中第"node.cur_level + 1"层(即第"node.cur_level + 1"个时间步)的状态
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
    输入: node 当前待计算value的节点, 
         last_node_value node的父节点的value值(该value值是从simulation节点到该节点的总value值,最终用途是MCTS的proppagation);
    输出: total_reward 即算出的node的value
    功能: 用于MCTS的simulation阶段每个节点的value值的计算。参照论文中"stage reward function(式3)"的定义。
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
        self.steps = cfg['max_step']    # 配置文件中为 8(每个时间步的预测步数,即预测线的长度)
        self.dt = cfg['delta_t']        # 配置文件中为 0.25
        self.config = cfg               # unprotected_left_turn.yaml

    """ 
    输入: ego 当前的agent(不一定是ego); others 表示其他车辆的列表(除了当前agent外其他所有对手 agents);
    输出: act agent当前时间步的最优动作序列; excepted_traj 对应最优轨迹(即预测轨迹)。
    功能: 车辆执行一次完整level-k规划, 返回当前该执行的最优动作和预期轨迹。
    """
    def planning(self, ego: VehicleBase, others: List[VehicleBase]) -> Tuple[utils.Action, StateList]:
        # other_prediction 类型为 List[StateList] :记录MCTS中每个 level 时其它 agents 的状态(区别于传统的预测轨迹)
        # print(f"/n  planning. name:{ego.name}, level{ego.level}")
        other_prediction = self.get_prediction(ego, others)
        actions, traj = self.forward_simulate(ego, others, other_prediction) # 使用MCTS前向仿真，生成自车最优动作和最优状态序列(即预期轨迹)

        return actions[0], traj

    """
    输入: ego 当前agent; others 除ego外其他所有对手agents; 
         traj 除ego外其他所有对手agents的预测轨迹(类型List[StateList],二维.第1维(每一行)表示第i个step时所有others的状态,第2维(每一列)表示一个obs的预测线); 
    输出: actions 当前agent的最优动作序列, expected_traj 当前agent的最优状态序列(与actions对应)
    功能: 从ego.state(agent当前状态)开始,ego.target为目标,考虑与others的交互,使用MCTS搜索出最优动作序列actions和最优状态序列expected_traj并返回。
         且根据论文可知MCTS的Reward function部分重复考虑了collision,safe dis,off-road,bet line,speed,yaw,decel因素。
         (注1:对每个agent的推理都会用到forward_simulate函数,因此形参ego只是指当前agent,并不一定是自车。
          注2:forward_simulate中最主要的就是MCTS过程,belief update操作理论上应该在level-k中进行,不在MCTS中进行。)
    """
    def forward_simulate(self, ego: VehicleBase, others: List[VehicleBase],
                         traj: List[StateList]) -> Tuple[List[utils.Action], StateList]:
        # 输入ego,others的vehicle对象(ego,others);others在MCTS中每个level的预测状态(traj); 进行一次完整的MCTS搜索
        mcts = MonteCarloTreeSearch(ego, others, traj, self.config)
        current_node = Node(state = ego.state, goal = ego.target)
        current_node = mcts.excute(current_node)    # 从current_node开始执行MCTS,return结果为下一个时间步的最优节点

        # 从下一个时间步最优节点开始,搜索最优动作序列与最优转态序列(预测). 
        # 注:搜索最终结果的过程也是一个tree policy的过程找到最优叶子节点,只不过该过程设置scalar为0(只考虑利用,不考虑探索)
        for _ in range(Node.MAX_LEVEL - 1):
            current_node = mcts.get_best_child(current_node, 0) #只考虑exploitation找到MCT最优子节点

        actions = current_node.actions  #最后一层的最优子节点上，已经记录了从根节点到该叶节点所有的动作列表了
        # 从下一个时间步最优节点开始先逆序获取状态列表再reverse，找到从根节点到最优叶子节点的最优状态序列，即为ego的预测 expected_traj
        state_list = StateList()
        while current_node != None:
            state_list.append(current_node.state)
            current_node = current_node.parent
        expected_traj = state_list.reverse()

        # 如果MCTS中获取最优状态序列长度小于预测轨迹长度,就用最后一个状态补足。(因mcts最大深度为Node.MAX_LEVEL，而预测长度为配置文件中的max_step)
        if len(expected_traj) < self.steps + 1:
            logging.debug(f"The max level of the node is not enough({len(expected_traj)}),"
                          f"using the last value to complete it.")
            expected_traj.expand(self.steps + 1)

        return actions, expected_traj

    """
    输入: ego 当前的agent(不一定是ego); others 表示其他车辆的列表(除了当前agent外其他所有对手 agents);
    输出: pred_trajectory 类型为List[StateList],即二维矩阵.
         第1维(每一行)表示第i个step时所有others的状态,第2维(每一列)表示一个obs的预测线。(预测线的总长度就是self.steps,配置文件中为8)
    功能: get_prediction()函数中主要实现了level-k gaming的变量递归调用的逻辑,递归的结束条件有2个:(1)就是当agent到达level-0;(2)就是当所有others的最优轨迹都计算出来后。
         forward_simulate()的主要功能就是当所有others的最优轨迹(预测)都生成之后,用MCTS来计算ego的最优动作序列与最优状态序列(预测)。
         (注1:belief update操作理论上应该在level-k中进行,但本工程并未实现该功能。)
    """
    def get_prediction(self, ego: VehicleBase, others: List[VehicleBase]) -> List[StateList]:
        pred_trajectory = [] # 返回值:类型为List[StateList],即二维矩阵.第1维(每一行)表示第i个step时所有others的状态,第2维(每一列)表示一个obs的预测线。
        pred_trajectory_trans = []  # 类型List[StateList],第1维(每一行)表示一个obs的预测线。(结构与返回值 pred_trajectory 互为转置)
        # print(f"prediction. name:{ego.name}, level{ego.level}")

        if ego.level == 0:
            #递归结束条件: 当ego为level-0时,其它agents都认为是静态的.因此对others的预测轨迹都是其当前状态。
            for i in range(self.steps + 1):
                pred_traj: StateList = StateList()
                for other in others:
                    pred_traj.append(other.state)
                pred_trajectory.append(pred_traj)
            return pred_trajectory
        elif ego.level > 0:
            for idx in range(len(others)):      # 遍历每辆 others
                if others[idx].is_get_target:
                    # 如果该other已经到达终点，则其预测线中的每个状态都是other[idx]的当前状态(也是其终点状态).
                    pred_traj: StateList = StateList()
                    for i in range(self.steps + 1):
                        pred_traj.append(others[idx].state)
                    pred_trajectory_trans.append(pred_traj)
                    continue
                # 交换角色,降低层级,递归预测,前向仿真(递归模拟其他车辆在level-k-1的交互,最终到level-0时结果才是确定的,才能往>0的level反推。)
                exchanged_ego: VehicleBase = others[idx]
                exchanged_ego.level = ego.level - 1 # 先只把与ego交互的第idx这1个other的level变为ego.level-1,其它所有agents的level不变,执行下一次递归调用get_prediction()
                exchanged_others: List[VehicleBase] = [ego]
                for i in range(len(others)):
                    if i != idx:
                        exchanged_others.append(others[i])
                exchage_pred_others = self.get_prediction(exchanged_ego, exchanged_others)
                # forward_simulate使用MCTS得到exchanged_ego的 当前agent的最优动作序列 _, 最优状态序列 pred_idx_vechicle(与动作序列对应)
                _, pred_idx_vechicle = self.forward_simulate(exchanged_ego, exchanged_others, exchage_pred_others)
                pred_trajectory_trans.append(pred_idx_vechicle)
        else:
            # level-k的level异常
            logging.error("get_prediction() excute error, the level must be >= 0 and < 3 !")
            return pred_trajectory

        # 将 pred_trajectory_trans 转置,得到 pred_trajectory(即重新组织数据结构,并返回)
        for i in range(self.steps + 1):
            state = StateList()
            for states in pred_trajectory_trans:
                state.append(states[i])
            pred_trajectory.append(state)

        return pred_trajectory
