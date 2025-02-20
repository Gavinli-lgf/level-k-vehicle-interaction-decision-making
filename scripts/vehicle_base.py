'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-04-27 16:17:27
LastEditTime: 2024-10-31 01:01:31
FilePath: /vehicle-interaction-decision-making/scripts/vehicle_base.py
Copyright 2024 puyu, All Rights Reserved.
'''

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from utils import State
from env import EnvCrossroads

"""
抽象基类，用于定义车辆的基本属性和方法: 它提供了车辆的尺寸、安全区、环境等基本信息，以及一些静态方法来计算车辆的二维边界和安全区。
"""
class VehicleBase(ABC):
    # 定义车辆、安全区的尺寸参数
    length = 5
    width = 2
    safe_length = 8
    safe_width = 2.4
    env: Optional[EnvCrossroads] = None

    # 初始化车辆的名称name与状态state
    def __init__(self, name: str):
        self.name: str = name
        self.state: State = State()

    # 输入:目标偏移tar_offset; 输出:车辆的二维边界 vehicle,类型为 np.ndarray; 
    @staticmethod
    def get_box2d(tar_offset: State) -> np.ndarray:
        vehicle = np.array(
            [[-VehicleBase.length/2, VehicleBase.length/2,
              VehicleBase.length/2, -VehicleBase.length/2, -VehicleBase.length/2],
            [VehicleBase.width/2, VehicleBase.width/2,
             -VehicleBase.width/2, -VehicleBase.width/2, VehicleBase.width/2]]
        )
        rot = np.array([[np.cos(tar_offset.yaw), -np.sin(tar_offset.yaw)],
                     [np.sin(tar_offset.yaw), np.cos(tar_offset.yaw)]])

        vehicle = np.dot(rot, vehicle)
        vehicle += np.array([[tar_offset.x], [tar_offset.y]])

        return vehicle

    # 输入:目标偏移tar_offset; 输出:车辆的安全区 safezone,类型为 np.ndarray;
    @staticmethod
    def get_safezone(tar_offset: State) -> np.ndarray:
        safezone = np.array(
            [[-VehicleBase.safe_length/2, VehicleBase.safe_length/2,
              VehicleBase.safe_length/2, -VehicleBase.safe_length/2, -VehicleBase.safe_length/2],
            [VehicleBase.safe_width/2, VehicleBase.safe_width/2,
             -VehicleBase.safe_width/2, -VehicleBase.safe_width/2, VehicleBase.safe_width/2]]
        )
        rot = np.array([[np.cos(tar_offset.yaw), -np.sin(tar_offset.yaw)],
                     [np.sin(tar_offset.yaw), np.cos(tar_offset.yaw)]])

        safezone = np.dot(rot, safezone)
        safezone += np.array([[tar_offset.x], [tar_offset.y]])

        return safezone

    # 初始化类中的车辆、安全区的尺寸参数:初始化类属性 env, length, width, safe_length, safe_width
    @staticmethod
    def initialize(env: EnvCrossroads, len: float, width: float,
                   safe_len: float, safe_width: float):
        VehicleBase.env = env
        VehicleBase.length = len
        VehicleBase.width = width
        VehicleBase.safe_length = safe_len
        VehicleBase.safe_width = safe_width
