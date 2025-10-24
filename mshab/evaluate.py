# 导入必要的Python库
import json  # 处理JSON数据格式
import random  # 生成随机数
import sys  # 系统相关操作（如命令行参数）
import time  # 时间相关功能
from collections import defaultdict, deque  # 高级数据结构（带默认值的字典和双端队列）
from dataclasses import asdict, dataclass, field  # 数据类处理
from pathlib import Path  # 面向对象的文件路径操作
from typing import TYPE_CHECKING  # 类型检查相关

# 第三方库
from dacite import from_dict  # 字典转数据类
from omegaconf import OmegaConf  # 配置管理
from tqdm import tqdm  # 进度条显示

# Gymnasium相关
from gymnasium import spaces  # 强化学习空间定义

import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架

# ManiSkill特定导入
import mani_skill.envs  # ManiSkill环境注册
from mani_skill import ASSET_DIR  # 资源目录路径
from mani_skill.utils import common  # 通用工具函数

# 自定义模块导入
from mshab.agents.bc import Agent as BCAgent  # BC算法代理
from mshab.agents.act import Agent as ACTAgent  # ACT算法代理
from mshab.agents.dp import Agent as DPAgent  # 扩散策略代理
from mshab.agents.brs import Agent as BRSAgent # BRS算法代理
from mshab.agents.ppo import Agent as PPOAgent  # PPO算法代理
from mshab.agents.sac import Agent as SACAgent  # SAC算法代理
from mshab.envs.make import EnvConfig, make_env  # 环境创建配置和函数
from mshab.envs.planner import CloseSubtask, OpenSubtask, PickSubtask, PlaceSubtask  # 子任务类型
from mshab.utils.array import recursive_slice, to_tensor  # 数组处理工具
from mshab.utils.config import parse_cfg  # 配置解析
from mshab.utils.logger import Logger, LoggerConfig  # 日志记录
from mshab.utils.time import NonOverlappingTimeProfiler  # 时间分析器

# 类型检查专用导入
if TYPE_CHECKING:
    from mshab.envs import SequentialTaskEnv  # 顺序任务环境类型提示

# 定义策略类型与任务/子任务的映射关系
POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS = dict(
    # BC策略配置
    bc_placed_500=dict(
        prepare_groceries=dict(
            place=["all"],  # 准备食材任务中放置子任务的目标对象
        ),
    ),
    bc_dropped_500=dict(
        prepare_groceries=dict(
            place=["all"],  # 准备食材任务中放置子任务的目标对象
        ),
    ),
    bc_placed_dropped_500=dict(
        prepare_groceries=dict(
            place=["all"],  # 准备食材任务中放置子任务的目标对象
        ),
    ),
    bc=dict(
        tidy_house=dict(
            pick=["all"],  # 整理房间任务中拾取子任务的目标对象
            #place=["all"],  # 整理房间任务中放置子任务的目标对象
        ),
        prepare_groceries=dict(
            pick=["all"],  # 准备食材任务中拾取子任务的目标对象
            place=["all"],  # 准备食材任务中放置子任务的目标对象
        ),
        set_table=dict(
            pick=["all"],  # 摆桌子任务中拾取子任务的目标对象
            place=["all"],  # 摆桌子任务中放置子任务的目标对象
            open=["fridge", "kitchen_counter"],  # 摆桌子任务中打开子任务的目标对象
            close=["fridge", "kitchen_counter"],  # 摆桌子任务中关闭子任务的目标对象
        ),
    ),
    # 扩散策略配置
    dp=dict(
        tidy_house=dict(
            pick=["all"],  # 整理房间任务中拾取子任务的目标对象
            place=["all"],  # 整理房间任务中放置子任务的目标对象
        ),
        prepare_groceries=dict(
            pick=["all"],  # 准备食材任务中拾取子任务的目标对象
            place=["all"],  # 准备食材任务中放置子任务的目标对象
        ),
        set_table=dict(
            pick=["all"],  # 摆桌子任务中拾取子任务的目标对象
            place=["all"],  # 摆桌子任务中放置子任务的目标对象
            open=["fridge", "kitchen_counter"],  # 摆桌子任务中打开子任务的目标对象
            close=["fridge", "kitchen_counter"],  # 摆桌子任务中关闭子任务的目标对象
        ),
    ),
    # 强化学习策略配置
    rl=dict(
        tidy_house=dict(
            pick=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 整理房间任务中拾取子任务的目标对象列表
            place=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 整理房间任务中放置子任务的目标对象列表
            navigate=["all"],  # 整理房间任务中导航子任务的目标对象
        ),
        prepare_groceries=dict(
            pick=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 准备食材任务中拾取子任务的目标对象列表
            place=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 准备食材任务中放置子任务的目标对象列表
            navigate=["all"],  # 准备食材任务中导航子任务的目标对象
        ),
        set_table=dict(
            pick=["013_apple", "024_bowl", "all"],  # 摆桌子任务中拾取子任务的目标对象列表
            place=["013_apple", "024_bowl", "all"],  # 摆桌子任务中放置子任务的目标对象列表
            navigate=["all"],  # 摆桌子任务中导航子任务的目标对象
            open=["fridge", "kitchen_counter"],  # 摆桌子任务中打开子任务的目标对象
            close=["fridge", "kitchen_counter"],  # 摆桌子任务中关闭子任务的目标对象
        ),
    ),

    # ACT算法配置
    act=dict(
        tidy_house=dict(
            pick=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 整理房间任务中拾取子任务的目标对象列表
            place=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 整理房间任务中放置子任务的目标对象列表
            navigate=["all"],  # 整理房间任务中导航子任务的目标对象
        ),
        prepare_groceries=dict(
            pick=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 准备食材任务中拾取子任务的目标对象列表
            place=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 准备食材任务中放置子任务的目标对象列表
            navigate=["all"],  # 准备食材任务中导航子任务的目标对象
        ),
        set_table=dict(
            pick=["013_apple"],  #, "024_bowl", "all"],  # 摆桌子任务中拾取子任务的目标对象列表
            # place=["013_apple", "024_bowl", "all"],  # 摆桌子任务中放置子任务的目标对象列表
            # navigate=["all"],  # 摆桌子任务中导航子任务的目标对象
            # open=["fridge", "kitchen_counter"],  # 摆桌子任务中打开子任务的目标对象
            # close=["fridge", "kitchen_counter"],  # 摆桌子任务中关闭子任务的目标对象
        ),
    ),

    # ACT算法配置
    act_multi_head=dict(
        tidy_house=dict(
            pick=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 整理房间任务中拾取子任务的目标对象列表
            place=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 整理房间任务中放置子任务的目标对象列表
            navigate=["all"],  # 整理房间任务中导航子任务的目标对象
        ),
        prepare_groceries=dict(
            pick=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 准备食材任务中拾取子任务的目标对象列表
            place=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 准备食材任务中放置子任务的目标对象列表
            navigate=["all"],  # 准备食材任务中导航子任务的目标对象
        ),
        set_table=dict(
            # pick=["024_bowl"],  #, "024_bowl", "all"],  # 摆桌子任务中拾取子任务的目标对象列表
            # place=["024_bowl"],  # 摆桌子任务中放置子任务的目标对象列表
            # navigate=["all"],  # 摆桌子任务中导航子任务的目标对象
            # open=["fridge", "kitchen_counter"],  # 摆桌子任务中打开子任务的目标对象
            close=["kitchen_counter"],  # 摆桌子任务中关闭子任务的目标对象
        ),
    ),

    # BRS算法配置
    brs=dict(
        tidy_house=dict(
            pick=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 整理房间任务中拾取子任务的目标对象列表
            place=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 整理房间任务中放置子任务的目标对象列表
            navigate=["all"],  # 整理房间任务中导航子任务的目标对象
        ),
        prepare_groceries=dict(
            pick=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 准备食材任务中拾取子任务的目标对象列表
            place=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 准备食材任务中放置子任务的目标对象列表
            navigate=["all"],  # 准备食材任务中导航子任务的目标对象
        ),
        set_table=dict(
            # pick=["013_apple"],  #, "024_bowl", "all"],  # 摆桌子任务中拾取子任务的目标对象列表
            place=["013_apple"],  # 摆桌子任务中放置子任务的目标对象列表
            # navigate=["all"],  # 摆桌子任务中导航子任务的目标对象
            # open=["fridge", "kitchen_counter"],  # 摆桌子任务中打开子任务的目标对象
            # close=["kitchen_counter"],  # 摆桌子任务中关闭子任务的目标对象
        ),
    ),

    # BRS算法配置
    brs_without_extra=dict(
        tidy_house=dict(
            pick=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 整理房间任务中拾取子任务的目标对象列表
            place=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 整理房间任务中放置子任务的目标对象列表
            navigate=["all"],  # 整理房间任务中导航子任务的目标对象
        ),
        prepare_groceries=dict(
            pick=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 准备食材任务中拾取子任务的目标对象列表
            place=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 准备食材任务中放置子任务的目标对象列表
            navigate=["all"],  # 准备食材任务中导航子任务的目标对象
        ),
        set_table=dict(
            # pick=["013_apple"],  #, "024_bowl", "all"],  # 摆桌子任务中拾取子任务的目标对象列表
            place=["013_apple"],  # 摆桌子任务中放置子任务的目标对象列表
            # navigate=["all"],  # 摆桌子任务中导航子任务的目标对象
            # open=["fridge", "kitchen_counter"],  # 摆桌子任务中打开子任务的目标对象
            # close=["kitchen_counter"],  # 摆桌子任务中关闭子任务的目标对象
        ),
    ),

    # BRS算法配置
    brs_one_decoder=dict(
        tidy_house=dict(
            pick=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 整理房间任务中拾取子任务的目标对象列表
            place=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 整理房间任务中放置子任务的目标对象列表
            navigate=["all"],  # 整理房间任务中导航子任务的目标对象
        ),
        prepare_groceries=dict(
            pick=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 准备食材任务中拾取子任务的目标对象列表
            place=[
                "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
                "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box", 
                "009_gelatin_box", "010_potted_meat_can", "024_bowl", "all"
            ],  # 准备食材任务中放置子任务的目标对象列表
            navigate=["all"],  # 准备食材任务中导航子任务的目标对象
        ),
        set_table=dict(
            #pick=["013_apple"],  #, "024_bowl", "all"],  # 摆桌子任务中拾取子任务的目标对象列表
            place=["013_apple"],  # 摆桌子任务中放置子任务的目标对象列表
            # navigate=["all"],  # 摆桌子任务中导航子任务的目标对象
            # open=["fridge", "kitchen_counter"],  # 摆桌子任务中打开子任务的目标对象
            #close=["kitchen_counter"],  # 摆桌子任务中关闭子任务的目标对象
        ),
    ),
)

# 评估配置数据类
@dataclass
class EvalConfig:
    seed: int  # 随机种子
    task: str  # 任务名称
    eval_env: EnvConfig  # 评估环境配置
    logger: LoggerConfig  # 日志配置
    
    # 可选参数
    policy_type: str = "rl_per_obj"  # 策略类型
    max_trajectories: int = 1000  # 最大轨迹数
    save_trajectory: bool = False  # 是否保存轨迹
    
    # 后初始化字段
    policy_key: str = field(init=False)  # 策略键值（自动生成）

    def __post_init__(self):
        """配置后初始化验证"""
        # 验证任务类型有效性
        assert self.task in ["tidy_house", "prepare_groceries", "set_table"]
        # 验证任务计划文件路径有效性
        assert self.task in self.eval_env.task_plan_fp
        
        # 验证策略类型有效性
        valid_policies = ["rl_all_obj", "rl_per_obj"] + list(POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS.keys())
        assert self.policy_type in valid_policies
        
        # 提取策略键值（移除rl前缀修饰）
        self.policy_key = (
            self.policy_type.split("_")[0]
            if "rl" in self.policy_type
            else self.policy_type
        )
        
        # 配置日志记录
        self.logger.exp_cfg = asdict(self)
        # 避免日志配置中的循环引用
        del self.logger.exp_cfg["logger"]["exp_cfg"]

# 配置加载函数
def get_mshab_train_cfg(cfg: EvalConfig) -> EvalConfig:
    """将OmegaConf配置转换为EvalConfig数据类"""
    return from_dict(data_class=EvalConfig, data=OmegaConf.to_container(cfg))

# 主评估函数
def eval(cfg: EvalConfig):
    # 初始化时间分析器
    timer = NonOverlappingTimeProfiler()
    
    # 设置随机种子
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True  # 确保CUDA操作确定性

    # 设备选择（优先GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 环境初始化
    # ----------------------------
    # 创建日志记录器
    logger = Logger(
        logger_cfg=cfg.logger,
        save_fn=None,  # 无特殊保存函数
    )
    # 创建评估环境
    eval_envs = make_env(
        cfg.eval_env,
        video_path=logger.eval_video_path,  # 设置视频保存路径
    )
    # 获取底层环境对象
    uenv: SequentialTaskEnv = eval_envs.unwrapped
    # 重置环境并获取初始观测
    eval_obs, _ = eval_envs.reset(seed=cfg.seed, options=dict(reconfigure=True))
    
    # 人机交互渲染模式特殊处理
    if uenv.render_mode == "human":
        # 渲染环境
        uenv.render()

        # 保存原始回调函数引用
        _original_after_control_step = uenv._after_control_step
        _original_after_simulation_step = uenv._after_simulation_step

        # 计算每个模拟步骤的时间
        time_per_sim_step = uenv.control_timestep / uenv._sim_steps_per_control

        # 重写控制步回调函数以实现实时渲染
        def wrapped_after_control_step(self):
            """重写控制步结束回调"""
            # 调用原始回调
            _original_after_control_step()

            # 设置下一控制步的开始时间
            self._control_step_start_time = time.time()
            self._cur_sim_step = 0
            # 计算下一控制步的结束时间
            self._control_step_end_time = (
                self._control_step_start_time + self.control_timestep
            )

        # 重写模拟步回调函数以实现实时渲染
        def wrapped_after_simulation_step(self):
            """重写模拟步结束回调"""
            # 调用原始回调
            _original_after_simulation_step()
            
            # 初始化时间跟踪变量
            if getattr(self, "_control_step_start_time", None) is None:
                self._control_step_start_time = time.time()
                self._cur_sim_step = 0
                self._control_step_end_time = (
                    self._control_step_start_time + self.control_timestep
                )
                self._realtime_drift = 0  # 实时漂移量

            # 计算当前模拟步的结束时间
            step_end_time = self._control_step_start_time + (
                time_per_sim_step * (self._cur_sim_step + 1)
            )
            
            # 如果当前时间早于步结束时间，进行渲染
            if time.time() < step_end_time:
                # 如果启用了GPU模拟，获取所有GPU数据
                if self.gpu_sim_enabled:
                    self.scene._gpu_fetch_all()
                # 渲染环境
                self.render()
                # 计算需要休眠的时间
                sleep_time = step_end_time - time.time()
                # 如果休眠时间大于0，进行休眠
                if sleep_time > 0:
                    time.sleep(sleep_time)
            # 增加当前模拟步计数
            self._cur_sim_step += 1

        # 应用重写的回调函数
        uenv._after_control_step = wrapped_after_control_step.__get__(uenv)
        uenv._after_simulation_step = wrapped_after_simulation_step.__get__(uenv)

    # ----------------------------
    # 空间定义
    # ----------------------------
    # 获取单个环境的观测空间
    obs_space = uenv.single_observation_space
    # 获取单个环境的动作空间
    act_space = uenv.single_action_space

    # ----------------------------
    # 代理初始化
    # ----------------------------
    # 扩散策略动作历史缓存
    dp_action_history = deque([])
    # BRS算法动作历史缓存
    brs_action_history = deque([])
    # ACT算法
    ts = 0  # 时间步计数器
    actions_to_take = None  # 待执行动作序列
    all_time_actions = None  # 全时动作表
    exp_weights = None  # 指数权重变量
    def act_init():
        nonlocal ts, actions_to_take, all_time_actions, exp_weights
        ts = 0
        if algo_cfg.temporal_agg:
            all_time_actions = torch.zeros([cfg.eval_env.num_envs, cfg.eval_env.max_episode_steps, 
                cfg.eval_env.max_episode_steps+algo_cfg.num_queries, eval_envs.single_action_space.shape[0]], device=device)
            exp_weights = None
        else:
            actions_to_take = torch.zeros([cfg.eval_env.num_envs, algo_cfg.num_queries, eval_envs.single_action_space.shape[0]], device=device)

    #算法配置文件
    algo_cfg = None
    # 策略加载函数
    def get_policy_act_fn(algo_cfg_path, algo_ckpt_path):
        """根据配置和检查点加载策略并返回动作函数"""
        nonlocal algo_cfg  # 使用外部作用域的algo_cfg变量
        # 解析算法配置
        algo_cfg = parse_cfg(default_cfg_path=algo_cfg_path).algo
        
        # PPO策略
        if algo_cfg.name == "ppo":
            # 创建PPO代理
            policy = PPOAgent(eval_obs, act_space.shape)
            # 设置为评估模式
            policy.eval()
            # 加载模型权重
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            # 将策略移至指定设备
            policy.to(device)
            # 定义动作函数（使用确定性策略）
            policy_act_fn = lambda obs: policy.get_action(obs, deterministic=True)
        
        # SAC策略
        elif algo_cfg.name == "sac":
            # 获取像素观测空间
            pixels_obs_space: spaces.Dict = obs_space["pixels"]
            # 获取状态观测空间
            state_obs_space: spaces.Box = obs_space["state"]
            # 创建模型像素观测空间
            model_pixel_obs_space = dict()
            # 处理每个像素观测空间
            for k, space in pixels_obs_space.items():
                shape, low, high, dtype = (
                    space.shape,
                    space.low,
                    space.high,
                    space.dtype,
                )
                # 如果形状是4维（通常是时间×通道×高度×宽度）
                if len(shape) == 4:
                    # 重塑为通道×高度×宽度
                    shape = (shape[0] * shape[1], shape[-2], shape[-1])
                    # 重塑低值边界
                    low = low.reshape((-1, *low.shape[-2:]))
                    # 重塑高值边界
                    high = high.reshape((-1, *high.shape[-2:]))
                # 添加到模型像素观测空间
                model_pixel_obs_space[k] = spaces.Box(low, high, shape, dtype)
            # 创建字典形式的模型像素观测空间
            model_pixel_obs_space = spaces.Dict(model_pixel_obs_space)
            
            # 创建SAC代理
            policy = SACAgent(
                model_pixel_obs_space,
                state_obs_space.shape,
                act_space.shape,
                actor_hidden_dims=list(algo_cfg.actor_hidden_dims),
                critic_hidden_dims=list(algo_cfg.critic_hidden_dims),
                critic_layer_norm=algo_cfg.critic_layer_norm,
                critic_dropout=algo_cfg.critic_dropout,
                encoder_pixels_feature_dim=algo_cfg.encoder_pixels_feature_dim,
                encoder_state_feature_dim=algo_cfg.encoder_state_feature_dim,
                cnn_features=list(algo_cfg.cnn_features),
                cnn_filters=list(algo_cfg.cnn_filters),
                cnn_strides=list(algo_cfg.cnn_strides),
                cnn_padding=algo_cfg.cnn_padding,
                log_std_min=algo_cfg.actor_log_std_min,
                log_std_max=algo_cfg.actor_log_std_max,
                device=device,
            )
            # 设置为评估模式
            policy.eval()
            # 加载模型权重
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            # 将策略移至指定设备
            policy.to(device)
            # 定义动作函数（使用演员网络）
            policy_act_fn = lambda obs: policy.actor(
                obs["pixels"],
                obs["state"],
                compute_pi=False,
                compute_log_pi=False,
            )[0]
        
        # BC策略
        elif algo_cfg.name == "bc":
            # 创建BC代理
            policy = BCAgent(eval_obs, act_space.shape)
            # 设置为评估模式
            policy.eval()
            # 加载模型权重
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            # 将策略移至指定设备
            policy.to(device)
            # 定义动作函数
            policy_act_fn = lambda obs: policy(obs)
        
        # 扩散策略
        elif algo_cfg.name == "diffusion_policy":
            # 验证环境配置
            assert cfg.eval_env.continuous_task
            assert cfg.eval_env.stack is not None and cfg.eval_env.frame_stack is None
            
            # 创建扩散策略代理
            policy = DPAgent(
                single_observation_space=obs_space,
                single_action_space=act_space,
                obs_horizon=algo_cfg.obs_horizon,
                act_horizon=algo_cfg.act_horizon,
                pred_horizon=algo_cfg.pred_horizon,
                diffusion_step_embed_dim=algo_cfg.diffusion_step_embed_dim,
                unet_dims=algo_cfg.unet_dims,
                n_groups=algo_cfg.n_groups,
                device=device,
            )
            # 设置为评估模式
            policy.eval()
            # 加载模型权重
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            # 将策略移至指定设备
            policy.to(device)

            # 扩散策略专用动作函数
            def get_dp_act(obs):
                """管理动作历史并返回下一个动作"""
                # 如果动作历史为空，生成新动作序列
                if len(dp_action_history) == 0:
                    # 获取动作序列并转置维度
                    actions = policy.get_action(obs)
                    # 将动作序列添加到历史队列
                    dp_action_history.extend(actions.transpose(0, 1)) # ?
                # 返回队列中的第一个动作
                return dp_action_history.popleft()
            
            # 使用专用动作函数
            policy_act_fn = get_dp_act

        elif algo_cfg.name == "brs" or algo_cfg.name == "brs_one_decoder" or algo_cfg.name == "brs_without_extra":
            # 验证环境配置
            assert cfg.eval_env.continuous_task
            assert cfg.eval_env.stack is not None and cfg.eval_env.frame_stack is None
            
            # 创建扩散策略代理
            policy = BRSAgent(eval_envs, algo_cfg)
            # 设置为评估模式
            policy.eval()
            # 加载模型权重
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            # 将策略移至指定设备
            policy.to(device)

            # BRS算法动作函数
            def get_brs_act(obs):
                """管理动作历史并返回下一个动作"""
                # 如果动作历史为空，生成新动作序列
                if len(brs_action_history) == 0:
                    # 获取动作序列并转置维度
                    actions = policy.get_action(obs)
                    # 将动作序列添加到历史队列
                    brs_action_history.extend(actions.transpose(0, 1))
                # 返回队列中的第一个动作
                return brs_action_history.popleft()
            
            # 使用专用动作函数
            policy_act_fn = get_brs_act

        # ACT算法    
        elif algo_cfg.name == "act" or algo_cfg.name == "act_multi_head":
            policy = ACTAgent(eval_envs, algo_cfg)
            # 设置为评估模式
            policy.eval()
            # 加载模型权重
            policy.load_state_dict(
                torch.load(algo_ckpt_path, map_location=device)["agent"]
            )
            # 将策略移至指定设备
            policy.to(device)

            # 初始化时间聚合机制
            if algo_cfg.temporal_agg:
                # 时间聚合模式下每步查询一次策略
                query_frequency = 1
            else:
                # 非时间聚合模式下按查询频率更新
                query_frequency = algo_cfg.num_queries
            act_init()
            def get_act(obs):
                nonlocal ts, actions_to_take, all_time_actions, exp_weights
                if ts % query_frequency == 0:
                    # 获取动作序列（num_queries步的动作）
                    action_seq = policy.get_action(obs)
                # 时间聚合模式处理
                if algo_cfg.temporal_agg:
                    # 确保查询频率为1
                    assert query_frequency == 1, "query_frequency != 1 has not been implemented for temporal_agg==1."
                    # 将新动作序列存入全时动作表
                    all_time_actions[:, ts, ts:ts+algo_cfg.num_queries] = action_seq
                    # 提取当前时间步的所有历史动作
                    actions_for_curr_step = all_time_actions[:, :, ts]
                    # 创建动作填充状态掩码
                    actions_populated = torch.zeros(cfg.eval_env.max_episode_steps, dtype=torch.bool, device=device)
                    # 标记有效动作范围
                    actions_populated[max(0, ts + 1 - algo_cfg.num_queries):ts+1] = True
                    # 过滤出有效动作
                    actions_for_curr_step = actions_for_curr_step[:, actions_populated]
                    
                    # 设置时间衰减权重
                    k = 0.01
                    if ts < algo_cfg.num_queries:
                        # 计算指数权重
                        exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step[0]), device=device))
                        # 归一化权重
                        exp_weights = exp_weights / exp_weights.sum()
                        # 扩展权重到所有环境
                        exp_weights = torch.tile(exp_weights, (cfg.eval_env.num_envs, 1))
                        # 增加维度用于广播
                        exp_weights = torch.unsqueeze(exp_weights, -1)
                    
                    # 计算加权平均动作
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=1)
                else:
                    # 非时间聚合模式：按索引选择当前动作
                    if ts % query_frequency == 0:
                        actions_to_take = action_seq
                    raw_action = actions_to_take[:, ts % query_frequency]
                ts += 1  # 增加时间步计数器
                if ts >= cfg.eval_env.max_episode_steps:
                    act_init()  # 重置时间步和动作序列
                return raw_action
            
            policy_act_fn = get_act

        else:
            # 不支持算法类型
            raise NotImplementedError(f"algo {algo_cfg.name} not supported")
        
        # 预热策略（执行一次推理）
        policy_act_fn(to_tensor(eval_obs, device=device, dtype="float"))
        # 返回动作函数
        return policy_act_fn

    # 策略存储路径
    # mshab_ckpt_dir = ASSET_DIR / "mshab_checkpoints"
    # # 如果默认路径不存在，使用备用路径
    # if not mshab_ckpt_dir.exists():
    #     mshab_ckpt_dir = Path("mshab_checkpoints")
    mshab_ckpt_dir = Path("/raid/ljh/mshab/mshab_checkpoints")

    # 加载所有策略
    policies = dict()
    # 遍历配置中的策略类型、任务和子任务
    for subtask_name, subtask_targs in POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS[cfg.policy_key][cfg.task].items():
        # 初始化子任务策略字典
        policies[subtask_name] = dict()
        # 遍历目标对象
        for targ_name in subtask_targs:
            # 构建配置文件路径
            cfg_path = (
                mshab_ckpt_dir
                / cfg.policy_key
                / cfg.task
                / subtask_name
                / targ_name
                / "config.yml"
            )
            # 构建模型检查点路径
            ckpt_path = (
                mshab_ckpt_dir
                / cfg.policy_key
                / cfg.task
                / subtask_name
                / targ_name
                / "policy.pt"
            )
            # 加载策略并存储
            policies[subtask_name][targ_name] = get_policy_act_fn(cfg_path, ckpt_path)

    # 综合动作函数
    def act(obs):
        """根据当前状态选择动作"""
        with torch.no_grad():
            with torch.device(device):
                # 初始化动作张量（全零）
                action = torch.zeros(eval_envs.num_envs, *act_space.shape)

                # 获取当前子任务指针（索引）
                subtask_pointer = uenv.subtask_pointer.clone()
                # 定义获取子任务类型的函数
                get_subtask_type = lambda: uenv.task_ids[
                    torch.clip(
                        subtask_pointer,
                        max=len(uenv.task_plan) - 1,
                    )
                ]
                # 获取当前子任务类型
                subtask_type = get_subtask_type()

                # 根据子任务类型分类环境索引
                pick_env_idx = subtask_type == 0  # 拾取任务环境索引
                place_env_idx = subtask_type == 1  # 放置任务环境索引
                navigate_env_idx = subtask_type == 2  # 导航任务环境索引
                open_env_idx = subtask_type == 3  # 打开任务环境索引
                close_env_idx = subtask_type == 4  # 关闭任务环境索引

                # 获取每个环境的Sapien对象名称
                sapien_obj_names = [None] * uenv.num_envs
                # 遍历所有环境
                for env_num, subtask_num in enumerate(
                    torch.clip(subtask_pointer, max=len(uenv.task_plan) - 1)
                ):
                    # 获取当前子任务
                    subtask = uenv.task_plan[subtask_num]
                    # 如果是拾取或放置子任务
                    if isinstance(subtask, PickSubtask) or isinstance(
                        subtask, PlaceSubtask
                    ):
                        # 获取对象名称
                        sapien_obj_names[env_num] = (
                            uenv.subtask_objs[subtask_num]._objs[env_num].name
                        )
                    # 如果是打开或关闭子任务
                    elif isinstance(subtask, OpenSubtask) or isinstance(
                        subtask, CloseSubtask
                    ):
                        # 获取对象名称
                        sapien_obj_names[env_num] = (
                            uenv.subtask_articulations[subtask_num]._objs[env_num].name
                        )
                
                # 获取目标名称
                targ_names = []
                # 遍历所有Sapien对象名称
                for sapien_on in sapien_obj_names:
                    if sapien_on is None:
                        # 无目标对象
                        targ_names.append(None)
                    else:
                        # 在任务目标名称中查找匹配项
                        matched = False
                        for tn in task_targ_names:
                            if tn in sapien_on:
                                targ_names.append(tn)
                                matched = True
                                break
                        if not matched:
                            targ_names.append(None)
                # 验证目标名称数量
                assert len(targ_names) == uenv.num_envs

                # 如果策略类型为"rl_per_obj"或进行打开/关闭任务，需要按对象查询策略
                if (
                    cfg.policy_type == "rl_per_obj"
                    or torch.any(open_env_idx)
                    or torch.any(close_env_idx)
                ):
                    # 初始化目标名称到环境索引的映射
                    tn_env_idxs = dict()
                    # 遍历所有环境和目标名称
                    for env_num, tn in enumerate(targ_names):
                        if tn not in tn_env_idxs:
                            tn_env_idxs[tn] = []
                        tn_env_idxs[tn].append(env_num)
                    # 转换为布尔索引张量
                    for k, v in tn_env_idxs.items():
                        bool_env_idx = torch.zeros(uenv.num_envs, dtype=torch.bool)
                        bool_env_idx[v] = True
                        tn_env_idxs[k] = bool_env_idx

                # 设置子任务目标策略动作
                def set_subtask_targ_policy_act(subtask_name, subtask_env_idx):
                    """为指定子任务设置策略动作"""
                    # 如果是按对象策略或打开/关闭任务（导航除外）
                    if (
                        cfg.policy_type == "rl_per_obj"
                        or subtask_name
                        in [
                            "open",
                            "close",
                        ]
                    ) and subtask_name != "navigate":
                        # 遍历目标名称
                        for tn, targ_env_idx in tn_env_idxs.items():
                            # 计算当前子任务和目标的交集环境索引
                            subtask_targ_env_idx = subtask_env_idx & targ_env_idx
                            # 如果有环境需要处理
                            if torch.any(subtask_targ_env_idx):
                                # 切片观测
                                sliced_obs = recursive_slice(obs, subtask_targ_env_idx)
                                # 获取动作
                                action[subtask_targ_env_idx] = policies[subtask_name][
                                    tn
                                ](sliced_obs)
                    else:
                        # 切片观测
                        sliced_obs = recursive_slice(obs, subtask_env_idx)
                        # 获取动作 # 当更换小任务时，这里需要额外修改
                        action[subtask_env_idx] = policies[subtask_name]["024_bowl"](
                            sliced_obs
                        )

                # 根据子任务类型调用相应策略
                if torch.any(pick_env_idx):
                    set_subtask_targ_policy_act("pick", pick_env_idx)
                if torch.any(place_env_idx):
                    set_subtask_targ_policy_act("place", place_env_idx)
                if torch.any(navigate_env_idx):
                    set_subtask_targ_policy_act("navigate", navigate_env_idx)
                if torch.any(open_env_idx):
                    set_subtask_targ_policy_act("open", open_env_idx)
                if torch.any(close_env_idx):
                    set_subtask_targ_policy_act("close", close_env_idx)

                # 返回动作
                return action

    # ----------------------------
    # 评估运行
    # ----------------------------
    # 初始化任务目标名称集合
    task_targ_names = set()
    # 遍历RL策略配置中的子任务
    for subtask_name in POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS["rl"][cfg.task]:
        # 添加目标名称到集合
        task_targ_names.update(
            POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS["rl"][cfg.task][subtask_name]
        )

    # 重置环境并获取初始观测
    eval_obs = to_tensor(
        eval_envs.reset(seed=cfg.seed)[0], device=device, dtype="float"
    )
    # 初始化子任务失败计数器
    subtask_fail_counts = defaultdict(int)
    # 保存当前子任务指针
    last_subtask_pointer = uenv.subtask_pointer.clone()
    # 创建进度条
    pbar = tqdm(range(cfg.max_trajectories), total=cfg.max_trajectories)
    # 初始化步数计数器
    step_num = 0

    # 终止条件检查函数
    def check_done():
        """检查评估是否完成"""
        if cfg.save_trajectory:
            # 如果保存轨迹，检查是否达到最大轨迹数
            return eval_envs.env._env.reached_max_trajectories
        # 否则检查返回队列中的轨迹数
        return len(eval_envs.return_queue) >= cfg.max_trajectories

    # 进度更新函数
    def update_pbar(step_num):
        """更新进度条"""
        if cfg.save_trajectory:
            # 计算已保存轨迹的增量
            diff = eval_envs.env._env.num_saved_trajectories - pbar.last_print_n
        else:
            # 计算返回队列中的轨迹增量
            diff = len(eval_envs.return_queue) - pbar.last_print_n

        # 如果有增量，更新进度条
        if diff > 0:
            pbar.update(diff)

        # 更新进度条描述
        pbar.set_description(f"{step_num=}")

    # 失败计数更新函数
    def update_fail_subtask_counts(done):
        """更新子任务失败计数"""
        if torch.any(done):
            # 获取失败子任务编号
            subtask_nums = last_subtask_pointer[done]
            # 统计各子任务失败次数
            unique_subtasks, counts = np.unique(
                subtask_nums.cpu().numpy(), return_counts=True
            )
            for fail_subtask, num_envs in zip(unique_subtasks, counts):
                subtask_fail_counts[fail_subtask] += num_envs
            # 保存失败计数到JSON文件
            with open(logger.exp_path / "subtask_fail_counts.json", "w+") as f:
                json.dump(
                    dict(
                        (str(k), int(subtask_fail_counts[k]))
                        for k in sorted(subtask_fail_counts.keys())
                    ),
                    f,
                )

    # 主评估循环
    while not check_done():
        # 结束"other"计时段
        timer.end("other")
        # 保存当前子任务指针
        last_subtask_pointer = uenv.subtask_pointer.clone()
        # 获取动作
        action = act(eval_obs)
        # 结束"sample"计时段
        timer.end("sample")
        # 执行动作
        eval_obs, _, term, trunc, _ = eval_envs.step(action)
        # 结束"sim_sample"计时段
        timer.end("sim_sample")
        # 转换观测为张量
        eval_obs = to_tensor(
            eval_obs,
            device=device,
            dtype="float",
        )
        # 更新进度条
        update_pbar(step_num)
        # 更新失败计数
        update_fail_subtask_counts(term | trunc)
        # 如果是扩散策略且有环境终止/截断，清空动作历史
        if cfg.policy_key == "dp":
            if torch.any(term | trunc):
                dp_action_history.clear()
        if cfg.policy_key == "brs" or cfg.policy_key == "brs_one_decoder" or cfg.policy_key == "brs_without_extra":
            if torch.any(term | trunc):
                brs_action_history.clear()
        if cfg.policy_key == "act" or cfg.policy_key == "act_multi_head":
            act_init()
        # 增加步数
        step_num += 1

    # ----------------------------
    # 结果处理
    # ----------------------------
    # 如果有额外统计键，保存统计信息
    if len(cfg.eval_env.extra_stat_keys):
        torch.save(
            eval_envs.extra_stats,
            logger.exp_path / "eval_extra_stat_keys.pt",
        )

    # 打印子任务失败统计
    print(
        "subtask_fail_counts",
        dict((k, subtask_fail_counts[k]) for k in sorted(subtask_fail_counts.keys())),
    )

    # 计算结果指标
    results_logs = dict(
        num_trajs=len(eval_envs.return_queue),  # 轨迹数量
        return_per_step=common.to_tensor(eval_envs.return_queue, device=device)
        .float()
        .mean()
        / eval_envs.max_episode_steps,  # 平均每步回报
        success_once=common.to_tensor(eval_envs.success_once_queue, device=device)
        .float()
        .mean(),  # 至少成功一次的比例
        success_at_end=common.to_tensor(eval_envs.success_at_end_queue, device=device)
        .float()
        .mean(),  # 结束时成功的比例
        len=common.to_tensor(eval_envs.length_queue, device=device).float().mean(),  # 平均轨迹长度
    )
    
    # 计算时间指标
    time_logs = timer.get_time_logs(pbar.last_print_n * cfg.eval_env.max_episode_steps)
    
    # 打印结果
    print(
        "results",
        results_logs,
    )
    print("time", time_logs)
    print("total_time", timer.total_time_elapsed)

    # 保存结果到文件
    with open(logger.exp_path / "output.txt", "w") as f:
        f.write("results\n" + str(results_logs) + "\n")
        f.write("time\n" + str(time_logs) + "\n")

    # ----------------------------
    # 资源清理
    # ----------------------------
    # 关闭环境
    eval_envs.close()
    # 关闭日志记录器
    logger.close()

# 主程序入口
if __name__ == "__main__":
    # 获取配置文件路径（第一个命令行参数）
    PASSED_CONFIG_PATH = sys.argv[1]
    # 解析配置
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    # 执行评估
    eval(cfg)