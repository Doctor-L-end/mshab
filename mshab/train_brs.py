# 导入必要的库和模块
import os  # 操作系统接口
import random  # 随机数生成
import sys  # 系统相关参数和函数
from dataclasses import asdict, dataclass  # 数据类处理
from pathlib import Path  # 面向对象的文件系统路径
from typing import Dict, List, Literal, Optional, Union  # 类型提示

import h5py  # HDF5文件处理
from dacite import from_dict  # 从字典创建数据类实例
from omegaconf import OmegaConf  # 配置管理
from tqdm import tqdm  # 进度条显示

import gymnasium as gym  # 强化学习环境
from gymnasium import spaces  # 强化学习空间定义

import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
import torch.optim as optim  # 优化器
from torch.utils.data.sampler import BatchSampler, RandomSampler  # 数据采样
import torchvision.transforms as T
from diffusers.optimization import get_scheduler  # 学习率调度器
from diffusers.training_utils import EMAModel  # 指数移动平均模型

from mani_skill import ASSET_DIR  # ManiSkill资产目录
from mani_skill.utils import common  # ManiSkill通用工具

# 从自定义模块导入
from mshab.agents.brs import Agent  # 扩散策略代理
from mshab.agents.brs.utils import IterationBasedBatchSampler, worker_init_fn  # 数据采样工具
from mshab.agents.brs.utils import _get_merged_pc_vectorized
from mshab.envs.make import EnvConfig, make_env  # 环境配置和创建
from mshab.utils.array import to_tensor  # 数组转张量
from mshab.utils.config import parse_cfg  # 配置解析
from mshab.utils.dataclasses import default_field  # 数据类默认字段
from mshab.utils.dataset import ClosableDataLoader, ClosableDataset  # 可关闭的数据加载器
from mshab.utils.logger import Logger, LoggerConfig  # 日志记录
from mshab.utils.time import NonOverlappingTimeProfiler  # 时间分析器

from mshab.vis import visualize_point_cloud_open3d, save_pointcloud_to_pcd

# BRS算法的配置类
@dataclass
class BRSConfig:
    name: str = "brs"  # 算法名称

    # BRS模型训练参数
    lr: float = 7e-4  # 学习率
    batch_size: int = 256  # 批量大小
    obs_horizon: int = 2  # 观测历史长度
    pred_horizon: int = 8  # 预测动作序列长度

    # 数据集相关参数
    data_dir_fp: str = (  # 演示数据集路径
        ASSET_DIR
        / "scene_datasets/replica_cad_dataset/rearrange-dataset/tidy_hosue/pick"
    )
    trajs_per_obj: Union[Literal["all"], int] = "all"  # 每个对象加载的轨迹数
    truncate_trajectories_at_success: bool = False  # 是否在首次成功时截断轨迹
    max_image_cache_size: Union[Literal["all"], int] = 0  # 最大图像缓存大小
    num_dataload_workers: int = 0  # 数据加载工作线程数

    # 实验相关参数
    num_iterations: int = 1_000_000  # 总迭代次数
    eval_episodes: Optional[int] = None  # 评估周期数
    log_freq: int = 1000  # 日志记录频率
    eval_freq: int = 5000  # 评估频率
    save_freq: int = 5000  # 模型保存频率
    torch_deterministic: bool = True  # 是否启用确定性计算
    save_backup_ckpts: bool = False  # 是否保存备份检查点

# 定义训练配置类
@dataclass
class TrainConfig:
    seed: int  # 随机种子
    eval_env: EnvConfig  # 评估环境配置
    algo: BRSConfig  # 算法配置
    logger: LoggerConfig  # 日志记录器配置

    wandb_id: Optional[str] = None  # Weights & Biases ID
    resume_logdir: Optional[Union[Path, int, str]] = None  # 恢复训练的日志目录
    model_ckpt: Optional[Union[Path, int, str]] = None  # 模型检查点路径

    # 后初始化方法，用于验证和调整配置
    def __post_init__(self):
        # 确保评估环境配置正确
        if self.algo.eval_episodes is None:
            self.algo.eval_episodes = self.eval_env.num_envs
        self.algo.eval_episodes = max(self.algo.eval_episodes, self.eval_env.num_envs)
        assert self.algo.eval_episodes % self.eval_env.num_envs == 0

        # 确保恢复目录和清除输出不冲突
        assert (
            self.resume_logdir is None or not self.logger.clear_out
        ), "Can't resume to a cleared out logdir!"

        # 处理恢复训练的逻辑
        if self.resume_logdir is not None:
            self.resume_logdir = Path(self.resume_logdir)
            old_config_path = self.resume_logdir / "config.yml"
            if old_config_path.absolute() == Path(PASSED_CONFIG_PATH).absolute():
                assert (
                    self.resume_logdir == self.logger.exp_path
                ), "if setting resume_logdir, must set logger workspace and exp_name accordingly"
            else:
                assert (
                    old_config_path.exists()
                ), f"Couldn't find old config at path {old_config_path}"
                old_config = get_mshab_train_cfg(
                    parse_cfg(default_cfg_path=old_config_path)
                )
                # 更新日志配置
                self.logger.workspace = old_config.logger.workspace
                self.logger.exp_path = old_config.logger.exp_path
                self.logger.log_path = old_config.logger.log_path
                self.logger.model_path = old_config.logger.model_path
                self.logger.train_video_path = old_config.logger.train_video_path
                self.logger.eval_video_path = old_config.logger.eval_video_path

            # 设置模型检查点
            if self.model_ckpt is None:
                self.model_ckpt = self.logger.model_path / "latest.pt"

        # 验证模型检查点路径
        if self.model_ckpt is not None:
            self.model_ckpt = Path(self.model_ckpt)
            assert self.model_ckpt.exists(), f"Could not find {self.model_ckpt}"

        # 存储实验配置并清理冗余字段
        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]

# 从字典创建训练配置实例
def get_mshab_train_cfg(cfg: TrainConfig) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))

# 递归重新排序字典键以匹配参考字典结构
def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out

# 递归将h5py对象转换为numpy数组
def recursive_h5py_to_numpy(h5py_obs, slice=None):
    if isinstance(h5py_obs, h5py.Group) or isinstance(h5py_obs, dict):
        return dict(
            (k, recursive_h5py_to_numpy(h5py_obs[k], slice)) for k in h5py_obs.keys()
        )
    if isinstance(h5py_obs, list):
        return [recursive_h5py_to_numpy(x, slice) for x in h5py_obs]
    if isinstance(h5py_obs, tuple):
        return tuple(recursive_h5py_to_numpy(x, slice) for x in h5py_obs)
    if slice is not None:
        return h5py_obs[slice]
    return h5py_obs[:] # 当 h5py_obs是一个 h5py Dataset 对象时，对其进行切片操作（如 [slice]或 [:]）会自动返回一个 NumPy 数组。这是 h5py 库的设计特性，不需要显式调用转换函数。

# BRS数据集类（继承自可关闭数据集）
class BRSDataset(ClosableDataset):
    def __init__(
        self,
        data_path,
        obs_horizon,
        pred_horizon,
        control_mode,
        trajs_per_obj = "all",  # 每个物体的轨迹数
        max_image_cache_size=0,
        truncate_trajectories_at_success=False,
    ):
        data_path = Path(data_path)
        # 处理数据路径（目录或单个文件）
        if data_path.is_dir():
            h5_fps = [
                data_path / fp for fp in os.listdir(data_path) if fp.endswith(".h5")
            ]
        else:
            h5_fps = [data_path]

        # 初始化轨迹存储
        trajectories = dict(actions=[], observations=[])
        num_cached = 0
        self.h5_files = []
        
        self.transforms = T.Compose([T.Resize((224, 224), antialias=True)])

        # 遍历所有HDF5文件
        for fp_num, fp in enumerate(h5_fps):
            f = h5py.File(fp, "r")
            num_uncached_this_file = 0

            # 确定要加载的轨迹
            if trajs_per_obj == "all":
                keys = list(f.keys())
            else:
                keys = random.sample(list(f.keys()), k=trajs_per_obj)

            # 遍历每个轨迹
            for k in tqdm(keys, desc=f"hf file {fp_num}"):
                obs, act = f[k]["obs"], f[k]["actions"][:]

                obs_agent = obs["agent"]
                obs_extra = obs["extra"]["tcp_pose_wrt_base"]

                # 根据成功标志截断轨迹
                if truncate_trajectories_at_success:
                    success: List[bool] = f[k]["success"][:].tolist()
                    success_cutoff = min(success.index(True) + 1, len(success))
                    del success
                else:
                    success_cutoff = len(act)

                # 处理状态观测
                state_obs_list = []
                # 处理 obs_agent
                obs_agent_processed = recursive_h5py_to_numpy(obs_agent, slice=slice(success_cutoff + 1))
                if isinstance(obs_agent_processed, dict):
                    state_obs_list.extend(obs_agent_processed.values())
                else:
                    state_obs_list.append(obs_agent_processed)
                # 处理 obs_extra
                obs_extra_processed = recursive_h5py_to_numpy(obs_extra, slice=slice(success_cutoff + 1))
                if isinstance(obs_extra_processed, dict):
                    state_obs_list.extend(obs_extra_processed.values())
                else:
                    state_obs_list.append(obs_extra_processed)
                
                state_obs_list = [
                    x[:, None] if len(x.shape) == 1 else x for x in state_obs_list
                ]
                state_obs = torch.from_numpy(np.concatenate(state_obs_list, axis=1))
                act = torch.from_numpy(act)
            
                # 处理点云观测
                pixel_obs = dict(
                    fetch_head_rgb=obs["sensor_data"]["fetch_head"]["rgb"],
                    fetch_head_depth=obs["sensor_data"]["fetch_head"]["depth"],
                    fetch_hand_rgb=obs["sensor_data"]["fetch_hand"]["rgb"],
                    fetch_hand_depth=obs["sensor_data"]["fetch_hand"]["depth"],
                )
                # 根据缓存策略处理图像数据
                if (
                    max_image_cache_size == "all"
                    or len(act) <= max_image_cache_size - num_cached
                ):
                    pixel_obs = to_tensor(
                        recursive_h5py_to_numpy(
                            pixel_obs, slice=slice(success_cutoff + 1)
                        )
                    )
                    num_cached += len(act)
                else:
                    num_uncached_this_file += len(act)
                # print(obs['extra'].keys()) # 'tcp_pose_wrt_base', 'obj_pose_wrt_base', 'goal_pos_wrt_base', 'is_grasped'
                # print(obs['agent'].keys()) # 'qpos', 'qvel'
                # print(obs['sensor_param'].keys()) # 'fetch_head', 'fetch_hand'
                
                obs = to_tensor(recursive_h5py_to_numpy(obs, slice=slice(success_cutoff + 1)))
                sensor_data = obs["sensor_data"]
                if sensor_data["fetch_head"]["depth"].ndim == 4:
                    sensor_data["fetch_head"]["depth"] = sensor_data["fetch_head"]["depth"].squeeze(-1)
                if sensor_data["fetch_hand"]["depth"].ndim == 4:
                    sensor_data["fetch_hand"]["depth"] = sensor_data["fetch_hand"]["depth"].squeeze(-1)

                pointcloud_obs = _get_merged_pc_vectorized(
                    sensor_data, obs["sensor_param"], success_cutoff+1
                )

                # 存储轨迹数据
                trajectories["actions"].append(act)
                pixel_obs = {} # 取消图像数据存储 节省内存
                trajectories["observations"].append(dict(state=state_obs, pointcloud=pointcloud_obs, **pixel_obs))

            # 关闭文件或保留打开
            if num_uncached_this_file == 0:
                f.close()
            else:
                self.h5_files.append(f)
        # 根据控制模式设置动作填充
        if (
            "delta_pos" in control_mode
            or control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,)
            )
        else:
            raise NotImplementedError(f"Control Mode {control_mode} not supported")
        
        # 设置观测和预测长度
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (
            obs_horizon,
            pred_horizon,
        )
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        
        # 预计算所有可能的轨迹切片
        for traj_idx in range(num_traj):
            L = trajectories["observations"][traj_idx]["state"].shape[0] - 1
            total_transitions += L

            # 计算填充
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            
            # 生成所有切片
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    # 获取单个数据项
    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
    
        _act_seq = []
        for i in reversed(range(self.obs_horizon)):
            index_start = start - self.obs_horizon + 1 + i
            index_end = index_start + self.pred_horizon
            # 处理动作序列
            act_seq = self.trajectories["actions"][traj_idx][max(0, index_start) : index_end]
            # 在轨迹开始前填充
            if index_start < 0:
                act_seq = torch.cat([act_seq[0].repeat(-index_start, 1), act_seq], dim=0)
            # 在轨迹结束后填充
            if index_end > L:
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                act_seq = torch.cat([act_seq, pad_action.repeat(index_end - L, 1)], dim=0)
            _act_seq.append(act_seq)
        act_seq = torch.stack(_act_seq, dim=0)  # (obs_horizon, pred_horizon, act_dim)

        # 处理观测序列
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]
            # 处理图像数据格式
            if len(obs_seq[k].shape) == 4:
                obs_seq[k] = self.transforms(to_tensor(obs_seq[k]).permute(0, 3, 1, 2))  # FS, D, H, W 原图128x128
            # 在轨迹开始前填充
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)

        # 验证序列长度
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[1] == self.pred_horizon
            and act_seq.shape[0] == self.obs_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

    # 返回数据集长度
    def __len__(self):
        return len(self.slices)

    # 关闭所有打开的文件
    def close(self):
        for h5_file in self.h5_files:
            h5_file.close()

# 训练主函数
def train(cfg: TrainConfig):
    # 设置随机种子
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.algo.torch_deterministic

    # 设置设备（优先使用GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建评估环境
    print("创建评估环境中...")
    eval_envs = make_env(
        cfg.eval_env,
        video_path=cfg.logger.eval_video_path,  # 评估视频保存路径
    )
    # 验证动作空间类型
    assert isinstance(
        eval_envs.single_action_space, gym.spaces.Box
    ), "仅支持连续动作空间"
    print("创建完成")

    # 初始化环境
    eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)
    eval_envs.action_space.seed(cfg.seed + 1_000_000)

    # -------------------------------------------------------------------------------------------------
    # 智能体和优化器初始化
    # -------------------------------------------------------------------------------------------------
    agent = Agent(eval_envs, args=cfg.algo).to(device)
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(eval_envs, args=cfg.algo).to(device)

    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=cfg.algo.lr,
    )

    # 创建学习率调度器
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=cfg.algo.num_iterations,
    )

    # 模型保存函数
    def save(save_path):
        ema.copy_to(ema_agent.parameters())
        torch.save(
            {
                "agent": agent.state_dict(),
                "ema_agent": ema_agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            save_path,
        )

    # 模型加载函数
    def load(load_path):
        checkpoint = torch.load(load_path)
        agent.load_state_dict(checkpoint["agent"])
        ema_agent.load_state_dict(checkpoint["ema_agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # 初始化日志记录器
    logger = Logger(
        logger_cfg=cfg.logger,
        save_fn=save,  # 注册保存函数
    )

    # 加载模型检查点（如果提供）
    if cfg.model_ckpt is not None:
        load(cfg.model_ckpt)

    # -------------------------------------------------------------------------------------------------
    # 数据加载器初始化
    # -------------------------------------------------------------------------------------------------
    brs_dataset = BRSDataset(
        cfg.algo.data_dir_fp,
        cfg.algo.obs_horizon,
        cfg.algo.pred_horizon,
        control_mode=eval_envs.unwrapped.control_mode,
        trajs_per_obj=cfg.algo.trajs_per_obj,
        max_image_cache_size=cfg.algo.max_image_cache_size,
        truncate_trajectories_at_success=cfg.algo.truncate_trajectories_at_success,
    )

    # 创建数据采样器
    sampler = RandomSampler(brs_dataset, replacement=False)
    batch_sampler = BatchSampler(
        sampler, batch_size=cfg.algo.batch_size, drop_last=True
    )
    batch_sampler = IterationBasedBatchSampler(batch_sampler, cfg.algo.num_iterations)
    
    # 创建数据加载器
    train_dataloader = ClosableDataLoader(
        brs_dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.algo.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=cfg.seed),
        pin_memory=True,
        persistent_workers=(cfg.algo.num_dataload_workers > 0),
    )

    print("dataset loaded!", flush=True)

    # -------------------------------------------------------------------------------------------------
    # 开始训练循环
    # -------------------------------------------------------------------------------------------------
    iteration = 0
    logger_start_log_step = logger.last_log_step + 1 if logger.last_log_step > 0 else 0

    # 检查频率的辅助函数
    def check_freq(freq):
        return iteration % freq == 0

    # 存储环境统计信息的函数
    def store_env_stats(key):
        log_env = eval_envs
        # 存储基本统计信息
        logger.store(
            key,
            return_per_step=common.to_tensor(log_env.return_queue, device=device)
            .float()
            .mean()
            / log_env.max_episode_steps,
            success_once=common.to_tensor(log_env.success_once_queue, device=device)
            .float()
            .mean(),
            success_at_end=common.to_tensor(log_env.success_at_end_queue, device=device)
            .float()
            .mean(),
            len=common.to_tensor(log_env.length_queue, device=device).float().mean(),
        )
        # 存储额外统计信息
        extra_stat_logs = dict()
        for k, v in log_env.extra_stats.items():
            extra_stat_values = torch.stack(v)
            extra_stat_logs[f"{k}_once"] = torch.mean(
                torch.any(extra_stat_values, dim=1).float()
            )
            extra_stat_logs[f"{k}_at_end"] = torch.mean(
                extra_stat_values[..., -1].float()
            )
        logger.store(f"extra/{key}", **extra_stat_logs)
        # 重置环境队列
        log_env.reset_queues()

    # 设置代理为训练模式
    agent.train()

    # 创建时间分析器
    timer = NonOverlappingTimeProfiler()

    # 主训练循环
    for iteration, data_batch in tqdm(
        enumerate(train_dataloader),
        initial=logger_start_log_step,
        total=cfg.algo.num_iterations,
    ):
        # 将数据移动到设备
        data_batch = to_tensor(data_batch, device=device, dtype=torch.float)
        if iteration + logger_start_log_step > cfg.algo.num_iterations:
            break

        # 准备观测数据
        obs_batch_dict = data_batch["observations"]

        obs_batch_dict = {
            k: v.cuda(non_blocking=True) for k, v in obs_batch_dict.items()
        }
        act_batch = data_batch["actions"].cuda(non_blocking=True)
        # if iteration == 0:
        #     save_pointcloud_to_pcd(obs_batch_dict['pointcloud'][0, -1].cpu().numpy(), 'debug_pointcloud.pcd')
            # visualize_point_cloud_open3d(obs_batch_dict['pointcloud'][0, -1].cpu().numpy())
        # 计算损失
        loss = agent.compute_loss(
            obs=obs_batch_dict,
            action_seq=act_batch,
        )

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  # 更新学习率

        # 更新EMA模型
        ema.step(agent.parameters())

        # 记录损失和学习率
        logger.store("losses", loss=loss.item())
        logger.store("charts", learning_rate=optimizer.param_groups[0]["lr"])
        timer.end(key="train")

        # 日志记录
        if check_freq(cfg.algo.log_freq):
            if iteration > 0:
                logger.store("time", **timer.get_time_logs(iteration))
            logger.log(logger_start_log_step + iteration)
            timer.end("log")

        # 评估
        if cfg.algo.eval_freq:
            if check_freq(cfg.algo.eval_freq):
                with torch.no_grad():
                    # 使用EMA模型进行评估
                    ema.copy_to(ema_agent.parameters())
                    agent.eval()
                    obs, info = eval_envs.reset()
                    
                    # 运行评估环境直到收集足够的评估周期
                    while len(eval_envs.return_queue) < cfg.algo.eval_episodes:
                        obs = common.to_tensor(obs, device)
                        # 获取动作序列
                        action_seq = agent.get_action(obs)
                        # 执行动作序列
                        for i in range(action_seq.shape[1]):
                            obs, rew, terminated, truncated, info = eval_envs.step(
                                action_seq[:, i]
                            )
                            if truncated.any():
                                break

                        # 确保所有环境同时截断
                        if truncated.any():
                            assert (
                                truncated.all() == truncated.any()
                            ), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                    
                    # 恢复训练模式
                    agent.train()
                    
                    # 存储评估统计信息
                    if len(eval_envs.return_queue) > 0:
                        store_env_stats("eval")
                    logger.log(logger_start_log_step + iteration)
                    timer.end(key="eval")

        # 保存检查点
        if check_freq(cfg.algo.save_freq):
            if cfg.algo.save_backup_ckpts:
                save(logger.model_path / f"{iteration}_ckpt.pt")
            save(logger.model_path / "latest.pt")
            timer.end(key="checkpoint")

    # 清理资源
    train_dataloader.close()
    eval_envs.close()
    logger.close()

# 主程序入口
if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]  # 从命令行参数获取配置路径
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))  # 解析配置
    train(cfg)  # 启动训练