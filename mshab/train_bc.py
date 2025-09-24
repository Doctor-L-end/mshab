# 文档和实验结果参考：https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

# 导入标准库
import json  # JSON数据处理
import os  # 操作系统接口
import random  # 随机数生成
import sys  # 系统相关功能
from dataclasses import asdict, dataclass, field  # 数据类处理
from pathlib import Path  # 路径操作
from typing import Dict, List, Optional, Union  # 类型提示

# 导入第三方库
import h5py  # HDF5文件处理
from dacite import from_dict  # 字典转数据类
from omegaconf import OmegaConf  # 配置管理
import gymnasium as gym  # 强化学习环境
import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch神经网络函数
import torchvision.transforms as T

# ManiSkill特定导入
import mani_skill.envs  # ManiSkill机器人操作环境
from mani_skill.utils import common  # ManiSkill工具函数

# 项目特定导入
from mshab.agents.bc import Agent  # 行为克隆(BC)智能体
from mshab.envs.make import EnvConfig, make_env  # 环境配置和创建
from mshab.utils.array import to_tensor  # 数组转张量工具
from mshab.utils.config import parse_cfg  # 配置解析
from mshab.utils.dataclasses import default_field  # 数据类默认字段
from mshab.utils.dataset import ClosableDataLoader, ClosableDataset  # 可关闭数据集和加载器
from mshab.utils.logger import Logger, LoggerConfig  # 日志记录器
from mshab.utils.time import NonOverlappingTimeProfiler  # 时间分析器

# 行为克隆(BC)算法的配置类
@dataclass
class BCConfig:
    name: str = "bc"  # 算法名称

    # 训练参数
    lr: float = 3e-4  # 学习率
    batch_size: int = 512  # 批量大小

    # 运行参数
    epochs: int = 100  # 训练轮数
    eval_freq: int = 1  # 评估频率（按轮数计）
    log_freq: int = 1  # 日志记录频率（按轮数计）
    save_freq: int = 1  # 模型保存频率（按轮数计）
    save_backup_ckpts: bool = False  # 是否保存不覆盖的检查点副本

    # 数据集参数
    data_dir_fp: str = None  # 数据目录路径（包含.h5文件）
    max_cache_size: int = 0  # CPU内存中最大缓存数据点数
    trajs_per_obj: Union[str, int] = "all"  # 每个物体使用的轨迹数（"all"表示全部）

    torch_deterministic: bool = True  # 是否启用PyTorch确定性模式

    # 从环境配置中传递的字段（初始化后设置）
    num_eval_envs: int = field(init=False)  # 并行评估环境数量

    # 配置后处理
    def _additional_processing(self):
        assert self.name == "bc", "算法配置错误"  # 验证算法名称

        # 处理trajs_per_obj参数
        try:
            self.trajs_per_obj = int(self.trajs_per_obj)  # 尝试转换为整数
        except:
            pass
        # 验证类型
        assert isinstance(self.trajs_per_obj, int) or self.trajs_per_obj == "all"

# 训练配置类（包含所有配置）
@dataclass
class TrainConfig:
    seed: int  # 随机种子
    eval_env: EnvConfig  # 评估环境配置
    algo: BCConfig  # 算法配置
    logger: LoggerConfig  # 日志记录器配置

    wandb_id: Optional[str] = None  # Weights & Biases ID
    resume_logdir: Optional[Union[Path, str]] = None  # 恢复训练日志目录
    model_ckpt: Optional[Union[Path, int, str]] = None  # 模型检查点路径

    # 初始化后处理
    def __post_init__(self):
        # 恢复目录与清除输出目录冲突检查
        assert (
            self.resume_logdir is None or not self.logger.clear_out
        ), "无法恢复已清除的输出目录！"

        # 处理恢复目录
        if self.resume_logdir is not None:
            self.resume_logdir = Path(self.resume_logdir)
            old_config_path = self.resume_logdir / "config.yml"
            
            # 检查新旧配置路径是否相同
            if old_config_path.absolute() == Path(PASSED_CONFIG_PATH).absolute():
                assert (
                    self.resume_logdir == self.logger.exp_path
                ), "设置resume_logdir时必须匹配日志目录"
            else:
                # 加载旧配置
                assert old_config_path.exists(), f"找不到旧配置：{old_config_path}"
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

            # 设置默认模型检查点
            if self.model_ckpt is None:
                self.model_ckpt = self.logger.model_path / "latest.pt"

        # 处理模型检查点路径
        if self.model_ckpt is not None:
            self.model_ckpt = Path(self.model_ckpt)
            assert self.model_ckpt.exists(), f"找不到模型检查点：{self.model_ckpt}"

        # 设置算法配置中的评估环境数量
        self.algo.num_eval_envs = self.eval_env.num_envs
        self.algo._additional_processing()  # 执行算法配置后处理

        # 准备日志配置（排除递归字段）
        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]

# 从配置字典创建训练配置对象
def get_mshab_train_cfg(cfg: TrainConfig) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))

# 递归计算对象中所有张量的总内存大小（字节）
def recursive_tensor_size_bytes(obj):
    extra_obj_size = 0
    if isinstance(obj, dict):
        # 字典：递归计算所有值
        extra_obj_size = sum([recursive_tensor_size_bytes(v) for v in obj.values()])
    elif isinstance(obj, list) or isinstance(obj, tuple):
        # 列表/元组：递归计算所有元素
        extra_obj_size = sum([recursive_tensor_size_bytes(x) for x in obj])
    elif isinstance(obj, torch.Tensor):
        # 张量：计算元素数×元素大小
        extra_obj_size = obj.nelement() * obj.element_size()
    # 返回对象本身大小+所有内容大小
    return sys.getsizeof(obj) + extra_obj_size

# 行为克隆数据集类（继承自可关闭数据集）
class BCDataset(ClosableDataset):
    def __init__(
        self,
        data_dir_fp: str,  # 数据目录路径
        max_cache_size: int,  # 最大缓存大小
        transform_fn=torch.from_numpy,  # 数据转换函数
        trajs_per_obj: Union[str, int] = "all",  # 每个物体的轨迹数
        cat_state=True,  # 是否拼接状态特征
        cat_pixels=False,  # 是否拼接像素特征
    ):
        data_dir_fp: Path = Path(data_dir_fp)
        self.data_files: List[h5py.File] = []  # HDF5文件列表
        self.json_files: List[Dict] = []  # JSON元数据列表
        self.obj_names_in_loaded_order: List[str] = []  # 物体名称列表

        self.transforms = T.Compose([T.Resize((224, 224), antialias=True)])

        # 处理输入路径（文件或目录）
        if data_dir_fp.is_file():
            data_file_names = [data_dir_fp.name]
            data_dir_fp = data_dir_fp.parent
        else:
            data_file_names = os.listdir(data_dir_fp)
        
        # 加载所有HDF5和JSON文件
        for data_fn in data_file_names:
            if data_fn.endswith(".h5"):
                json_fn = data_fn.replace(".h5", ".json")
                self.data_files.append(h5py.File(data_dir_fp / data_fn, "r"))
                with open(data_dir_fp / json_fn, "rb") as f:
                    self.json_files.append(json.load(f))
                self.obj_names_in_loaded_order.append(data_fn.replace(".h5", ""))

        # 创建数据集索引到文件索引的映射
        self.dataset_idx_to_data_idx = dict()
        dataset_idx = 0
        
        # 遍历所有JSON文件（每个文件对应一个物体）
        for file_idx, json_file in enumerate(self.json_files):
            # 选择要使用的轨迹
            if trajs_per_obj == "all":
                use_ep_jsons = json_file["episodes"]  # 使用所有轨迹
            else:
                # 验证轨迹数量
                assert trajs_per_obj <= len(json_file["episodes"]), (
                    f"要求{trajs_per_obj}条轨迹，但物体{self.obj_names_in_loaded_order[file_idx]}"
                    f"只有{len(json_file['episodes'])}条"
                )
                # 随机采样轨迹
                use_ep_jsons = random.sample(json_file["episodes"], k=trajs_per_obj)

            # 为每个轨迹的每个时间步创建映射
            for ep_json in use_ep_jsons:
                ep_id = ep_json["episode_id"]
                for step in range(ep_json["elapsed_steps"]):
                    self.dataset_idx_to_data_idx[dataset_idx] = (file_idx, ep_id, step)
                    dataset_idx += 1
        self._data_len = dataset_idx  # 数据集总大小

        # 缓存相关初始化
        self.max_cache_size = max_cache_size
        self.cache = dict()

        # 数据转换配置
        self.transform_fn = transform_fn
        self.cat_state = cat_state
        self.cat_pixels = cat_pixels

    # 递归转换数据
    def transform_idx(self, x, data_index):
        if isinstance(x, h5py.Group) or isinstance(x, dict):
            # 组/字典：递归转换所有值
            return dict((k, self.transform_idx(v, data_index)) for k, v in x.items())
        # 数据项：转换为张量并调整形状
        out = self.transform_fn(np.array(x[data_index]))
        if len(out.shape) == 0:
            out = out.unsqueeze(0)  # 标量转换为1D张量
        return out

    # 获取单个数据项
    def get_single_item(self, index):
        # 检查缓存
        if index in self.cache:
            return self.cache[index]

        # 解析索引
        file_num, ep_num, step_num = self.dataset_idx_to_data_idx[index]
        ep_data = self.data_files[file_num][f"traj_{ep_num}"]

        # 转换观测数据
        observation = ep_data["obs"]
        agent_obs = self.transform_idx(observation["agent"], step_num)
        extra_obs = self.transform_idx(observation["extra"], step_num)
        
        # print(observation["sensor_data"]["fetch_head"]["depth"].shape) # (201, 128, 128, 1)  201代表时间步
        # 处理深度图像数据
        fetch_head_depth = (
            self.transform_idx(
                observation["sensor_data"]["fetch_head"]["depth"], step_num
            )
            .squeeze(-1)  # 移除单维度
            .unsqueeze(0)  # 添加通道维度
        )
        # print(fetch_head_depth.shape) # (1, 128, 128)  1表示通道维度
        fetch_hand_depth = (
            self.transform_idx(
                observation["sensor_data"]["fetch_hand"]["depth"], step_num
            )
            .squeeze(-1)
            .unsqueeze(0)
        )

        fetch_head_rgb = (
            self.transform_idx(
                observation["sensor_data"]["fetch_head"]["rgb"], step_num
            ).permute(2, 0, 1)
        )

        fetch_hand_rgb = (
            self.transform_idx(
                observation["sensor_data"]["fetch_hand"]["rgb"], step_num
            ).permute(2, 0, 1)
        )

        # 构建状态观测
        state_obs = (
            dict(
                state=torch.cat(
                    [
                        *agent_obs.values(),
                        *extra_obs.values(),
                    ],
                    axis=0,
                )
            )
            if self.cat_state
            else dict(agent_obs=agent_obs, extra_obs=extra_obs)
        )
        
        # 构建像素观测
        pixel_obs = (
            dict(all_depth=torch.stack([fetch_head_rgb, fetch_head_depth, fetch_hand_rgb, fetch_hand_depth], axis=-3))
            if self.cat_pixels
            else dict(
                fetch_head_rgb=self.transforms(fetch_head_rgb),
                fetch_head_depth=self.transforms(fetch_head_depth),
                fetch_hand_rgb=self.transforms(fetch_hand_rgb),
                fetch_hand_depth=self.transforms(fetch_hand_depth),
            )
        )

        # 合并观测
        obs = dict(**state_obs, pixels=pixel_obs)
        # 获取动作
        act = self.transform_idx(ep_data["actions"], step_num)

        res = (obs, act)  # 返回(观测, 动作)对
        
        # 缓存结果（如果缓存未满）
        if len(self.cache) < self.max_cache_size:
            self.cache[index] = res
        return res

    # 获取数据项（支持索引和切片）
    def __getitem__(self, indexes):
        if isinstance(indexes, int):
            return self.get_single_item(indexes)
        return [self.get_single_item(i) for i in indexes]

    # 返回数据集大小
    def __len__(self):
        return self._data_len

    # 关闭所有打开的文件
    def close(self):
        for f in self.data_files:
            f.close()

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
    agent = Agent(eval_obs, eval_envs.unwrapped.single_action_space.shape).to(device)
    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=cfg.algo.lr,
    )

    # 模型保存函数
    def save(save_path):
        torch.save(
            dict(
                agent=agent.state_dict(),  # 智能体参数
                optimizer=optimizer.state_dict(),  # 优化器状态
            ),
            save_path,
        )

    # 模型加载函数
    def load(load_path):
        checkpoint = torch.load(str(load_path), map_location=device)
        agent.load_state_dict(checkpoint["agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])

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
    bc_dataset = BCDataset(
        cfg.algo.data_dir_fp,
        cfg.algo.max_cache_size,
        cat_state=cfg.eval_env.cat_state,
        cat_pixels=cfg.eval_env.cat_pixels,
        trajs_per_obj=cfg.algo.trajs_per_obj,
    )
    # 打印数据集信息
    logger.print(
        f"创建BC数据集: {len(bc_dataset)}样本, "
        f"每个物体{cfg.algo.trajs_per_obj}条轨迹, "
        f"共{len(bc_dataset.obj_names_in_loaded_order)}个物体",
        flush=True,
    )
    bc_dataloader = ClosableDataLoader(
        bc_dataset,
        batch_size=cfg.algo.batch_size,
        shuffle=True,
        num_workers=2,  # 数据加载工作线程数
    )

    # 训练轮次初始化
    epoch = 0
    logger_start_log_step = logger.last_log_step + 1 if logger.last_log_step > 0 else 0

    # 辅助函数：检查是否满足频率条件
    def check_freq(freq):
        return epoch % freq == 0

    # 辅助函数：收集环境统计信息
    def store_env_stats(key):
        assert key == "eval", "BC仅支持评估环境"
        log_env = eval_envs
        # 计算并存储各项指标
        logger.store(
            key,
            return_per_step=common.to_tensor(log_env.return_queue, device=device)
            .float()
            .mean()
            / log_env.max_episode_steps,  # 平均每步回报
            success_once=common.to_tensor(log_env.success_once_queue, device=device)
            .float()
            .mean(),  # 任务成功比例（任意时刻）
            success_at_end=common.to_tensor(log_env.success_at_end_queue, device=device)
            .float()
            .mean(),  # 任务成功比例（结束时）
            len=common.to_tensor(log_env.length_queue, device=device).float().mean(),  # 平均轨迹长度
        )
        log_env.reset_queues()  # 重置环境统计队列

    # 开始训练
    print("开始训练")
    timer = NonOverlappingTimeProfiler()  # 初始化时间分析器
    for epoch in range(cfg.algo.epochs):
        # 检查是否超过总轮次限制
        if epoch + logger_start_log_step > cfg.algo.epochs:
            break

        # 打印当前轮次信息
        logger.print(
            f"总轮次: {epoch + logger_start_log_step}; 当前进程轮次: {epoch}"
        )

        # 训练阶段
        tot_loss, n_samples = 0, 0
        for obs, act in iter(bc_dataloader):
            # 数据转移到设备
            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(
                act, device=device, dtype="float"
            )
            n_samples += act.size(0)  # 累计样本数

            # 前向传播
            pi = agent(obs)
            # 计算均方误差损失
            loss = F.mse_loss(pi, act)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()  # 累计损失
        loss_logs = dict(loss=tot_loss / n_samples)  # 计算平均损失
        timer.end(key="train")  # 记录训练时间

        # 日志记录
        if check_freq(cfg.algo.log_freq):
            logger.store(tag="losses", **loss_logs)  # 存储损失
            if epoch > 0:
                logger.store("time", **timer.get_time_logs(epoch))  # 存储时间信息
            logger.log(logger_start_log_step + epoch)  # 记录日志
            timer.end(key="log")  # 记录日志时间

        # 评估
        if cfg.algo.eval_freq:
            if check_freq(cfg.algo.eval_freq):
                agent.eval()  # 设置评估模式
                eval_obs, _ = eval_envs.reset()  # 重置环境

                # 运行完整评估轨迹
                for _ in range(eval_envs.max_episode_steps):
                    with torch.no_grad():  # 禁用梯度计算
                        action = agent(eval_obs)  # 选择动作
                    eval_obs, _, _, _, _ = eval_envs.step(action)  # 执行动作

                # 收集并记录评估统计信息
                if len(eval_envs.return_queue) > 0:
                    store_env_stats("eval")
                logger.log(logger_start_log_step + epoch)
                timer.end(key="eval")  # 记录评估时间

        # 模型保存
        if check_freq(cfg.algo.save_freq):
            if cfg.algo.save_backup_ckpts:
                save(logger.model_path / f"{epoch}_ckpt.pt")  # 保存备份检查点
            save(logger.model_path / "latest.pt")  # 保存最新检查点
            timer.end(key="checkpoint")  # 记录检查点保存时间

    # 训练结束保存最终模型
    save(logger.model_path / "final_ckpt.pt")
    save(logger.model_path / "latest.pt")

    # 清理资源
    bc_dataloader.close()
    eval_envs.close()
    logger.close()

# 主程序入口
if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]  # 从命令行参数获取配置路径
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))  # 解析配置
    train(cfg)  # 启动训练