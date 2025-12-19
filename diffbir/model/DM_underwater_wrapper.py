import torch
import torch.nn as nn
import os
import sys
import logging
import json
from collections import OrderedDict

# 添加DM_underwater路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
dm_underwater_path = os.path.join(current_dir, 'DM_underwater')
if dm_underwater_path not in sys.path:
    sys.path.append(dm_underwater_path)

import core.logger as Logger
from model import create_model

logger = logging.getLogger('base')

class DM_Underwater_Wrapper(nn.Module):
    """
    将DM_underwater的扩散模型包装成与SCEIR_Model兼容的接口
    """
    def __init__(self, opts, mode='test'):
        super().__init__()
        self.opts = opts
        self.mode = mode
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载DM_underwater配置
        self.opt = self._load_config()
        
        # 初始化DM_underwater模型（完全按照infer.py的方式）
        self.diffusion = create_model(self.opt)
        
        # 设置噪声调度（完全按照infer.py的方式）
        self.diffusion.set_new_noise_schedule(
            self.opt['model']['beta_schedule']['val'], schedule_phase='val')
        
        # DDPM没有to方法，但netG有，将netG移动到设备
        if hasattr(self.diffusion, 'netG'):
            self.diffusion.netG = self.diffusion.netG.to(self.device)
        
        # 设置模型为评估模式（DDPM本身无eval接口，调用包装器的eval以切到netG.eval）
        if mode == 'test':
            self.eval()
    
    def _load_config(self):
        """加载DM_underwater配置，使用infer.py的方式"""
        # 获取配置文件路径
        config_path = getattr(self.opts, 'DM_underwater_config', None)
        if config_path is None:
            # 使用默认配置文件
            config_path = os.path.join(dm_underwater_path, 'config', 'underwater.json')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"DM_underwater配置文件不存在: {config_path}")
        
        # 读取配置文件
        json_str = ''
        with open(config_path, 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'  # 移除注释
                json_str += line
        
        opt = json.loads(json_str, object_pairs_hook=OrderedDict)
        
        # 完全按照infer.py的方式：使用配置文件中的resume_state
        # 如果配置文件中没有设置resume_state，则使用传入的路径
        if not opt['path'].get('resume_state'):
            if hasattr(self.opts, 'DM_underwater_path') and self.opts.DM_underwater_path:
                # 确保使用绝对路径
                if not os.path.isabs(self.opts.DM_underwater_path):
                    opt['path']['resume_state'] = os.path.abspath(self.opts.DM_underwater_path)
                else:
                    opt['path']['resume_state'] = self.opts.DM_underwater_path
                logger.info(f"使用传入的DM_underwater权重路径: {opt['path']['resume_state']}")
            else:
                logger.warning("配置文件和参数中都没有设置权重路径，将使用随机初始化的模型")
        else:
            logger.info(f"使用配置文件中的权重路径: {opt['path']['resume_state']}")
        
        # 设置基本路径
        opt['phase'] = 'val'
        opt['gpu_ids'] = [0]  # 使用第一个GPU
        opt['distributed'] = False
        
        # 创建必要的目录
        experiments_root = os.path.join('experiments_val', f"{opt['name']}_wrapper")
        opt['path']['experiments_root'] = experiments_root
        for key, path in opt['path'].items():
            if 'resume' not in key and 'experiments' not in key:
                opt['path'][key] = os.path.join(experiments_root, path)
                os.makedirs(opt['path'][key], exist_ok=True)
        
        # 处理resume_state路径（如果是相对路径，相对于DM_underwater目录）
        if opt['path'].get('resume_state') and not os.path.isabs(opt['path']['resume_state']):
            # 将相对路径转换为相对于DM_underwater目录的绝对路径
            opt['path']['resume_state'] = os.path.join(dm_underwater_path, opt['path']['resume_state'])
            logger.info(f"转换相对路径为绝对路径: {opt['path']['resume_state']}")
        
        # 转换为NoneDict格式
        opt = Logger.dict_to_nonedict(opt)
        
        logger.info(f"加载DM_underwater配置: {config_path}")
        return opt
    
    def forward(self, batch):
        """
        前向传播，兼容SCEIR_Model的接口
        Args:
            batch: 输入数据，应该是tensor格式 [B, C, H, W]
        Returns:
            output: 增强后的图像
            ain_out: 中间输出（为了兼容性）
            pre_std: 中间输出（为了兼容性）
            pre_mean: 中间输出（为了兼容性）
        """
        # 确保输入格式正确
        if batch.dim() == 4:
            # 输入是 [B, C, H, W] 格式
            input_data = batch
        else:
            raise ValueError(f"Expected 4D tensor, got {batch.dim()}D")
        
        # 准备数据格式 - DM_underwater期望的格式
        data = {
            'HR': input_data,   # 目标图像（这里用输入图像作为目标）
            'SR': input_data,   # 输入图像
            'LR': input_data,   # 低分辨率图像
            'style': input_data # DM_underwater期望的风格图，数据集里通常与SR同源，这里用输入代替
        }
        
        # 设置数据
        self.diffusion.feed_data(data)
        
        # 进行推理 - 使用continous模式获取所有时间步，然后取最后batch_size个样本
        self.diffusion.test(continous=True)
        
        # 获取结果
        visuals = self.diffusion.get_current_visuals(need_LR=False)
        output = visuals['SR']  # 获取所有时间步的结果
        
        # 从所有时间步中提取最后batch_size个样本
        batch_size = input_data.shape[0]
        if output.dim() == 4:
            # 如果输出是 [N, C, H, W]，取最后batch_size个样本
            if output.shape[0] >= batch_size:
                output = output[-batch_size:]  # 取最后batch_size个样本
            else:
                # 如果样本数不足，直接报错退出
                raise ValueError(f"样本数不足！期望至少{batch_size}个样本，实际只有{output.shape[0]}个样本")
        else:
            # 如果输出维度不对，直接报错退出
            raise ValueError(f"输出维度错误！期望4D张量，实际是{output.dim()}D张量，形状: {output.shape}")
        
        # 将输出从[-1, 1]范围归一化到[0, 1]范围
        # print(output)
        output = (output + 1.0) / 2.0  # 从[-1,1]转换到[0,1]
        
        # 确保输出在正确的设备上
        if hasattr(self, 'device'):
            output = output.to(self.device)
        else:
            # 如果没有设备信息，使用输入数据的设备
            output = output.to(batch.device)
        
        # 为了兼容SCEIR_Model的接口，返回额外的中间结果
        ain_out = output  # 简化处理
        pre_std = torch.ones_like(output) * 0.5  # 占位符
        pre_mean = torch.ones_like(output) * 0.5  # 占位符

        # print("--------------------------------")
        # print("output.shape:", output.shape)
        # print("--------------------------------")
        
        return output, ain_out, pre_std, pre_mean
    
    def load_checkpoint(self, model_path, device):
        """加载预训练模型，完全按照infer.py的方式"""
        if model_path is None:
            logger.warning("未指定模型路径，使用随机初始化的模型")
            return
        
        # 完全按照infer.py的方式：直接调用load_network()
        # DM_underwater会自动处理权重路径（从配置中的resume_state读取）
        logger.info(f'加载DM_underwater模型从配置路径: {self.opt["path"]["resume_state"]}')
        try:
            # 使用DM_underwater的原始加载方式（完全按照infer.py）
            self.diffusion.load_network()
            logger.info('成功加载DM_underwater模型')
        except Exception as e:
            logger.error(f'加载模型失败: {e}')
            logger.warning('使用随机初始化的模型继续')
    
    def save_checkpoint(self, model_path):
        """保存模型（为了兼容性）"""
        state_dict = self.diffusion.netG.state_dict()
        torch.save(state_dict, model_path)
        logger.info(f"模型已保存到: {model_path}")
    
    def eval(self):
        """设置为评估模式"""
        return super().eval()
    
    def train(self, mode: bool = True):
        """设置为训练/评估模式，与nn.Module一致"""
        super().train(mode)
        # 同步内部netG状态
        if hasattr(self.diffusion, 'netG'):
            self.diffusion.netG.train(mode)
        return self
