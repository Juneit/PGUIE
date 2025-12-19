import os
from argparse import ArgumentParser
import copy
import psutil
import gc
import time
import random
from collections import defaultdict

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from diffbir.model.cldm_feautre_fusion import ControlLDM
from diffbir.model import SwinIR, Diffusion

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
dm_underwater_path = os.path.join(current_dir, 'diffbir', 'model', 'DM_underwater')
if dm_underwater_path not in sys.path:
    sys.path.append(dm_underwater_path)

from diffbir.model.SCEIR.train import SCEIR_Model

def get_stage1_model(cfg):
    model_type = getattr(cfg.train, 'stage1_model_type', 'sceir').lower()
    
    if model_type == 'dm_underwater':
        from diffbir.model.DM_underwater_wrapper import DM_Underwater_Wrapper
        return DM_Underwater_Wrapper
    elif model_type == 'sceir':
        from diffbir.model.SCEIR.train import SCEIR_Model
        return SCEIR_Model
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，支持的类型: 'sceir', 'dm_underwater'")
from diffbir.model.refinenet import RefineNet
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img, calculate_psnr_pt, calculate_ssim
from diffbir.sampler import SpacedSampler
from diffbir.utils.uw_metrics import calculate_underwater_metrics_batch
from diffbir.utils.feature_storage import FeatureStorage, FeatureExtractor


def print_memory_usage(accelerator, step_name=""):
    if accelerator.is_main_process:
        memory = psutil.virtual_memory()
        print(f"{step_name} - 系统内存: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                gpu_allocated = torch.cuda.memory_allocated(i)
                gpu_reserved = torch.cuda.memory_reserved(i)
                print(f"  GPU {i}: {gpu_allocated/1024**3:.1f}GB/{gpu_memory/1024**3:.1f}GB (分配), {gpu_reserved/1024**3:.1f}GB (保留)")


class TimeProfiler:
    def __init__(self):
        self.timers = defaultdict(list)
        self.current_timer = None
        self.start_time = None
    
    def start(self, timer_name):
        if self.current_timer is not None:
            self.stop()
        self.current_timer = timer_name
        self.start_time = time.time()
    
    def stop(self):
        if self.current_timer is not None and self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.timers[self.current_timer].append(elapsed)
            self.current_timer = None
            self.start_time = None
    
    def get_stats(self):
        stats = {}
        for timer_name, times in self.timers.items():
            if times:
                stats[timer_name] = {
                    'total': sum(times),
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
        return stats
    
    def print_stats(self):
        stats = self.get_stats()
        print("\n" + "="*60)
        print("时间统计报告")
        print("="*60)
        for timer_name, stat in stats.items():
            print(f"{timer_name:30s}: 总时间={stat['total']:8.3f}s, "
                  f"平均={stat['avg']:6.3f}s, "
                  f"最小={stat['min']:6.3f}s, "
                  f"最大={stat['max']:6.3f}s, "
                  f"次数={stat['count']:4d}")
        print("="*60)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_vae_fusion_weights(cldm, args, cfg, accelerator):
    vae_fusion_path = None
    if hasattr(args, 'vae_fusion_checkpoint') and args.vae_fusion_checkpoint:
        vae_fusion_path = args.vae_fusion_checkpoint
    elif hasattr(cfg.train, 'vae_fusion_resume') and cfg.train.vae_fusion_resume:
        vae_fusion_path = cfg.train.vae_fusion_resume
    
    if vae_fusion_path:
        try:
            vae_fusion_ckpt = torch.load(vae_fusion_path, map_location="cpu")
            fusion_state_dict = {}
            for key, value in vae_fusion_ckpt.items():
                if ('fusion_blocks' in key and (
                    'lq_proj' in key or 'ref_proj' in key or 'similarity_conv' in key or 
                    'global_similarity' in key or 'fusion' in key or 'fuse' in key or 
                    'feature_proj' in key or 'gate' in key)):
                    fusion_state_dict[key] = value
            
            if fusion_state_dict:
                missing_keys, unexpected_keys = cldm.vae.load_state_dict(fusion_state_dict, strict=False)
                if accelerator.is_main_process:
                    print(f"加载VAE特征融合权重从: {vae_fusion_path}")
                    if missing_keys:
                        print(f"  缺失的权重键: {missing_keys}")
                    if unexpected_keys:
                        print(f"  未预期的权重键: {unexpected_keys}")
                    print(f"  成功加载 {len(fusion_state_dict)} 个特征融合权重")
                    for key in sorted(fusion_state_dict.keys()):
                        print(f"    - {key}")
            else:
                if accelerator.is_main_process:
                    print(f"警告: 在权重文件 {vae_fusion_path} 中未找到特征融合相关的权重")
                    print("  期望的权重格式：")
                    print("    - fusion_blocks.X.lq_proj.* (SimilarityAwareFusion)")
                    print("    - fusion_blocks.X.ref_proj.* (SimilarityAwareFusion)")
                    print("    - fusion_blocks.X.similarity_conv.* (SimilarityAwareFusion)")
                    print("    - fusion_blocks.X.global_similarity.* (SimilarityAwareFusion)")
                    print("    - fusion_blocks.X.fusion.* (SimilarityAwareFusion)")
                    print("    - fusion_blocks.X.fuse.* (ConvNeXtFusion，向后兼容)")
                    print("    - fusion_blocks.X.feature_proj.*/gate.* (ResidualGatedFusion，向后兼容)")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"错误: 无法加载VAE特征融合权重 {vae_fusion_path}: {e}")
    else:
        if accelerator.is_main_process:
            print("未指定VAE特征融合权重路径，将使用默认初始化权重")




def test_on_file_list(args) -> None:
    profiler = TimeProfiler()
    
    # Setup accelerator:
    profiler.start("初始化")
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    profiler.stop()

    if accelerator.is_main_process:
        output_dir = args.output_dir if args.output_dir else cfg.inference.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"测试结果将保存到: {output_dir}")

    profiler.start("模型加载")
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    if args.checkpoint_path:
        cldm.load_controlnet_from_ckpt(torch.load(args.checkpoint_path, map_location="cpu"))
        if accelerator.is_main_process:
            print(f"加载controlnet权重从: {args.checkpoint_path}")
    else:
        if accelerator.is_main_process:
            print("警告: 未指定checkpoint路径，将使用预训练权重")

    load_vae_fusion_weights(cldm, args, cfg, accelerator)


    Stage1Model = get_stage1_model(cfg)
    model_type = getattr(cfg.train, 'stage1_model_type', 'sceir').lower()
    
    if model_type == 'dm_underwater':
        if not hasattr(cfg.train, 'DM_underwater_path'):
            raise ValueError("使用DM_underwater模型时，必须在配置文件中设置 DM_underwater_path")
        weight_path = cfg.train.DM_underwater_path
        
        # 设置DM_underwater配置文件路径
        if hasattr(cfg.train, 'DM_underwater_config'):
            cfg.train.DM_underwater_config = cfg.train.DM_underwater_config
        else:
            # 使用默认配置文件
            cfg.train.DM_underwater_config = None
    else:  # sceir
        weight_path = cfg.train.SCEIR_path
    

    sceir = Stage1Model(cfg, mode='test').to(device)
    sceir.load_checkpoint(weight_path, device)
    for p in sceir.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        print(f"使用{model_type.upper()}模型，权重路径: {weight_path}")
        if model_type == 'dm_underwater' and hasattr(cfg.train, 'DM_underwater_config') and cfg.train.DM_underwater_config:
            print(f"DM_underwater配置文件: {cfg.train.DM_underwater_config}")

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    profiler.stop()


    profiler.start("模型准备")
    cldm.eval()
    cldm.to(device)
    sceir.eval()
    sceir.to(device)
    diffusion.to(device)
    cldm = accelerator.prepare(cldm)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    profiler.stop()


    profiler.start("采样器创建")
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    profiler.stop()

    if accelerator.is_main_process:
        print("使用配置文件中的测试数据集配置")
    
    profiler.start("数据加载器准备")
    test_dataset = instantiate_from_config(cfg.dataset.test)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.inference.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    test_loader = accelerator.prepare(test_loader)
    profiler.stop()
    
    if accelerator.is_main_process:
        print(f"测试数据集包含 {len(test_dataset):,} 个图像")


    batch_transform = instantiate_from_config(cfg.batch_transform)

    # 设置测试模式（从配置中获取训练模式相关设置）
    train_mode = getattr(cfg.train, 'train_mode', 'diffusion')  # 'diffusion', 'reconstruction', 'joint', 或 'refinenet'
    enable_feature_fusion = getattr(cfg.train, 'enable_feature_fusion', False)


    feature_storage = None
    if enable_feature_fusion and accelerator.is_main_process:
        feature_db_path = getattr(cfg.train, 'feature_db_path', None)
        if feature_db_path and os.path.exists(feature_db_path):
            # 添加内存优化选项
            lazy_load = getattr(cfg.train, 'lazy_load_features', True)  # 延迟加载特征库
            if lazy_load:
                # 延迟初始化，只在需要时加载
                feature_storage = FeatureStorage(save_dir=feature_db_path, lazy_load=True)
                if accelerator.is_main_process:
                    print(f"特征库延迟加载模式已启用: {feature_db_path}")
            else:
                # 立即加载所有特征
                feature_storage = FeatureStorage(save_dir=feature_db_path)
                if accelerator.is_main_process:
                    info = feature_storage.get_database_info()
                    print("特征库信息:")
                    for level, level_info in info.items():
                        print(f"  {level}: {level_info['total_features']} 个特征")
            
            # 打印内存使用情况
            print_memory_usage(accelerator, "特征库初始化后")
        else:
            if accelerator.is_main_process:
                print(f"警告: 特征库路径不存在或未指定: {feature_db_path}")
                print("将使用原始的特征提取方法")


    def retrieve_features_from_database(clean_img, lq_img=None, feature_storage=None, batch_idx=None):
        if not enable_feature_fusion or feature_storage is None:
            return None, None
        
        profiler.start("特征检索")
        with torch.no_grad():
            pure_cldm.vae.intermediate_features.clear()
            posterior = pure_cldm.vae.encode(clean_img)
            query_features = pure_cldm.vae.get_intermediate_features()
            k = getattr(cfg.train, 'retrieval_k', 1)
            search_results = feature_storage.search_features(query_features, k=k)
            fusion_levels = getattr(cfg.train, 'fusion_levels', [0, 1, 2, 3])
            retrieved_features = []
            
            for level in fusion_levels:
                level_key = f'down_{level}'
                if level_key in search_results and search_results[level_key]:
                    best_match_name, similarity = search_results[level_key][0][0]
                    retrieved_feature = feature_storage.get_feature_by_name(best_match_name, level_key)
                    
                    if retrieved_feature is not None:
                        batch_size = clean_img.shape[0]
                        if retrieved_feature.dim() == 3:  # 单个图像的特征
                            retrieved_feature = retrieved_feature.unsqueeze(0).expand(batch_size, -1, -1, -1)
                        elif retrieved_feature.shape[0] != batch_size:
                            retrieved_feature = retrieved_feature[:1].expand(batch_size, -1, -1, -1)
                        
                        retrieved_features.append(retrieved_feature.to(device))
                        
                        if accelerator.is_main_process and batch_idx == 0:
                            print(f"层级 {level} 检索到特征: {best_match_name}, 相似度: {similarity:.4f}")
                    else:
                        if accelerator.is_main_process:
                            print(f"警告: 在层级 {level} 中未找到检索特征")
                        profiler.stop()
                        return None, None
                else:
                    if accelerator.is_main_process:
                        print(f"警告: 在层级 {level} 中未找到匹配结果")
                    profiler.stop()
                    return None, None
            
            # 提取lq图像特征（如果提供了lq图像）
            lq_features = None
            if lq_img is not None:
                # 清空之前的中间特征
                pure_cldm.vae.intermediate_features.clear()
                
                # 编码lq图像
                lq_posterior = pure_cldm.vae.encode(lq_img)
                lq_query_features = pure_cldm.vae.get_intermediate_features()
                
                if lq_query_features:
                    lq_features = []
                    for level in fusion_levels:
                        level_key = f'down_{level}'
                        if level_key in lq_query_features:
                            lq_feat = lq_query_features[level_key]
                            lq_features.append(lq_feat.to(device))
                        else:
                            # 如果没有找到对应层级的特征，使用零特征
                            print(f"====================warning=====================")
                            batch_size = clean_img.shape[0]
                            channels = retrieved_features[0].shape[1] if retrieved_features else 128
                            h, w = retrieved_features[0].shape[2:] if retrieved_features else (64, 64)
                            lq_features.append(torch.zeros(batch_size, channels, h, w, device=device))
        
        profiler.stop()
        return retrieved_features, lq_features

    # 开始测试
    if accelerator.is_main_process:
        print("开始测试...")
    
    with torch.no_grad():
        psnr_list = []
        ssim_list = []
        uw_metrics_list = []  # 新增：存储水下指标
        low_psnr_files = []  # 存储PSNR低于阈值的文件信息
        for batch_idx, batch in enumerate(tqdm(test_loader, disable=not accelerator.is_main_process)):
            
            profiler.start("数据预处理")
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, prompt = batch
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()
            profiler.stop()
            
            if train_mode == 'joint':
                # 联合训练模式：使用扩散过程，但VAE解码时使用特征融合
                profiler.start("SCEIR处理")
                clean, *_ = sceir(lq*2 - 1)
                # clean, *_ = sceir(lq)
                profiler.stop()
                
                profiler.start("条件准备")
                cond = pure_cldm.prepare_condition(clean, prompt)
                profiler.stop()
                
                # 采样
                profiler.start("扩散采样")
                z_0 = pure_cldm.vae_encode(gt)
                z = sampler.sample(
                    model=cldm,
                    device=device,
                    steps=cfg.inference.steps,
                    x_size=(len(gt), *z_0.shape[1:]),
                    cond=cond,
                    uncond=None,
                    cfg_scale=cfg.inference.cfg_scale,
                    progress=accelerator.is_main_process,
                )
                profiler.stop()
                
                # 使用特征融合进行解码
                retrieved_features, lq_features = retrieve_features_from_database(clean, lq, feature_storage, batch_idx)
                
                if accelerator.is_main_process and batch_idx == 0:
                    print(f"联合训练模式 - 检索特征数量: {len(retrieved_features) if retrieved_features is not None else 0}")
                    print(f"联合训练模式 - lq特征数量: {len(lq_features) if lq_features is not None else 0}")
                
                profiler.start("VAE解码")
                if retrieved_features is not None:
                    print(f"get retrieved_features")
                    pred = pure_cldm.vae.decode(z, retrieved_features, lq_features)
                else:
                    # 如果没有检索特征，使用普通的解码
                    pred = pure_cldm.vae.decode(z)

                pred = pred + clean
                pred = torch.clamp(pred, 0, 1)
                
            gt_img = (gt + 1) / 2   # [0, 1]
            

            profiler.start("指标计算")
            psnr = calculate_psnr_pt(pred, gt_img, crop_border=0, test_y_channel=False)
            psnr_list.append(psnr.cpu())
            print(f"PSNR: {psnr.item()}")
            ssim = calculate_ssim(pred, gt_img, crop_border=0, test_y_channel=False)
            ssim_list.append(ssim.cpu())
            print(f"SSIM: {ssim.item()}")

            # 检查PSNR是否低于阈值并记录文件信息
            psnr_threshold = getattr(args, 'psnr_threshold', 16.0)
            for i in range(len(gt)):
                current_psnr = psnr[i].item() if psnr.dim() > 0 else psnr.item()
                if current_psnr < psnr_threshold:
                    # 获取原始文件名（从数据集中获取）
                    if hasattr(test_dataset, 'gt_image_files'):
                        # 如果是CodeformerDataset，从文件列表中获取文件名
                        file_info = test_dataset.gt_image_files[batch_idx * cfg.inference.batch_size + i]
                        original_filename = os.path.basename(file_info["image_path"])
                        full_path = file_info["image_path"]
                    else:
                        # 否则使用批次索引作为文件名
                        original_filename = f"test_{batch_idx}_{i}.png"
                        full_path = f"batch_{batch_idx}_index_{i}"
                    
                    low_psnr_files.append({
                        'filename': original_filename,
                        'full_path': full_path,
                        'psnr': current_psnr,
                        'batch_idx': batch_idx,
                        'sample_idx': i
                    })
                    print(f"⚠️  PSNR低于阈值({psnr_threshold}): {original_filename}, PSNR: {current_psnr:.2f}")

            # 计算水下图像质量指标
            uw_metrics = calculate_underwater_metrics_batch(pred, gt_img, device=device)
            uw_metrics_list.append(uw_metrics)
            print(f"水下指标: {uw_metrics}")
            profiler.stop()

            # 保存结果
            if accelerator.is_main_process:
                profiler.start("结果保存")
                for i in range(len(gt)):
                    if hasattr(test_dataset, 'gt_image_files'):
                        file_info = test_dataset.gt_image_files[batch_idx * cfg.inference.batch_size + i]
                        original_filename = os.path.basename(file_info["image_path"])
                    else:
                        original_filename = f"test_{batch_idx}_{i}.png"
                    
                    name_without_ext = os.path.splitext(original_filename)[0]
                    

                    pred_path = os.path.join(output_dir, f"{name_without_ext}.png")
                    save_image(pred[i], pred_path)

                profiler.stop()

        avg_psnr = torch.cat(psnr_list).mean().item()
        print(f"平均PSNR: {avg_psnr}")
        avg_ssim = torch.cat(ssim_list).mean().item()
        print(f"平均SSIM: {avg_ssim}")
        
        avg_uw_metrics = {}
        for key in uw_metrics_list[0].keys():
            avg_uw_metrics[key] = sum(m[key] for m in uw_metrics_list) / len(uw_metrics_list)
        
        print("平均水下图像质量指标:")
        for k, v in avg_uw_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # 输出低PSNR文件统计信息
        if low_psnr_files:
            print(f"\n⚠️  发现 {len(low_psnr_files)} 个PSNR低于阈值({psnr_threshold})的图像:")
            for file_info in low_psnr_files:
                print(f"  - {file_info['filename']} (PSNR: {file_info['psnr']:.2f})")
        else:
            print(f"\n✅ 所有图像的PSNR都高于阈值({psnr_threshold})")
    
    if accelerator.is_main_process:
        profiler.print_stats()
    
    if accelerator.is_main_process:
        print(f"测试完成！结果已保存到: {output_dir}")
        
        results_file = os.path.join(output_dir, "test_results.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("测试结果汇总\n")
            f.write("=" * 50 + "\n")
            f.write(f"平均PSNR: {avg_psnr:.4f}\n")
            f.write(f"平均SSIM: {avg_ssim:.4f}\n")
            f.write("\n水下图像质量指标:\n")
            for k, v in avg_uw_metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            
            f.write("\n时间统计报告:\n")
            f.write("-" * 50 + "\n")
            stats = profiler.get_stats()
            for timer_name, stat in stats.items():
                f.write(f"{timer_name:30s}: 总时间={stat['total']:8.3f}s, "
                       f"平均={stat['avg']:6.3f}s, "
                       f"最小={stat['min']:6.3f}s, "
                       f"最大={stat['max']:6.3f}s, "
                       f"次数={stat['count']:4d}\n")
            
            if low_psnr_files:
                f.write(f"\nPSNR低于阈值({psnr_threshold})的图像列表:\n")
                f.write("-" * 50 + "\n")
                for file_info in low_psnr_files:
                    f.write(f"文件名: {file_info['filename']}\n")
                    f.write(f"完整路径: {file_info['full_path']}\n")
                    f.write(f"PSNR值: {file_info['psnr']:.2f}\n")
                    f.write(f"批次索引: {file_info['batch_idx']}, 样本索引: {file_info['sample_idx']}\n")
                    f.write("-" * 30 + "\n")
            else:
                f.write(f"\n✅ 所有图像的PSNR都高于阈值({psnr_threshold})\n")
        
        print(f"测试结果已保存到: {results_file}")


 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--test_mode", action="store_true", help="启用测试模式")
    parser.add_argument("--checkpoint_path", type=str, help="训练好的ControlNet checkpoint路径")
    parser.add_argument("--vae_fusion_checkpoint", type=str, help="训练好的VAE特征融合权重路径")
    parser.add_argument("--output_dir", type=str, help="测试结果输出目录")
    parser.add_argument("--psnr_threshold", type=float, default=16.0, help="PSNR阈值，低于此值的图像将被标记")
    args = parser.parse_args()
    test_on_file_list(args)
