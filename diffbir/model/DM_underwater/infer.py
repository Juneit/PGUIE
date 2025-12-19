import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from torch.utils.tensorboard import SummaryWriter
import os
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/underwater.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        # 提取文件名信息，避免传递给模型
        hr_filename = val_data['HR_filename'][0] if isinstance(val_data['HR_filename'], list) else val_data['HR_filename']
        
        # 从数据中移除filename字段，避免传递给模型
        model_data = {k: v for k, v in val_data.items() if k != 'HR_filename'}
        print("-------------------------------------------------")
        print("model_data keys:", model_data.keys())
        print("model_data['SR'] shape:", model_data['SR'].shape)
        print("-------------------------------------------------")
        diffusion.feed_data(model_data)
        start = time.time()
        diffusion.test(continous=True)
        end = time.time()
        print('Execution time:', (end - start), 'seconds')
        visuals = diffusion.get_current_visuals(need_LR=False)

        print("-------------------------------------------------")
        print("visuals image keys:", visuals.keys())
        print("visuals['SR'] shape:", visuals['SR'].shape)
        print("-------------------------------------------------")
        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        # 使用输入图像的文件名作为输出图像的保存名称
        file_name = hr_filename
        
        Metrics.save_img(
            Metrics.tensor2img(visuals['SR'][-1]), '{}/{}.png'.format(result_path, file_name)
        )

        # sr_img_mode = 'grid'
        # if sr_img_mode == 'single':
        #     # single img series
        #     sr_img = visuals['SR']  # uint8
        #     sample_num = sr_img.shape[0]
        #     for iter in range(0, sample_num):
        #         Metrics.save_img(
        #             Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        # else:
        #     # grid img
        #     sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
        #     Metrics.save_img(
        #         sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
        #     Metrics.save_img(
        #         Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
            # for i in range(len(visuals['SR'])):
            #     Metrics.save_img(
            #         Metrics.tensor2img(visuals['SR'][i]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, str(i)))

        # Metrics.save_img(
        #     hr_img, '{}/{}_hr.png'.format(result_path, file_name))
        # Metrics.save_img(
        #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
