import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

import torch
import torch.utils.data
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
# import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage

from utils.Loggers import get_logger
from utils.train_options import TrainOptions
from utils.dataloader import ViewDataset, PairedDataset
from trainer import Trainer

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from utils.utils import *

def main(opt, writer, opt_message):
    set_seed(1234)

    #######[ Save ]############################################################################
    save_out         = os.path.join(opt.out, 'generated')
    save_train       = os.path.join(opt.out, 'trained')
    os.makedirs(save_out,   exist_ok=True)
    os.makedirs(save_train, exist_ok=True)
    
    #######[ Network Trainer ]#################################################################
    opt.device_id    = [i for i in range(torch.cuda.device_count())]
    device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer          = Trainer(opt=opt, device=device)
    print(f'current device: {device}')
    
    #######[ Logger ]##########################################################################
    # current date and time
    now              = datetime.now() 
    date             = now.strftime("%m_%d_%H-%M")
    log_path         = opt.out+'/{}.log'
    train_log        = get_logger("{}".format(date), path=log_path)
    log_format       = '[{}][Data:{:>4}][{:>6}/{:>6}]'
    # share log with trainer
    trainer.train_log= train_log
    
    #######[ Data ]############################################################################
    # trainset_tex     = PairedDataset(opt=opt,    log=train_log)
    # validset_tex     = PairedDataset(opt=opt,    test=True)
    trainset_tex     = ViewDataset(opt=opt,    log=train_log)
    validset_tex     = ViewDataset(opt=opt,    test=True)
    trainloader_tex  = DataLoader(trainset_tex, batch_size=opt.batch_size, shuffle=True,  drop_last=True, num_workers=4)
    validloader_tex  = DataLoader(validset_tex, batch_size=opt.batch_size, shuffle=False, drop_last=True, num_workers=1)
    #######[ Logger ]##########################################################################
    iter_train       = iter(trainloader_tex)
    data_len         = len(trainloader_tex)
    dataV_len        = len(validloader_tex)
    dataV_len_div    = 1/dataV_len

    fin_idx          = opt.epochs
    idx              = trainer.model_on_one_gpu.continue_epoch

    # best_score       = {'lpips':1,'psnr':0,'ssim':0,'idx':0}
    best_score       = {'loss_mean':1, 'idx':0} # init # low loss_mean is better
    grid_column      = 1

    info_tensorboard = "tensorboard --logdir_spec={}:\'{}\' --port {} --host=0.0.0.0".format(opt.usermemo, opt.out, opt.port)
    
    if not opt.continue_train:
        train_log.info(opt_message)
    train_log.info(info_tensorboard)
    train_log.info(f"starting iter from: {idx}")

    # train iteration
    while idx < fin_idx:
        idx = idx if opt.mode == 'debug' else idx+1

        # get data
        try:
            data       = next(iter_train)
        except StopIteration:
            iter_train = iter(trainloader_tex)
            data       = next(iter_train)

        trainer.run_generator_one_step(data)
        trainer.run_discriminator_one_step(data)

        #######[ Log ]#########################################################################
        if idx % 100 == 0:
            logMessage = log_format.format('Train', data_len, idx, fin_idx)
            with torch.no_grad():
                g_losses, d_losses = trainer.get_latest_losses()

            # [ TENSORBOARD ] Generator
            for k, v in g_losses.items():
                logMessage += '  {}: {:<10.4f}'.format(k, v.mean())
                writer['Train'].add_scalar('Generator/'+k, v.mean(), global_step=(idx))
            
            # [ TENSORBOARD ] Discriminator
            for k, v in d_losses.items():
                logMessage += '  {}: {:<10.4f}'.format(k, v.mean())
                writer['Train'].add_scalar('Discriminator/'+k, v.mean(), global_step=(idx))
            
            # Log message
            train_log.info(logMessage)        

        #######[ Validation ]##################################################################
        if idx % opt.valid_iter == 0:
            logMessage = log_format.format('Total', dataV_len, idx, fin_idx)
            g_loss_temp = {}
            d_loss_temp = {}
            eval_temp   = {}

            for dataV in validloader_tex: # dataV.keys() : ['T_inputA', 'T_inputB', 'Vis_maskA', 'Vis_maskB', 'GT_texture', 'norm_map', 'has_GT', 'maskingA', 'maskingB'])
                g_loss, d_loss, lpipsA, psnrA, ssimA, lpipsB, psnrB, ssimB = trainer.get_validation_result(dataV)

                if len(eval_temp) == 0:
                    g_loss_temp = g_loss # dict_keys(['L1_A', 'L1_B', 'lpipsA', 'lpipsB'])
                    d_loss_temp = d_loss
                    eval_temp   = {**lpipsA, **psnrA, **ssimA, **lpipsB, **psnrB, **ssimB}
                else:
                    for k, v in g_loss.items():
                        g_loss_temp[k] = g_loss_temp[k] + v 
                    for k, v in d_loss.items():
                        d_loss_temp[k] = d_loss_temp[k] + v
                    for k, v in {**lpipsA, **psnrA, **ssimA, **lpipsB, **psnrB, **ssimB}.items():
                        eval_temp[k]   = eval_temp[k] + v 

            # average value
            g_loss_temp = {k:v*dataV_len_div for k,v in g_loss_temp.items()}
            d_loss_temp = {k:v*dataV_len_div for k,v in d_loss_temp.items()}
            eval_temp   = {k:v*dataV_len_div for k,v in eval_temp.items()}
            
            # [ TENSORBOARD ] Generator 
            for k, v in g_loss_temp.items():
                logMessage += '  {}: {:<10.4f}'.format(k, v.mean())
                writer['Valid'].add_scalar('Generator/'+k, v.mean(), global_step=(idx))
            # [ TENSORBOARD ] Discriminator
            for k, v in d_loss_temp.items():
                logMessage += '  {}: {:<10.4f}'.format(k, v.mean())
                writer['Valid'].add_scalar('Discriminator/'+k, v.mean(), global_step=(idx))
            # [ TENSORBOARD ] validation result
            for k, v in eval_temp.items():
                logMessage +='\n' if k in ['lpips'] else ''
                logMessage += '  {}: {:<10.4f}'.format(k, v.mean())
                writer['Valid'].add_scalar('Valdiation/'+k, v.mean(), global_step=(idx))
            
            # Log message
            train_log.info(logMessage)# """

            ###[ Visualization ]###############################################################
            test_inputA,  test_outputA, test_inputB,  test_outputB  = trainer.model_on_one_gpu(dataV, mode='visualize_valid')
            train_inputA, train_outputA, train_inputB, train_outputB = trainer.model_on_one_gpu(data,  mode='visualize_train')

            test_inputA  = make_grid(test_inputA.cpu(),  nrow=grid_column, padding=0)
            train_inputA = make_grid(train_inputA.cpu(), nrow=grid_column, padding=0)
            test_inputB  = make_grid(test_inputB.cpu(),  nrow=grid_column, padding=0)
            train_inputB = make_grid(train_inputB.cpu(), nrow=grid_column, padding=0)

            test_outputA = make_grid(test_outputA.cpu(), nrow=grid_column, padding=0)
            test_outputB = make_grid(test_outputB.cpu(), nrow=grid_column, padding=0)
            test_pmaskA  = make_grid(dataV['Vis_maskA'], nrow=grid_column, padding=0)
            test_pmaskB  = make_grid(dataV['Vis_maskB'], nrow=grid_column, padding=0)
            test_target  = make_grid(dataV['GT_texture'],nrow=grid_column, padding=0)
                
            train_outputA= make_grid(train_outputA.cpu(),nrow=grid_column, padding=0)
            train_outputB= make_grid(train_outputB.cpu(),nrow=grid_column, padding=0)
            train_pmaskA = make_grid(data['Vis_maskA'],  nrow=grid_column, padding=0)
            train_pmaskB = make_grid(data['Vis_maskB'],  nrow=grid_column, padding=0)
            train_target = make_grid(data['GT_texture'], nrow=grid_column, padding=0)
            
            writer['Valid'].add_image("T_inputA",  test_inputA,   global_step=(idx))
            writer['Valid'].add_image("T_inputB",  test_inputB,   global_step=(idx))
            writer['Valid'].add_image("Vis_maskA", test_pmaskA,   global_step=(idx))
            writer['Valid'].add_image("Vis_maskB", test_pmaskB,   global_step=(idx))
            writer['Valid'].add_image("outputA",   test_outputA,  global_step=(idx))
            writer['Valid'].add_image("outputB",   test_outputB,  global_step=(idx))
            writer['Valid'].add_image("target",    test_target,   global_step=(idx))                
            
            writer['Train'].add_image("T_inputA",  train_inputA,  global_step=(idx))
            writer['Train'].add_image("T_inputB",  train_inputB,  global_step=(idx))
            writer['Train'].add_image("Vis_maskA", train_pmaskA,  global_step=(idx))
            writer['Train'].add_image("Vis_maskB", train_pmaskB,  global_step=(idx))
            writer['Train'].add_image("outputA",   train_outputA, global_step=(idx))
            writer['Train'].add_image("outputB",   train_outputB, global_step=(idx))
            writer['Train'].add_image("target",    train_target,  global_step=(idx))

            ###[ SAVE LOCAL ]##################################################################
            test_img_nameA   = '{}/test_iter{:06}A.png'.format(save_out,  idx)
            test_img_nameB   = '{}/test_iter{:06}B.png'.format(save_out,  idx)
            test_resultA     = torch.cat((test_inputA, test_outputA, test_target), dim=-1)
            test_resultB     = torch.cat((test_inputB, test_outputB, test_target), dim=-1)
            ToPILImage()(test_resultA).save(test_img_nameA, 'PNG')
            ToPILImage()(test_resultB).save(test_img_nameB, 'PNG')

            train_img_nameA  = '{}/train_iter{:06}A.png'.format(save_train, idx)
            train_img_nameB  = '{}/train_iter{:06}B.png'.format(save_train, idx)
            train_resultA    = torch.cat((train_inputA, train_outputA, train_target), dim=-1)
            train_resultB    = torch.cat((train_inputB, train_outputB, train_target), dim=-1)
            ToPILImage()(train_resultA).save(train_img_nameA, 'PNG')
            ToPILImage()(train_resultB).save(train_img_nameB, 'PNG')

            ###[ Save model ]##################################################################
            # new_record = best_score['ssimA'] < eval_temp['ssimA'] 
            # eval_temp   = {**lpipsA, **psnrA, **ssimA, **lpipsB, **psnrB, **ssimB}
            
            # g_loss_temp.keys() dict_keys(['L1_A', 'L1_B', 'lpipsA', 'lpipsB']) # we need!!
            loss_mean = (g_loss_temp['L1_A'] + g_loss_temp['L1_B'] + lpipsA['lpips'] + lpipsB['lpips'])/4
            new_record = best_score['loss_mean'] > loss_mean

            if new_record:
                # best_score = { **eval_temp, 'idx':idx }
                best_score = {'loss_mean':loss_mean, 'idx':idx}
                trainer.save_model(path=opt.out, epoch=idx) # overwrite best one
                train_log.info("Best Score~! model saved successfully")                    
            else:
                best_record = ''
                for k,v in best_score.items():
                    best_record += '  {}: {:<10.4f}'.format(k, v)
                train_log.info(f"Current Best: {best_record}")

            ###[ UPDATE LEARNING RATE ]########################################################
            trainer.update_learning_rate(idx)

        # reset best score
        if idx == 8000:
            # best_score = {'lpips':1, 'psnr':0, 'ssim':0, 'idx':0}
            best_score = {'loss_mean':1, 'idx':0} # init # low loss_mean is better
        if idx == 20000:
            trainer.save_model(path=opt.out, epoch=idx)

        #######[ UPDATE PROGRESS ]#############################################################
        if opt.progressive:
            if idx % 4000 == 0:
                trainer.update_progress()

        if opt.mode == 'debug':
            break

    trainer.save_model(path=opt.out, epoch=idx) # save the last one
    train_log.info(f"saving latest model")

    bests = ''               
    for k, v in best_score.items():
        bests += '{}: {:<10.4f}'.format(k, v)
    train_log.info(f"Final Best Model: {bests}") 
    train_log.info(info_tensorboard)

if __name__ == "__main__":
    #######[ Parser ]##########################################################################
    train_opt   = TrainOptions()
    opt         = train_opt.parse()
    opt_message = train_opt.message

    #######[ Output ]##########################################################################
    output_folder = os.path.join(os.getcwd(), "output", opt.out)

    print(f'setting output directory: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)
    opt.out = output_folder

    #######[ Summary Writer ]##################################################################
    writer = {}
    for loss in ['Train', 'Valid']:
        log_dir = os.path.join(output_folder, loss)
        writer[loss] = SummaryWriter(log_dir)

    main(opt, writer, opt_message)