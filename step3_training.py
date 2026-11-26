import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Mesh_dataset import *
from imeshsegnet import *
from losses_and_metrics_for_mesh import *
import utils
import pandas as pd

if __name__ == '__main__':

    # multi-GPU: launch with `torchrun --nproc_per_node=<num_gpus> step3_training.py`
    # (one process per GPU); `python step3_training.py` still runs on a single GPU
    distributed = 'LOCAL_RANK' in os.environ
    if distributed:
        dist.init_process_group()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    elif torch.cuda.is_available():
        torch.cuda.set_device(utils.get_avail_gpu())
    is_main = not distributed or dist.get_rank() == 0  # only this process logs, validates and saves

    # csv lists of .npz files, see Npz_Mesh_Dataset in Mesh_dataset.py
    train_list = './data/train_list.csv'
    val_list = './data/val_list.csv'

    model_path = './models/'
    model_name = 'iMeshSegNet_17_classes'  # need to define
    checkpoint_name = 'latest_checkpoint.tar'

    num_classes = 17
    num_channels = 15
    num_epochs = 100
    num_workers = 8  # per GPU
    train_batch_size = 4  # per GPU
    val_batch_size = 4
    num_batches_to_print = 20

    os.makedirs(model_path, exist_ok=True)

    if is_main:
        writer = SummaryWriter(log_dir=os.path.join(model_path, 'runs', model_name))

    training_dataset = Npz_Mesh_Dataset(data_list_path=train_list,
                                        patch_size=9000,
                                        augment=True)
    val_dataset = Npz_Mesh_Dataset(data_list_path=val_list,
                                   patch_size=9000)

    train_sampler = DistributedSampler(training_dataset) if distributed else None
    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=train_batch_size,
                              shuffle=train_sampler is None,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers if is_main else 0,
                            pin_memory=True,
                            persistent_workers=is_main and num_workers > 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = iMeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
    opt = optim.Adam(model.parameters(), amsgrad=True)

    losses, mdsc, msen, mppv = [], [], [], []
    val_losses, val_mdsc, val_msen, val_mppv = [], [], [], []
    epoch_init = 0
    best_val_dsc = 0.0

    checkpoint_path = os.path.join(model_path, checkpoint_name)
    if os.path.exists(checkpoint_path):
        print('Resuming from checkpoint: {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_init = checkpoint['epoch']
        losses = checkpoint['losses']
        mdsc = checkpoint['mdsc']
        msen = checkpoint['msen']
        mppv = checkpoint['mppv']
        val_losses = checkpoint['val_losses']
        val_mdsc = checkpoint['val_mdsc']
        val_msen = checkpoint['val_msen']
        val_mppv = checkpoint['val_mppv']
        best_val_dsc = max(val_mdsc)
        del checkpoint

        # replay history into TensorBoard so curves are continuous
        for i_epoch in range(len(losses) if is_main else 0):
            writer.add_scalars('Loss', {'train': losses[i_epoch], 'val': val_losses[i_epoch]}, i_epoch+1)
            writer.add_scalars('DSC',  {'train': mdsc[i_epoch],   'val': val_mdsc[i_epoch]},   i_epoch+1)
            writer.add_scalars('SEN',  {'train': msen[i_epoch],   'val': val_msen[i_epoch]},   i_epoch+1)
            writer.add_scalars('PPV',  {'train': mppv[i_epoch],   'val': val_mppv[i_epoch]},   i_epoch+1)
    else:
        print('No checkpoint found, starting from scratch.')

    model = model.to(device, dtype=torch.float)
    # train_model syncs gradients across GPUs; model keeps the plain weights for validation/saving
    train_model = model
    if distributed:
        train_model = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)
    # mixed precision (AMP): the model runs in half precision on GPU, the loss stays in float32
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print('Training model...')
    class_weights = torch.ones(num_classes).to(device, dtype=torch.float)
    for epoch in range(epoch_init, num_epochs):

        # training
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_mdsc = 0.0
        running_msen = 0.0
        running_mppv = 0.0
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0
        for i_batch, batched_sample in enumerate(train_loader):

            inputs = batched_sample['cells'].to(device, dtype=torch.float, non_blocking=True)
            labels = batched_sample['labels'].to(device, dtype=torch.long, non_blocking=True)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

            opt.zero_grad()
            with torch.autocast(device.type, enabled=use_amp):
                outputs = train_model(inputs)
            outputs = outputs.float()
            loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            running_mdsc += dsc.item()
            running_msen += sen.item()
            running_mppv += ppv.item()
            loss_epoch += loss.item()
            mdsc_epoch += dsc.item()
            msen_epoch += sen.item()
            mppv_epoch += ppv.item()

            if is_main and i_batch % num_batches_to_print == num_batches_to_print - 1:
                print('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}'.format(
                    epoch+1, num_epochs, i_batch+1, len(train_loader),
                    running_loss/num_batches_to_print, running_mdsc/num_batches_to_print,
                    running_msen/num_batches_to_print, running_mppv/num_batches_to_print))
                global_step = epoch * len(train_loader) + i_batch + 1
                writer.add_scalar('Loss/train_batch', running_loss/num_batches_to_print, global_step)
                writer.add_scalar('DSC/train_batch',  running_mdsc/num_batches_to_print, global_step)
                writer.add_scalar('SEN/train_batch',  running_msen/num_batches_to_print, global_step)
                writer.add_scalar('PPV/train_batch',  running_mppv/num_batches_to_print, global_step)
                running_loss = running_mdsc = running_msen = running_mppv = 0.0

        if distributed:  # average the epoch sums over all GPUs
            sums = torch.tensor([loss_epoch, mdsc_epoch, msen_epoch, mppv_epoch], device=device)
            dist.all_reduce(sums)
            loss_epoch, mdsc_epoch, msen_epoch, mppv_epoch = (sums / dist.get_world_size()).tolist()
        if not is_main:
            continue

        losses.append(loss_epoch/len(train_loader))
        mdsc.append(mdsc_epoch/len(train_loader))
        msen.append(msen_epoch/len(train_loader))
        mppv.append(mppv_epoch/len(train_loader))

        # validation
        model.eval()
        with torch.no_grad():
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0
            running_val_loss = 0.0
            running_val_mdsc = 0.0
            running_val_msen = 0.0
            running_val_mppv = 0.0
            for i_batch, batched_val_sample in enumerate(val_loader):

                inputs = batched_val_sample['cells'].to(device, dtype=torch.float, non_blocking=True)
                labels = batched_val_sample['labels'].to(device, dtype=torch.long, non_blocking=True)
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

                with torch.autocast(device.type, enabled=use_amp):
                    outputs = model(inputs)
                outputs = outputs.float()
                loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

                running_val_loss += loss.item()
                running_val_mdsc += dsc.item()
                running_val_msen += sen.item()
                running_val_mppv += ppv.item()
                val_loss_epoch += loss.item()
                val_mdsc_epoch += dsc.item()
                val_msen_epoch += sen.item()
                val_mppv_epoch += ppv.item()

                if i_batch % num_batches_to_print == num_batches_to_print - 1:
                    print('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}, val_sen: {6}, val_ppv: {7}'.format(
                        epoch+1, num_epochs, i_batch+1, len(val_loader),
                        running_val_loss/num_batches_to_print, running_val_mdsc/num_batches_to_print,
                        running_val_msen/num_batches_to_print, running_val_mppv/num_batches_to_print))
                    running_val_loss = running_val_mdsc = running_val_msen = running_val_mppv = 0.0

            val_losses.append(val_loss_epoch/len(val_loader))
            val_mdsc.append(val_mdsc_epoch/len(val_loader))
            val_msen.append(val_msen_epoch/len(val_loader))
            val_mppv.append(val_mppv_epoch/len(val_loader))

            print('*****\nEpoch: {}/{}, loss: {}, dsc: {}, sen: {}, ppv: {}\n         val_loss: {}, val_dsc: {}, val_sen: {}, val_ppv: {}\n*****'.format(
                epoch+1, num_epochs, losses[-1], mdsc[-1], msen[-1], mppv[-1],
                val_losses[-1], val_mdsc[-1], val_msen[-1], val_mppv[-1]))
            writer.add_scalars('Loss', {'train': losses[-1],  'val': val_losses[-1]}, epoch+1)
            writer.add_scalars('DSC',  {'train': mdsc[-1],    'val': val_mdsc[-1]},   epoch+1)
            writer.add_scalars('SEN',  {'train': msen[-1],    'val': val_msen[-1]},   epoch+1)
            writer.add_scalars('PPV',  {'train': mppv[-1],    'val': val_mppv[-1]},   epoch+1)

        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses, 'mdsc': mdsc, 'msen': msen, 'mppv': mppv,
                    'val_losses': val_losses, 'val_mdsc': val_mdsc,
                    'val_msen': val_msen, 'val_mppv': val_mppv},
                   checkpoint_path)

        if best_val_dsc < val_mdsc[-1]:
            best_val_dsc = val_mdsc[-1]
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'losses': losses, 'mdsc': mdsc, 'msen': msen, 'mppv': mppv,
                        'val_losses': val_losses, 'val_mdsc': val_mdsc,
                        'val_msen': val_msen, 'val_mppv': val_mppv},
                       os.path.join(model_path, '{}_best.tar'.format(model_name)))

        pd.DataFrame({'loss': losses, 'DSC': mdsc, 'SEN': msen, 'PPV': mppv,
                      'val_loss': val_losses, 'val_DSC': val_mdsc,
                      'val_SEN': val_msen, 'val_PPV': val_mppv}).to_csv('losses_metrics_vs_epoch.csv')

    if is_main:
        writer.close()
