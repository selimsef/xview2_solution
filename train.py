import argparse
import os

from tools.xview_metric import XviewMetrics

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import models
from augs import Lighting, RandomSizedCropAroundBbox

from albumentations import Compose, HorizontalFlip, VerticalFlip, RGBShift, RandomBrightnessContrast, \
    RandomGamma, RandomRotate90, Transpose

import losses
from dataset.xview_dataset import XviewSingleDataset

from apex.parallel import DistributedDataParallel, convert_syncbn_model
from tensorboardX import SummaryWriter

from tools.config import load_config
from tools.utils import create_optimizer, AverageMeter

from apex import amp

from losses import dice_round

import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist

torch.backends.cudnn.benchmark = True


def create_train_transforms(conf):
    height = conf['crop_height']
    width = conf['crop_width']
    return Compose([
        RandomSizedCropAroundBbox(min_max_height=(int(height * 0.8), int(height * 1.2)), w2h_ratio=1., height=height,
                               width=width, p=1),
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        Transpose(),
        Lighting(alphastd=0.3),
        RandomBrightnessContrast(p=0.2),
        RandomGamma(p=0.2),
        RGBShift(p=0.2)
    ], additional_targets={'image1': 'image'}
    )


def create_val_transforms(conf):
    return Compose([
    ], additional_targets={'image1': 'image'})


def main():
    parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
    arg('--workers', type=int, default=6, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='damage_')
    arg('--data-dir', type=str, default="/home/selim/datasets/xview/train")
    arg('--folds-csv', type=str, default='folds.csv')
    arg('--logdir', type=str, default='logs')
    arg('--zero-score', action='store_true', default=False)
    arg('--from-zero', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg('--freeze-epochs', type=int, default=1)
    arg("--local_rank", default=0, type=int)
    arg("--opt-level", default='O1', type=str)
    arg("--predictions", default="../oof_preds", type=str)
    arg("--test_every", type=int, default=1)

    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True

    conf = load_config(args.config)
    model = models.__dict__[conf['network']](seg_classes=conf['num_classes'], backbone_arch=conf['encoder'])

    model = model.cuda()
    if args.distributed:
        model = convert_syncbn_model(model)
    damage_loss_function = losses.__dict__[conf["damage_loss"]["type"]](**conf["damage_loss"]["params"]).cuda()
    mask_loss_function = losses.__dict__[conf["mask_loss"]["type"]](**conf["mask_loss"]["params"]).cuda()
    loss_functions = {"damage_loss": damage_loss_function, "mask_loss": mask_loss_function}
    optimizer, scheduler = create_optimizer(conf['optimizer'], model)

    dice_best = 0
    xview_best = 0
    start_epoch = 0
    batch_size = conf['optimizer']['batch_size']

    data_train = XviewSingleDataset(mode="train",
                                    fold=args.fold,
                                    data_path=args.data_dir,
                                    folds_csv=args.folds_csv,
                                    transforms=create_train_transforms(conf['input']),
                                    multiplier=conf["data_multiplier"],
                                    normalize=conf["input"].get("normalize", None))
    data_val = XviewSingleDataset(mode="val",
                                  fold=args.fold,
                                  data_path=args.data_dir,
                                  folds_csv=args.folds_csv,
                                  transforms=create_val_transforms(conf['input']),
                                  normalize=conf["input"].get("normalize", None)
                                  )
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=args.workers,
                                   shuffle=train_sampler is None, sampler=train_sampler, pin_memory=False,
                                   drop_last=True)
    val_batch_size = 1
    val_data_loader = DataLoader(data_val, batch_size=val_batch_size, num_workers=args.workers, shuffle=False,
                                 pin_memory=False)

    os.makedirs(args.logdir, exist_ok=True)
    summary_writer = SummaryWriter(args.logdir + '/' + args.prefix + conf['encoder'])
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['state_dict']
            if conf['optimizer'].get('zero_decoder', False):
                for key in state_dict.copy().keys():
                    if key.startswith("module.final"):
                        del state_dict[key]
            state_dict = {k[7:]: w for k, w in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            if not args.from_zero:
                start_epoch = checkpoint['epoch']
                if not args.zero_score:
                    dice_best = checkpoint.get('dice_best', 0)
                    xview_best = checkpoint.get('xview_best', 0)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.from_zero:
        start_epoch = 0
    current_epoch = start_epoch

    if conf['fp16']:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          loss_scale='dynamic')

    snapshot_name = "{}{}_{}_{}".format(args.prefix, conf['network'], conf['encoder'], args.fold)

    if args.distributed:
        model = DistributedDataParallel(model, delay_allreduce=True)
    else:
        model = DataParallel(model).cuda()
    for epoch in range(start_epoch, conf['optimizer']['schedule']['epochs']):
        if epoch < args.freeze_epochs:
            print("Freezing encoder!!!")
            if hasattr(model.module, 'encoder_stages1'):
                model.module.encoder_stages1.eval()
                model.module.encoder_stages2.eval()
                for p in model.module.encoder_stages1.parameters():
                    p.requires_grad = False
                for p in model.module.encoder_stages2.parameters():
                    p.requires_grad = False
            else:
                model.module.encoder_stages.eval()
                for p in model.module.encoder_stages.parameters():
                    p.requires_grad = False
        else:
            if hasattr(model.module, 'encoder_stages1'):
                print("Unfreezing encoder!!!")
                model.module.encoder_stages1.train()
                model.module.encoder_stages2.train()
                for p in model.module.encoder_stages1.parameters():
                    p.requires_grad = True
                for p in model.module.encoder_stages2.parameters():
                    p.requires_grad = True
            else:
                model.module.encoder_stages.train()
                for p in model.module.encoder_stages.parameters():
                    p.requires_grad = True
        train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                    args.local_rank)

        model = model.eval()
        if args.local_rank == 0:
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'dice_best': dice_best,
                'xview_best': xview_best,
            }, args.output_dir + '/' + snapshot_name + "_last")
            if epoch % args.test_every == 0:
                preds_dir = os.path.join(args.predictions, snapshot_name)
                dice_best, xview_best = evaluate_val(args, val_data_loader, xview_best, dice_best, model,
                                                     snapshot_name=snapshot_name,
                                                     current_epoch=current_epoch,
                                                     optimizer=optimizer, summary_writer=summary_writer,
                                                     predictions_dir=preds_dir)
        current_epoch += 1


def evaluate_val(args, data_val, xview_best, dice_best, model, snapshot_name, current_epoch, optimizer, summary_writer,
                 predictions_dir):
    print("Test phase")
    model = model.eval()
    dice, xview_score = validate(model, data_loader=data_val, predictions_dir=predictions_dir)
    if args.local_rank == 0:
        summary_writer.add_scalar('val/dice', float(dice), global_step=current_epoch)
        summary_writer.add_scalar('val/xview_score', float(xview_score), global_step=current_epoch)
        if dice > dice_best:
            if args.output_dir is not None:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'dice_best': dice,
                    'xview_best': xview_score,

                }, args.output_dir + snapshot_name + "_best_dice")
            dice_best = dice
        if xview_score > xview_best:
            if args.output_dir is not None:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'dice_best': dice,
                    'xview_best': xview_score,
                }, args.output_dir + snapshot_name + "_best_xview")
            xview_best = xview_score
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'dice_best': dice_best,
            'xview_best': xview_best,
        }, args.output_dir + snapshot_name + "_last")
        print("dice: {}, dice_best: {}".format(dice, dice_best))
        print("xview: {}, xview_best: {}".format(xview_score, xview_best))
    return dice_best, xview_best


def validate(net, data_loader, predictions_dir):
    os.makedirs(predictions_dir, exist_ok=True)
    preds_dir = predictions_dir + "/predictions"
    os.makedirs(preds_dir, exist_ok=True)
    targs_dir = predictions_dir + "/targets"
    os.makedirs(targs_dir, exist_ok=True)
    with torch.no_grad():
        for sample in tqdm(data_loader):
            imgs = sample["image"].cuda().float()
            mask = sample["mask"].cuda().float()
            original_mask = sample["original_mask"].cuda().long().cpu().numpy()

            output = net(imgs)
            binary_pred = torch.sigmoid(output[:, 4:, ...])

            damage_preds = torch.sigmoid(output[:, :4, ...]).cpu().numpy()
            for i in range(output.shape[0]):
                damage_pred = damage_preds[i]
                first = np.zeros((1, 1024, 1024))
                first[:, :, :] = 0.1
                damage_pred = np.concatenate([first, damage_pred], axis=0)
                cv2.imwrite(os.path.join(preds_dir,
                                         "test_localization_" + sample["img_name"][i] + "_prediction.png"),
                            (binary_pred[i, 0].cpu().numpy() > 0.3) * 1)
                cv2.imwrite(os.path.join(preds_dir,
                                         "test_damage_" + sample["img_name"][i] + "_prediction.png"),
                            np.argmax(damage_pred, axis=0))
                cv2.imwrite(os.path.join(targs_dir,
                                         "test_localization_" + sample["img_name"][i] + "_target.png"),
                            mask.cpu().numpy()[i, 4])
                cv2.imwrite(
                    os.path.join(targs_dir, "test_damage_" + sample["img_name"][i] + "_target.png"),
                    original_mask[i])
    d = XviewMetrics.compute_score(preds_dir, targs_dir, "out.json")
    for k, v in d.items():
        print("{}:{}".format(k, v))
    return d["localization_f1"], d["score"]


def train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                local_rank):
    losses = AverageMeter()
    damage_f1 = AverageMeter()
    localization_f1 = AverageMeter()
    iterator = tqdm(train_data_loader)
    model.train()
    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(current_epoch)
    for i, sample in enumerate(iterator):
        imgs = sample["image"].cuda()
        masks = sample["mask"].cuda().float()
        out_mask = model(imgs)
        mask_band = 4
        with torch.no_grad():
            pred = torch.sigmoid(out_mask[:, :, ...])
            d = dice_round(pred[:, mask_band:, ...], masks[:, mask_band:, ...], t=0.5).item()
            loc_f1 = 0
            for i in range(4):
                loc_f1 += 1/(dice_round(pred[:, i:i+1, ...], masks[:, i:i+1, ...], t=0.3).item() + 1e-3)
            loc_f1 = 4/loc_f1
        localization_f1.update(d, imgs.size(0))
        damage_f1.update(loc_f1, imgs.size(0))

        mask_loss = loss_functions["mask_loss"](out_mask[:, mask_band:, ...].contiguous(),
                                                masks[:, mask_band:, ...].contiguous())
        damage_loss = loss_functions["damage_loss"](out_mask[:, :mask_band, ...].contiguous(),
                                                    masks[:, :mask_band, ...].contiguous())
        loss = 0.7 * damage_loss + 0.3 * mask_loss
        losses.update(loss.item(), imgs.size(0))
        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss ({loss.avg:.4f}); Localization F1 ({dice.avg:.4f}); Damage F1 ({damage.avg:.4f}); ".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, dice=localization_f1, damage=damage_f1))
        optimizer.zero_grad()
        if conf['fp16']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
        optimizer.step()
        torch.cuda.synchronize()
        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * len(train_data_loader))

    if local_rank == 0:
        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=current_epoch)
        summary_writer.add_scalar('train/loss', float(losses.avg), global_step=current_epoch)


if __name__ == '__main__':
    main()
