import os
import time
import random
import json
from tqdm import tqdm

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from configs.opts import parser
from model.temp_video_modelv3 import Temp_Model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_datasetV3 import AVEDatasetV2
import torch.nn.functional as F

# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

config_path = 'configs/main.json'
with open(config_path) as fp:
    config = json.load(fp)
print(config)




def main():
    # utils variable
    global args, logger, writer, dataset_configs

    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0

    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    '''Create snapshot_pred dir for copying code and saving models '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')

    '''Dataset'''
    train_dataloader = DataLoader(
        AVEDatasetV2('./data/', split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        AVEDatasetV2('./data/', split='test'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    '''model setting'''
    mainModel = main_model(in_channels=10,feature_dim=768)

    mainModel = nn.DataParallel(mainModel).cuda()
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)

    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError

    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    '''Tensorboard and Code backup'''
    writer = SummaryWriter(args.snapshot_pref)


    '''Training and Testing'''
    for epoch in range(args.n_epoch):
        loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)
        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            test_list.clear()
            acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch)
            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='Supervised',
                    epoch=epoch + 1,
                )
            print("-----------------------------")
            print("best acc and epoch:", best_accuracy, best_accuracy_epoch)
            print("-----------------------------")
            test_list.clear()
        scheduler.step()


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    kl_losses = AverageMeter()
    end_time = time.time()

    model.train()

    model.double()
    optimizer.zero_grad()


    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, text_feat, pseudo_label, audio_feat , audio_text_feat, audio_pseudo_label, labels = batch_data
        bs = visual_feature.shape[0]
        labels = labels.double().cuda()
        pseudo_label = pseudo_label.double().cuda()
        visual_feature = visual_feature.double().cuda()
        audio_feat = audio_feat.double().cuda()
        audio_pseudo_label = audio_pseudo_label.double().cuda()
        is_event_scores, event_scores, kl_loss, vis_is_event_scores, vis_event_scores, audio_is_event_scores, audio_event_scores  = model(visual_feature, text_feat,
                                                                                                                                          audio_feat, audio_text_feat)
        is_event_scores = is_event_scores.squeeze().contiguous()
        is_event_scores = is_event_scores.reshape(bs, -1)
        vis_is_event_scores = vis_is_event_scores.squeeze().contiguous()
        vis_is_event_scores = vis_is_event_scores.reshape(bs, -1)
        audio_is_event_scores = audio_is_event_scores.squeeze().contiguous()
        audio_is_event_scores = audio_is_event_scores.reshape(bs, -1)


        labels_foreground = pseudo_label
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_BCE = labels_BCE.reshape(bs,-1)
        labels_event, _ = labels_evn.max(-1)

        audio_labels_foreground = audio_pseudo_label
        audio_labels_BCE, audio_labels_evn = audio_labels_foreground.max(-1)
        audio_labels_BCE = audio_labels_BCE.reshape(bs, -1)
        audio_labels_event, _ = audio_labels_evn.max(-1)

        loss_is_event = criterion(is_event_scores.reshape(bs,-1), labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        vis_loss_is_event = criterion(vis_is_event_scores.reshape(bs, -1), labels_BCE.cuda())
        vis_loss_event_class = criterion_event(vis_event_scores, labels_event.cuda())
        audio_loss_is_event = criterion(audio_is_event_scores.reshape(bs, -1), audio_labels_BCE.cuda())
        audio_loss_event_class = criterion_event(audio_event_scores, audio_labels_event.cuda())

        loss = loss_is_event + loss_event_class + kl_loss + vis_loss_is_event + vis_loss_event_class + audio_loss_is_event + audio_loss_event_class

        loss.backward()

        '''Compute Accuracy'''
        acc = compute_accuracy_supervised(is_event_scores, event_scores, pseudo_label)
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)


        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()


        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'

                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg: .3f})'
            )

        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)
    logger.info(
            f'**************************************************************************\t'
            f"\tTrain results (acc): {train_acc.avg:.4f}%."

        )
    return losses.avg

test_list = []
@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()
    kl_losses = AverageMeter()
    model.eval()
    model.double()



    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feature, text_feat, pseudo_label, audio_feat, audio_text_feat, audio_pseudo_label, labels = batch_data
        bs = visual_feature.shape[0]
        labels = labels.double().cuda()
        pseudo_label = pseudo_label.double().cuda()
        visual_feature = visual_feature.double().cuda()
        audio_feat = audio_feat.double().cuda()
        audio_pseudo_label = audio_pseudo_label.double().cuda()
        is_event_scores, event_scores, kl_loss, vis_is_event_scores, vis_event_scores, audio_is_event_scores, audio_event_scores = model(
            visual_feature, text_feat, audio_feat, audio_text_feat)
        is_event_scores = is_event_scores.squeeze().contiguous()
        is_event_scores = is_event_scores.reshape(bs, -1)
        vis_is_event_scores = vis_is_event_scores.squeeze().contiguous()
        vis_is_event_scores = vis_is_event_scores.reshape(bs, -1)
        audio_is_event_scores = audio_is_event_scores.squeeze().contiguous()
        audio_is_event_scores = audio_is_event_scores.reshape(bs, -1)


        labels_foreground = pseudo_label
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_BCE = labels_BCE.reshape(bs, -1)
        labels_event, _ = labels_evn.max(-1)

        audio_labels_foreground = audio_pseudo_label
        audio_labels_BCE, audio_labels_evn = audio_labels_foreground.max(-1)
        audio_labels_BCE = audio_labels_BCE.reshape(bs, -1)
        audio_labels_event, _ = audio_labels_evn.max(-1)

        loss_is_event = criterion(is_event_scores.reshape(bs, -1), labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        vis_loss_is_event = criterion(vis_is_event_scores.reshape(bs, -1), labels_BCE.cuda())
        vis_loss_event_class = criterion_event(vis_event_scores, labels_event.cuda())
        audio_loss_is_event = criterion(audio_is_event_scores.reshape(bs, -1), audio_labels_BCE.cuda())
        audio_loss_event_class = criterion_event(audio_event_scores, audio_labels_event.cuda())

        loss = loss_is_event + loss_event_class + kl_loss + vis_loss_is_event + vis_loss_event_class + audio_loss_is_event + audio_loss_event_class

        acc = compute_accuracy_supervised(is_event_scores, event_scores, pseudo_label)
        accuracy.update(acc.item(), bs * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(test_dataloader)}]\t'

                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})'
            )

    '''Add loss in an epoch to Tensorboard'''
    if not eval_only:
        writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)

    logger.info(
        f'**************************************************************************\t'
        f"\tEvaluation results (acc): {accuracy.avg:.4f}%."
        f"\t results (kl_loss): {kl_losses.avg:.4f}%."
    )
    return accuracy.avg


def compute_accuracy_supervised(is_event_scores, event_scores, labels):
    _, targets = labels.max(-1)
    # pos pred
    is_event_scores = is_event_scores.sigmoid()
    scores_pos_ind = is_event_scores > 0.5
    scores_mask = scores_pos_ind == 0
    _, event_class = event_scores.max(-1)
    pred = scores_pos_ind.long()
    pred *= event_class[:, None]

    pred[scores_mask] = 28  # 28 denotes bg
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())

    return acc

def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)



if __name__ == '__main__':
    main()



