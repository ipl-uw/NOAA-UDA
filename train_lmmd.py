import time
from itertools import cycle

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import os
from util import calculate_num_class, hierarchy_dict, compute_accuracy_model7_track_based_level_2_only, track_based_accuracy_level2_only

from IPython import embed
from fish_rail_dataloader_track_based import Fish_Rail_Dataset, BalancedBatchSampler, calculate_sample_weight, BalancedBatchSamplerPreSaved
from tensorboardX import SummaryWriter
import torch
import timm
from fish_rail_dataloader_track_based import Fish_Rail_Tracking_Result
from loss_funcs.classifier import Classifier
from loss_funcs.lmmd import LMMDLoss
from loss_funcs.dynamic_lmmd import DynamicLMMDLoss
import torch.nn as nn
import argparse


from prefetch_generator import BackgroundGenerator

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UDA training requires source data and target data. And a pre-trained model on source data.")
    parser.add_argument('-batch_size', '--batch_size', type=int, default=40,
                        help="batch_size is shared for target and source data.")
    parser.add_argument('-eval_batch_size', '--eval_batch_size', type=int, default=40,
                        help="batch_size for evaluation on target data if labels are provided.")
    parser.add_argument('-NUM_EPOCHS', '--NUM_EPOCHS', type=int, default=150,
                        help="total training epochs including pretrained epochs.")
    parser.add_argument('-multi_level', '--multi_level', type=str, default=False,
                        help='LMMD loss can be applied on multi-level features or only final level features.')
    parser.add_argument('-balance_sampler', '--balance_sampler', type=str, default=True,
                        help='make sure in each batch, source data has all classes, thus overwrite batch size with number of classes.')

    parser.add_argument('--evaluate_on_target', action='store_true',
                        help='if there are labels for target data, then we can evaluate different models during training.')
    parser.add_argument('-model_save_name', '--model_save_name', type=str,
                        default='-combined_target_data-aug_cutmix_autoaug-vessel6-546-250-lre-4-1level-balance-batch34-01weight-no_multi',
                        help="model specific parameters saved in name.")
    parser.add_argument('-tracking_result', '--tracking_result', type=str,
                        default='/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel6-546-250.csv',
                        help="target data obtained from tracking algorithm.")
    parser.add_argument('-uda_img_dir', '--uda_img_dir', type=str,
                        default='/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel6-546-250.csv',
                        help="target image data")

    args = parser.parse_args()

    ##########################
    ### SETTINGS
    ##########################

    # Hyperparameters
    RANDOM_SEED = 1
    # LEARNING_RATE = 0.000001
    # LEARNING_RATE = 0.001
    LEARNING_RATE = 0.0001
    NUM_EPOCHS =args.NUM_EPOCHS

    lmmd_loss_weight = 0.1

    # Architecture
    # BATCH_SIZE = 64 +32
    # BATCH_SIZE_val = 64 +32

    # resnet18
    # BATCH_SIZE = 256 +64
    # BATCH_SIZE_val = 512

    BATCH_SIZE = args.batch_size
    BATCH_SIZE_val = args.eval_batch_size
    BATCH_SIZE_UDA = args.batch_size
    multi_level = args.multi_level
    balance_sampler = args.balance_sampler

    img_size=224
    DEVICE = 'cuda:0'

    NUM_level_1_CLASSES,  NUM_level_2_CLASSES= calculate_num_class(hierarchy_dict)
    # if balance_sampler:
    #     BATCH_SIZE = NUM_level_2_CLASSES
    #     BATCH_SIZE_UDA = NUM_level_2_CLASSES

    # folder to save model
    # model_name = 'resnet18'
    model_name = 'resnext50_32x4d'
    model_save_path = './checkpoints/' +model_name +'_lmmd-' +args.model_save_name
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    writer = SummaryWriter('./logs/'+model_name +'_lmmd-'+args.model_save_name)
    save_path_val = './per img predictions val/'+model_name +'_lmmd-'+args.model_save_name
    save_path_tr = './per img predictions train/'+model_name +'_lmmd-'+args.model_save_name


    pretrain=True
    pretrain_epoch = 93
    pretained_model_save_path = './checkpoints/' + model_name + '_aug' + '_cutmix_autoaug'
    if not pretrain:
        pretrain_epoch=0
    else:
        assert pretrain_epoch!=None
        CHECKPOINT_PATH = os.path.join(pretained_model_save_path, 'parameters_epoch_'+str(pretrain_epoch)+'.pkl')

    # Note that transforms.ToTensor() already divides pixels by 255. internally
    custom_transform_train = transforms.Compose([transforms.Resize((img_size, img_size)),
                                           transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           # transforms.RandomRotation(degrees=15, expand=False, center=None, fill=None),
                                           transforms.RandomVerticalFlip(p=0.5),
                                           transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                                           # transforms.AutoAugment(),
                                           transforms.ToTensor()])

    custom_transform_val = transforms.Compose([transforms.Resize((img_size, img_size)),
                                           transforms.ToTensor()])

    valid_gt_path = './rail_cropped_data/labels_track_based/fish-rail-valid-plus_sleeper_shark_nonfish-level2_only.csv'
    train_gt_path = 'rail_cropped_data/labels_track_based/fish-rail-train-plus_sleeper_shark_nonfish-level2_only.csv'
    img_dir = './rail_cropped_data/cropped_box_with_sleeper_shark_non_fish'

    train_dataset = Fish_Rail_Dataset(csv_path=train_gt_path,
                                  img_dir= img_dir,
                                  transform=custom_transform_train,
                                  hierarchy = hierarchy_dict)


    valid_dataset = Fish_Rail_Dataset(csv_path=valid_gt_path,
                                  img_dir=img_dir,
                                  transform=custom_transform_val,
                                  hierarchy = hierarchy_dict)

    if balance_sampler:
        n_samples = 1
        # BATCH_SIZE_UDA = NUM_level_2_CLASSES * n_samples
        BATCH_SIZE_UDA = args.batch_size # Suzanne doesn't have enough GPU memory
        BATCH_SIZE = BATCH_SIZE_UDA
        # embed()
        # sampler = BalancedBatchSampler(train_dataset, NUM_level_2_CLASSES, n_samples=n_samples)
        sampler = BalancedBatchSamplerPreSaved(train_dataset, NUM_level_2_CLASSES, n_samples=n_samples)

        # sample_weights = calculate_sample_weight(n_classes=NUM_level_2_CLASSES, trainset=train_dataset)
        # sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(train_dataset), replacement=True)
        train_loader = DataLoaderX(dataset=train_dataset,
                                   batch_size=BATCH_SIZE,
                                   # collate_fn=collate_fn,
                                   shuffle=False,
                                   num_workers=0,
                                   drop_last=True,
                                   sampler=sampler)

    else:
        train_loader = DataLoaderX(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  # collate_fn=collate_fn,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

    # valid_loader = DataLoaderX(dataset=valid_dataset,
    #                           batch_size=BATCH_SIZE_val,
    #                           # collate_fn=collate_fn,
    #                           shuffle=False,
    #                           num_workers=0)

    # uda_img_dir = 'test_sleeper_shark/AK-50308-220423_214636-C1H-025-220524_210051_809_1-20230105T014047Z-001/AK-50308-220423_214636-C1H-025-220524_210051_809_1'
    # tracking_result = 'test_sleeper_shark/AK-50308-220423_214636-C1H-025-220524_210051_809_1-20230105T014047Z-001-result/AK-50308-220423_214636-C1H-025-220524_210051_809_1/tracking_result_with_huber.csv'

    #since combined_csv hsa full path
    uda_img_dir = args.uda_img_dir
    # tracking_result = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data.csv'
    # tracking_result = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_3gt.csv'
    # tracking_result = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_6gt.csv'

    # tracking_result = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel10-593.csv'
    # tracking_result = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel5-515.csv'
    # tracking_result = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_10gt.csv'
    # tracking_result = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel6-546-250.csv'
    tracking_result = args.tracking_result
    # tracking_result = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel9-136-072.csv'
    # tracking_result = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel15-761-819-AK5038.csv'
    # tracking_result = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel11-953.csv'
    # tracking_result = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel5-515.csv'
    # tracking_result = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel10-593.csv'




    target_dataset = Fish_Rail_Tracking_Result(csv_path=tracking_result,
                                            img_dir=uda_img_dir,
                                            transform=custom_transform_val,
                                            crop=True)


    target_loader = DataLoaderX(dataset=target_dataset,
                             batch_size=BATCH_SIZE_UDA,
                             shuffle=True,
                             num_workers=0,
                             drop_last=True)

    target_eval_dataset = Fish_Rail_Tracking_Result(csv_path=tracking_result,
                                            img_dir=uda_img_dir,
                                            transform=custom_transform_val,
                                            crop=True,
                                            return_label=True)

    valid_loader = DataLoaderX(dataset=target_eval_dataset,
                              batch_size=BATCH_SIZE_val,
                              # collate_fn=collate_fn,
                              shuffle=False,
                              num_workers=0)

    torch.manual_seed(RANDOM_SEED)

    ##########################
    ### COST AND OPTIMIZER
    ##########################

    # model = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=NUM_level_2_CLASSES)
    if multi_level:
        model = timm.create_model(model_name, pretrained=True, features_only=True)  # return multi-level futures
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=0) # return pooled futures

    clf = Classifier(num_class=NUM_level_2_CLASSES, feature_dim=2048)

    #### DATA PARALLEL START ####
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs")
    #     model = nn.DataParallel(model)
    #### DATA PARALLEL END ####
    model.to(DEVICE)
    clf.to(DEVICE)

    if pretrain:
        model.load_state_dict(torch.load(CHECKPOINT_PATH), strict=False)
        clf.load_state_dict(torch.load(CHECKPOINT_PATH), strict=False)
        print('loaded pretrained model: ', CHECKPOINT_PATH)
        NUM_EPOCHS = NUM_EPOCHS-pretrain_epoch  #50-29=21




    #### start training ###
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001, amsgrad=False)
    optimizer = torch.optim.AdamW(list(model.parameters())+list(clf.parameters()), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)  #1个epoch 减小lr

    loss = torch.nn.CrossEntropyLoss()
    transfer_loss = LMMDLoss(num_class=NUM_level_2_CLASSES)
    # transfer_loss = DynamicLMMDLoss(num_class=NUM_level_2_CLASSES)

    # dummy_input = torch.rand(10, 3, img_size, img_size).to(DEVICE)
    # writer.add_graph(model, (dummy_input,))


    # torch.backends.cudnn.benchmark = True  #在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销。我一般都会加
    start_time = time.time()
    best_acc_2_p1p2_val_31_img_based = []
    best_acc_2_p1p2_val_maxmax_img_based = []

    # EQL_loss = SoftmaxEQL(lambda_1=20000, lambda_2=5000, ignore_prob=0.5, file_name='./labels_track_based/fish-rail-train.csv')

    #EQL_loss = SoftmaxEQL(lambda_1=10000, lambda_2=1000, ignore_prob=0.8, file_name='Z:/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/labels_track_based/fish-rail-train-plus_sleeper_shark_nonfish.csv')

    pooling = nn.AdaptiveAvgPool2d(1)

    for epoch in range(NUM_EPOCHS):  # 0-20

        #training
        model.train()
        clf.train()

        # start_time = time.time()
        train_loader_iterator = iter(train_loader)
        for batch_idx, item2 in enumerate(target_loader):
            try:
                item1 = next(train_loader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_loader)
                item1 = next(dataloader_iterator)


        # for batch_idx, (item1, item2) in enumerate(zip(train_loader, cycle(target_loader))):
        #     torch.cuda.empty_cache()  # 个命令是清除没用的临时变量的。
            # print('Time elapsed: %.3f s' % ((time.time() - start_time)))
            # start_time = time.time()

            # if batch_idx>=len(target_loader):
            #     iter_target = iter(target_loader)
                # break


            # target_imgs, target_img_names, target_track_ids = next(iter_target)
            # target_imgs = target_imgs.to(DEVICE)
            # step-1: get data from training dataset & target dataset


            imgs, targets, targets_split, img_name, id = item1
            target_imgs, target_img_names, target_track_ids = item2

            # targets_split = targets_split.to(DEVICE)
            # img_name =  torch.tensor(img_name).to(DEVICE)
            # id = id.to(DEVICE)
            # target_img_names = target_img_names.to(DEVICE)
            # target_track_ids = target_track_ids.to(DEVICE)




            target_imgs = target_imgs.to(DEVICE)
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            if batch_idx <5:
                print(targets[:,1])

            optimizer.zero_grad()

            # Step-2: run through the model together, get features
            features = model(torch.cat((imgs, target_imgs)))
            # features = model(imgs)

            # Step-3: run through the classifier, get the logits
            if multi_level:
                # use final level features for classification
                pred = clf(pooling(features[-1]).squeeze())
            else:
                pred = clf(features)

            # Step-4: Seperate features & logits
            source_logits = pred[0:BATCH_SIZE]
            target_logits = pred[BATCH_SIZE:]

            # Step-5: classification loss
            clf_loss = loss(source_logits, targets[:,1].long())

            # Step-6: lmmd loss. get pseudo labels
            target_probas = torch.nn.functional.softmax(target_logits, dim=1)

            if multi_level:
                lmmd_loss = 0
                for fea_level,fea in enumerate(features):
                    # Seperate source and target features
                    # embed()
                    if fea_level <3:
                        continue
                    source_features = fea[0:BATCH_SIZE]
                    target_features = fea[BATCH_SIZE:]
                    lmmd_loss += transfer_loss(source=pooling(source_features).squeeze(), target=pooling(target_features).squeeze(), source_label=targets[:, 1], target_logits=target_probas)

            else:
                source_features = features[0:BATCH_SIZE]
                target_features = features[BATCH_SIZE:]
                lmmd_loss = transfer_loss(source=source_features, target=target_features, source_label=targets[:,1], target_logits = target_probas)

            cost = clf_loss + lmmd_loss_weight * lmmd_loss
            # cost = clf_loss

            assert targets!=None, embed()



            cost.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            if batch_idx%5==0:
                writer.add_scalars('scalar/loss',
                                   {'total loss': cost.item(), 'clf loss': clf_loss.detach().item(), 'lmmd loss': lmmd_loss_weight * lmmd_loss.detach().item()},
                                   (epoch + pretrain_epoch + 1) * len(target_loader) + batch_idx)
                writer.flush()


            ## LOGGING
            if not batch_idx % 2:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f | clf loss: %.4f | lmmd loss: %.4f ' % (
                    epoch + 1 + pretrain_epoch, NUM_EPOCHS + pretrain_epoch, batch_idx, len(target_loader), cost.detach().item(), clf_loss.detach().item(), lmmd_loss_weight *lmmd_loss.detach().item()))  #0+1+29=30

            # del cost, features, pred, clf_loss, lmmd_loss, targets_split, img_name, id, target_img_names, target_track_ids, target_imgs, imgs, targets


        scheduler.step()

        if (epoch+pretrain_epoch) > 5 and args.evaluate_on_target:
        # if True:
            torch.cuda.empty_cache()
            model.eval()
            clf.eval()
            ### for model 7
            with torch.set_grad_enabled(False):  # save memory during inference

                avg_level_2_acc_p1p2_31_val, acc_2_p1p2_31_val = compute_accuracy_model7_track_based_level_2_only(
                    [model, clf], valid_loader, epoch,DEVICE, save_path_val, lmmd=True, multi_level=multi_level)

                ##根据记录下来的confidence，计算tarck-based的accuracy
                avg_level_2_acc_p1p2_31_val_track, acc_2_p1p2_31_val_track=\
                    track_based_accuracy_level2_only(save_path_val, epoch)


                print(
                    'Track-based Epoch: %03d/%03d | Valid: Level-2 Avg p1p2 max out of 31: %.3f%%' % (
                        epoch + 1 +pretrain_epoch, NUM_EPOCHS+pretrain_epoch,
                        avg_level_2_acc_p1p2_31_val_track * 100,

                    ))

                print('Track-based Individual accuracy: Valid: '
                      'Level-2 p1p2 max out of 31:', acc_2_p1p2_31_val_track)

                print('Image-based Epoch: %03d/%03d | Valid: Level-2 Avg p1p2 max out of 31: %.3f%%' % (
                        epoch+1+pretrain_epoch,NUM_EPOCHS+pretrain_epoch,
                        avg_level_2_acc_p1p2_31_val * 100
                    ))

                print('Image-based Individual accuracy: Valid: '
                      'Level-2 p1p2 max out of 31:', acc_2_p1p2_31_val)



            best_acc_2_p1p2_val_31_img_based.append(avg_level_2_acc_p1p2_31_val)



            writer.add_scalars('scalar/img-based val avg accuracy', {
                                                           'level-1&2 max out of 31': avg_level_2_acc_p1p2_31_val},
                               epoch +1+pretrain_epoch)
            writer.add_scalars('scalar/img-based val individual level-1&2 max out of 31 accuracy', acc_2_p1p2_31_val,
                               epoch+1+pretrain_epoch)

            writer.add_scalars('scalar/track-based val avg accuracy',
                               {
                                'level-1&2 max out of 31': avg_level_2_acc_p1p2_31_val_track,},
                               epoch+1+pretrain_epoch)
            writer.add_scalars('scalar/track-based val individual level-1&2 max out of 31 accuracy', acc_2_p1p2_31_val_track,
                               epoch+1+pretrain_epoch)



            writer.flush()




        torch.cuda.empty_cache()  #个命令是清除没用的临时变量的。


        torch.save(model.state_dict(), os.path.join(model_save_path, 'parameters_epoch_' + str(epoch+1+pretrain_epoch)  + '.pkl'))
        torch.save(clf.state_dict(),
                   os.path.join(model_save_path, 'clf_parameters_epoch_' + str(epoch + 1 + pretrain_epoch) + '.pkl'))




        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    writer.close()


    # embed()


