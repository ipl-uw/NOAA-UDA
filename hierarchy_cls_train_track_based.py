import time

from torch.utils.data import DataLoader
from torchvision import transforms
import os
from util import compute_accuracy_model12, compute_accuracy_model0,calculate_num_class,\
    calculate_num_class_for_each_head, hierarchy_dict, find_level2_head_loss_for_model12, calculate_num_class_model0, \
    draw_loss,show_img, compute_accuracy_model7_track_based, find_level2_head_loss_for_model7, track_based_accuracy, SoftmaxEQL, collate_fn, compute_accuracy_model7_track_based_level_2_only, track_based_accuracy_level2_only

import torch.nn.functional as F
from IPython import embed
from fish_rail_dataloader_track_based import Fish_Rail_Dataset
from tensorboardX import SummaryWriter
import torch
from vit_pytorch import ViT


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 250


# Architecture
BATCH_SIZE = 512
BATCH_SIZE_val = 1024 *3
img_size=224
DEVICE = 'cuda:0' # default GPU device

NUM_level_1_CLASSES,  NUM_level_2_CLASSES= calculate_num_class(hierarchy_dict)  # model1 model2   37

# folder to save model
model_save_path = './checkpoints_plus_sleeper_shark_nonfish_vit_level2'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
writer = SummaryWriter('./logs-plus_sleeper_shark_nonfish_vit_level2')
save_path_val = './per img predictions val plus_sleeper_shark_nonfish_vit_level2'
save_path_tr = './per img predictions tr plus_sleeper_shark_nonfish_vit_level2'


pretrain=False
if not pretrain:
    pretrain_epoch=0
else:
    pretrain_epoch = 15
    CHECKPOINT_PATH = os.path.join(model_save_path, 'parameters_epoch_'+str(pretrain_epoch)+'.pkl')

# Note that transforms.ToTensor() already divides pixels by 255. internally
custom_transform_train = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomRotation(degrees=15, expand=False, center=None, fill=None),
                                       transforms.RandomVerticalFlip(p=0.5),
                                       transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                                       transforms.ToTensor()])

custom_transform_val = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor()])

valid_gt_path = '/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=Jie%20Mei,volume=homes/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/labels_track_based/fish-rail-valid-plus_sleeper_shark_nonfish.csv'
train_gt_path = '/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=Jie%20Mei,volume=homes/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/labels_track_based/fish-rail-train-plus_sleeper_shark_nonfish.csv'
img_dir = '/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=Jie%20Mei,volume=homes/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/cropped_box_with_sleeper_shark_non_fish'

train_dataset = Fish_Rail_Dataset(csv_path=train_gt_path,
                              img_dir= img_dir,
                              transform=custom_transform_train,
                              hierarchy = hierarchy_dict)


valid_dataset = Fish_Rail_Dataset(csv_path=valid_gt_path,
                              img_dir=img_dir,
                              transform=custom_transform_val,
                              hierarchy = hierarchy_dict)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          # collate_fn=collate_fn,
                          shuffle=True,
                          num_workers=0)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE_val,
                          # collate_fn=collate_fn,
                          shuffle=False,
                          num_workers=0)


torch.manual_seed(RANDOM_SEED)

##########################
### COST AND OPTIMIZER
##########################

model = ViT(
    image_size = img_size,
    patch_size = 32,
    num_classes = NUM_level_2_CLASSES,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

#### DATA PARALLEL START ####
# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs")
#     model = nn.DataParallel(model)
#### DATA PARALLEL END ####
model.to(DEVICE)

if pretrain:
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print('loaded pretrained model: ', CHECKPOINT_PATH)
    NUM_EPOCHS = NUM_EPOCHS-pretrain_epoch  #50-29=21




#### start training ###
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001, amsgrad=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)  #1个epoch 减小lr

loss = torch.nn.CrossEntropyLoss()


# dummy_input = torch.rand(10, 3, img_size, img_size).to(DEVICE)
# writer.add_graph(model, (dummy_input,))


torch.backends.cudnn.benchmark = True  #在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销。我一般都会加
start_time = time.time()
best_acc_2_p1p2_val_31_img_based = []
best_acc_2_p1p2_val_maxmax_img_based = []

# EQL_loss = SoftmaxEQL(lambda_1=20000, lambda_2=5000, ignore_prob=0.5, file_name='./labels_track_based/fish-rail-train.csv')

#EQL_loss = SoftmaxEQL(lambda_1=10000, lambda_2=1000, ignore_prob=0.8, file_name='Z:/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/labels_track_based/fish-rail-train-plus_sleeper_shark_nonfish.csv')

for epoch in range(NUM_EPOCHS):  # 0-20

    #training
    model.train()

    for batch_idx, (imgs, targets, label_split, _,_) in enumerate(train_loader):
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)
        label_split = label_split.to(DEVICE)

        # show_img(features)
        ### FORWARD AND BACK PROP  for hierarchy for model12
        # logits_list, probas_list, probas_level2 = model(imgs)   # for model 67

        probas_level2 = model(imgs)  # for model 67


        ## Equalization loss
        #cost_level_1, cost_level_2 = EQL_loss(logits_list, targets)



        ### 计算第一个head的loss
        # level_1_target = targets[:, 0]
        # level_1_logits = logits_list[0]
        # cost_level_1 = F.cross_entropy(level_1_logits, level_1_target)

        # # cost_level_2 = find_level2_head_loss_for_model12(label_split, logits_list)  #6 个head其中一个的loss
        # cost_level_2 = find_level2_head_loss_for_model7(probas_level2, targets)
        level_2_targets = targets[:, 1]
        idx = torch.where(level_2_targets != -1)
        cost_level_2 = loss(probas_level2[idx], level_2_targets[idx])


        # 确保 head 1得到足够的Loss取更新参数
        # lambda_1 = max(np.floor(cost_level_2.item()/cost_level_1.item()),1)
        # print(lambda_1)
        # cost_level_1 = LEVEL_1_coef *  cost_level_1
        # cost = cost_level_1+cost_level_2
        cost = cost_level_2



        ### FORWARD AND BACK PROP  for model-0s
        # logits, probas = model(features)
        # cost = F.cross_entropy(logits, targets)
        assert targets!=None, embed()


        ### tensorboard

        # writer.add_scalars('scalar/loss',
        #                 {'total loss': cost, 'level-1 loss': cost_level_1, 'level-2 loss': cost_level_2}, (epoch + pretrain_epoch+1) * len(train_loader) + batch_idx)

        writer.add_scalars('scalar/loss',
                           {'total loss': cost, 'level-2 loss': cost_level_2},
                           (epoch + pretrain_epoch + 1) * len(train_loader) + batch_idx)

        writer.flush()


        optimizer.zero_grad()

        cost.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        # if not batch_idx % 50:
        #     print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f | cost level-1: %.4f | cost level-2: %.4f' % (
        #         epoch + 1 + pretrain_epoch, NUM_EPOCHS + pretrain_epoch, batch_idx, len(train_loader), cost, cost_level_1, cost_level_2))  #0+1+29=30
        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f | cost level-2: %.4f' % (
                epoch + 1 + pretrain_epoch, NUM_EPOCHS + pretrain_epoch, batch_idx, len(train_loader), cost, cost_level_2))  #0+1+29=30


    scheduler.step()

    if (epoch+pretrain_epoch) > 5 and (epoch+pretrain_epoch) %2==0:
    # if (epoch + pretrain_epoch) >= 0:
        model.eval()
        ### for model 7
        with torch.set_grad_enabled(False):  # save memory during inference

            avg_level_2_acc_p1p2_31_val, acc_2_p1p2_31_val = compute_accuracy_model7_track_based_level_2_only(
                model, valid_loader, epoch,DEVICE, save_path_val)

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
                           epoch +pretrain_epoch)
        writer.add_scalars('scalar/img-based val individual level-1&2 max out of 31 accuracy', acc_2_p1p2_31_val,
                           epoch+pretrain_epoch)

        writer.add_scalars('scalar/track-based val avg accuracy',
                           {
                            'level-1&2 max out of 31': avg_level_2_acc_p1p2_31_val_track,},
                           epoch+pretrain_epoch)
        writer.add_scalars('scalar/track-based val individual level-1&2 max out of 31 accuracy', acc_2_p1p2_31_val_track,
                           epoch+pretrain_epoch)



        writer.flush()




    torch.cuda.empty_cache()  #个命令是清除没用的临时变量的。


    torch.save(model.state_dict(), os.path.join(model_save_path, 'parameters_epoch_' + str(epoch+1+pretrain_epoch)  + '.pkl'))




    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
writer.close()


embed()


