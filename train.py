import time

from torch.utils.data import DataLoader
from torchvision import transforms
import os
from util import calculate_num_class, hierarchy_dict, compute_accuracy_model7_track_based_level_2_only, track_based_accuracy_level2_only, cutmix_data

from IPython import embed
from fish_rail_dataloader_track_based import Fish_Rail_Dataset
from tensorboardX import SummaryWriter
import torch
import timm
import numpy as np

from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100

# Architecture
# BATCH_SIZE = 64 +32
# BATCH_SIZE_val = 64 +32

# resnet18
# BATCH_SIZE = 256 +64
# BATCH_SIZE_val = 512

BATCH_SIZE = 64 +32
BATCH_SIZE_val = 256

CUT_MIX = True

img_size=224
DEVICE = 'cuda:0'

NUM_level_1_CLASSES,  NUM_level_2_CLASSES= calculate_num_class(hierarchy_dict)

# folder to save model
# model_name = 'resnet18'
model_name = 'resnext50_32x4d'
model_save_path = './checkpoints/' +model_name +'_aug' +'_cutmix' +'_autoaug'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
writer = SummaryWriter('./logs/'+model_name+'_aug' +'_cutmix'+'_autoaug')
save_path_val = './per img predictions val/'+model_name +'_aug' +'_cutmix'+'_autoaug'
save_path_tr = './per img predictions train/'+model_name +'_aug' +'_cutmix'+'_autoaug'


pretrain=False
pretrain_epoch = 10
if not pretrain:
    pretrain_epoch=0
else:
    assert pretrain_epoch!=None
    CHECKPOINT_PATH = os.path.join(model_save_path, 'parameters_epoch_'+str(pretrain_epoch)+'.pkl')

# Note that transforms.ToTensor() already divides pixels by 255. internally
custom_transform_train = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       # transforms.RandomRotation(degrees=15, expand=False, center=None, fill=None),
                                       transforms.RandomVerticalFlip(p=0.5),
                                       transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                                       transforms.AutoAugment(),
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

train_loader = DataLoaderX(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          # collate_fn=collate_fn,
                          shuffle=True,
                          num_workers=4)

valid_loader = DataLoaderX(dataset=valid_dataset,
                          batch_size=BATCH_SIZE_val,
                          # collate_fn=collate_fn,
                          shuffle=False,
                          num_workers=4)


torch.manual_seed(RANDOM_SEED)

##########################
### COST AND OPTIMIZER
##########################

# model = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=NUM_level_2_CLASSES)
model = timm.create_model(model_name, pretrained=True, num_classes=NUM_level_2_CLASSES)

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


# torch.backends.cudnn.benchmark = True  #在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销。我一般都会加
start_time = time.time()
best_acc_2_p1p2_val_31_img_based = []
best_acc_2_p1p2_val_maxmax_img_based = []

# EQL_loss = SoftmaxEQL(lambda_1=20000, lambda_2=5000, ignore_prob=0.5, file_name='./labels_track_based/fish-rail-train.csv')

#EQL_loss = SoftmaxEQL(lambda_1=10000, lambda_2=1000, ignore_prob=0.8, file_name='Z:/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/labels_track_based/fish-rail-train-plus_sleeper_shark_nonfish.csv')

for epoch in range(NUM_EPOCHS):  # 0-20

    #training
    model.train()

    # start_time = time.time()
    for batch_idx, (imgs, targets, targets_split, img_name, id) in enumerate(train_loader):
        # print('Time elapsed: %.3f s' % ((time.time() - start_time)))
        # start_time = time.time()



        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        if CUT_MIX and np.random.rand(1) < 0.5:
            x, labels_a, labels_b, lam = cutmix_data(x=imgs, y=targets, alpha=1.0)
            logit = model(x)
            cost = lam * loss(logit, labels_a[:,1]) + (1 - lam) * loss(
                logit, labels_b[:,1]
            )

        else:
            pred = model(imgs)
            cost = loss(pred, targets[:,1])

        assert targets!=None, embed()

        if batch_idx%50==0:
            writer.add_scalars('scalar/loss',
                               {'total loss': cost},
                               (epoch + pretrain_epoch + 1) * len(train_loader) + batch_idx)
            writer.flush()

        cost.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()


        ### LOGGING
        if not batch_idx % 2:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' % (
                epoch + 1 + pretrain_epoch, NUM_EPOCHS + pretrain_epoch, batch_idx, len(train_loader), cost))  #0+1+29=30



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


