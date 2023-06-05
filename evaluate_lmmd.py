# from Model_0 import resnet101

from Model_7 import resnet101
# from hierarchy_cls_train import model_save_path,train_loader,valid_loader,DEVICE,NUM_CLASSES, GRAYSCALE
import torch
import torch.nn as nn
import os
from util import compute_accuracy_model0,calculate_num_class,hierarchy_dict, \
    calculate_num_class_model0, compute_accuracy_model12, \
    compute_accuracy_model7_track_based, track_based_accuracy,\
    track_based_accuracy_majority_vote,Otsu_Threshold, compute_accuracy_model7_track_based_level_2_only, track_based_accuracy_level2_only
from IPython import embed
from torchvision import transforms
from fish_rail_dataloader_track_based import Fish_Rail_Dataset, Fish_Rail_Tracking_Result

from torch.utils.data import DataLoader
import timm

from prefetch_generator import BackgroundGenerator
from loss_funcs.classifier import Classifier
from loss_funcs.lmmd import LMMDLoss


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

GRAYSCALE = False
# NUM_CLASSES = calculate_num_class(hierarchy_dict)  #37
# NUM_CLASSES = calculate_num_class_model0(hierarchy_dict)  # model0   31
NUM_level_1_CLASSES,  NUM_level_2_CLASSES= calculate_num_class(hierarchy_dict)

model_name = 'resnext50_32x4d'
model_save_path = './checkpoints/' +model_name+'_lmmd'
DEVICE =  'cuda:0'
BATCH_SIZE=256 *3
img_size=224

# model-7
save_path_val = './per img predictions val/'+model_name+'_lmmd'
save_path_tr = './per img predictions train/'+model_name+'_lmmd'


custom_transform = transforms.Compose([transforms.Resize((img_size, img_size)),

                                       transforms.ToTensor()])


valid_gt_path = './rail_cropped_data/labels_track_based/fish-rail-valid-plus_sleeper_shark_nonfish-level2_only.csv'
train_gt_path = 'rail_cropped_data/labels_track_based/fish-rail-train-plus_sleeper_shark_nonfish-level2_only.csv'
img_dir = './rail_cropped_data/cropped_box_with_sleeper_shark_non_fish'

train_dataset = Fish_Rail_Dataset(csv_path=train_gt_path,
                              img_dir=img_dir,
                              transform=custom_transform,
                              hierarchy = hierarchy_dict)


valid_dataset = Fish_Rail_Dataset(csv_path=valid_gt_path,
                              img_dir=img_dir,
                              transform=custom_transform,
                              hierarchy = hierarchy_dict)


train_loader = DataLoaderX(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=0)

valid_loader = DataLoaderX(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=0)

uda_img_dir = 'test_sleeper_shark/AK-50308-220423_214636-C1H-025-220524_210051_809_1-20230105T014047Z-001/AK-50308-220423_214636-C1H-025-220524_210051_809_1'
tracking_result = 'test_sleeper_shark/AK-50308-220423_214636-C1H-025-220524_210051_809_1-20230105T014047Z-001-result/AK-50308-220423_214636-C1H-025-220524_210051_809_1/tracking_result_with_huber.csv'
target_dataset = Fish_Rail_Tracking_Result(csv_path=tracking_result,
                                        img_dir=uda_img_dir,
                                        transform=custom_transform,
                                        crop=True)

### load model
# best_epoch=23     # nonfish + sleeper shark
best_epoch=11     # nonfish + sleeper shark
best_epoch=6     # nonfish + sleeper shark
stop_at_level_1_threshold=0.85
model = timm.create_model(model_name, pretrained=True, num_classes=0)
PATH = os.path.join(model_save_path,'parameters_epoch_'+str(best_epoch)+'.pkl')
model.load_state_dict(torch.load(PATH))
model.to(DEVICE)

clf = Classifier(num_class=NUM_level_2_CLASSES, feature_dim=2048)
PATH = os.path.join(model_save_path,'clf_parameters_epoch_'+str(best_epoch)+'.pkl')
clf.load_state_dict(torch.load(PATH))
clf.to(DEVICE)


### 最后测试一下  for model7
model.eval()

# for model7
with torch.set_grad_enabled(False):  # save memory during inference
    avg_level_2_acc_p1p2_31_val, acc_2_p1p2_31_val = compute_accuracy_model7_track_based_level_2_only(
        [model,clf], valid_loader, best_epoch, DEVICE, save_path_val, lmmd=True)

    ##根据记录下来的confidence，计算tarck-based的accuracy
    avg_level_2_acc_p1p2_31_val_track, acc_2_p1p2_31_val_track = \
        track_based_accuracy_level2_only(save_path_val, best_epoch)

    print(
        'Track-based Epoch: %03d | Valid: Level-2 Avg p1p2 max out of 31: %.3f%%' % (
            best_epoch,
            avg_level_2_acc_p1p2_31_val_track * 100,

        ))

    print('Track-based Individual accuracy: Valid: '
          'Level-2 p1p2 max out of 31:', acc_2_p1p2_31_val_track)

    print('Image-based Epoch: %03d | Valid: Level-2 Avg p1p2 max out of 31: %.3f%%' % (
        best_epoch,
        avg_level_2_acc_p1p2_31_val * 100
    ))

    print('Image-based Individual accuracy: Valid: '
          'Level-2 p1p2 max out of 31:', acc_2_p1p2_31_val)








embed()