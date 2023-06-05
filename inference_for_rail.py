from fish_rail_dataloader_track_based import Fish_Rail_Tracking_Result
from fish_rail_dataloader_track_based import Fish_Rail_Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from Model_7 import resnet101
import os, torch
from util import calculate_num_class,hierarchy_dict, level_1_names, level_2_names
from tqdm import tqdm
import argparse
import timm
import numpy as np

from IPython import embed

from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hierarchical classification")

    # parser.add_argument('-img_dir', '--img_dir', type=str,
    #                     default='Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\Vessel6-190904_181759-C1H-011-190907_025159_354_selected',
    #                     help="folder to frames folder")
    # parser.add_argument('-tracking_result', '--tracking_result', type=str,
    #                     default='Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark/tracking_result_with_huber.csv',
    #                     help="folder to tracking_result_with_huber_processed.csv")

    parser.add_argument('-img_dir', '--img_dir', type=str,
                        default='/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/unlabeled_target_data/AK-50308-220423_214636-C1H-025-220524_210051_809_1',
                        help="folder to frames folder")
    parser.add_argument('-tracking_result', '--tracking_result', type=str,
                        default='/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/AK-50308-220423_214636-C1H-025-220524_210051_809_1/tracking_result_with_huber.csv',
                        help="folder to tracking_result_with_huber_processed.csv")

    args = parser.parse_args()

    # Set frames path, tracking result csv path, batch_size
    img_dir = args.img_dir
    tracking_result = args.tracking_result



    BATCH_SIZE = 256 *3
    img_size = 224


    # Read tracking result
    custom_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                           transforms.ToTensor()])

    dataset = Fish_Rail_Tracking_Result(csv_path=tracking_result,
                                        img_dir=img_dir,
                                        transform=custom_transform,
                                        crop=True,
                                        species_column='species')


    data_loader = DataLoaderX(dataset=dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4)
    # Load Model
    GRAYSCALE = False
    # NUM_CLASSES = calculate_num_class(hierarchy_dict)
    NUM_level_1_CLASSES, NUM_level_2_CLASSES = calculate_num_class(hierarchy_dict)
    # model_save_path = './checkpoints-model7-track_based more'
    # best_epoch=135   #model-7  more

    model_name = 'resnext50_32x4d'
    # no lmmd model
    # model_save_path = './checkpoints/' + model_name +'_aug'+'_cutmix_autoaug'
    # prediction_result = tracking_result.replace(tracking_result.split('/')[-1],
    #                                             'flat_classification_result_aug_cutmix_autoaug_on_SR_gt.csv')
    # best_epoch = 93  # with aug + cutmis + autoaug

    # 28% finetuned model
    # model_save_path = './checkpoints/' + model_name +'finetune_5%'
    # best_epoch = 95  # with aug + cutmis + autoaug
    # prediction_result = tracking_result.replace(tracking_result.split('/')[-1],
    #                                             'flat_classification_result_finetune_5%.csv')

    # 3% finetuned model
    model_save_path = './checkpoints/' + model_name + 'finetune_1%'
    best_epoch = 100  # with aug + cutmis + autoaug
    prediction_result = tracking_result.replace(tracking_result.split('/')[-1],
                                                'flat_classification_result_finetune_1%.csv')


    device = 'cuda:0'
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_level_2_CLASSES)
    PATH = os.path.join(model_save_path,'parameters_epoch_'+str(best_epoch)+'.pkl')
    model.load_state_dict(torch.load(PATH))
    model.to(device)

    # evaluate
    model.eval()
    id_group= {}
    id_species = {}
    id_group_score= {}
    id_species_score = {}

    id_species_score_top2 = {}

    accumulate_group = torch.zeros(())
    with torch.set_grad_enabled(False):
        print('Running hierachical classification on %s...' %tracking_result)

        #accumulate confidence score (level-1 and lvel-2) of each frame!
        for i, (imgs, img_names, track_ids) in tqdm(enumerate(data_loader)):
            imgs = imgs.to(device)
            # _, probas, probas_level2 = model(imgs)  ### model 7
            pred = model(imgs)  ### model 7

            m = torch.nn.Softmax(dim=1)
            pred = m(pred)

            # probas_level_1 = probas[0]

            for id in set(track_ids.numpy()):
                if id not in id_group:
                    # id_group[id] = 0
                    id_species[id] = 0
                index = torch.where(track_ids==id)
                # id_group[id] += torch.sum(probas_level_1[index], dim=0)
                id_species[id] +=torch.sum(pred[index], dim=0)



        # pick max score as prediction for level-1 and level-2
        for id in id_species:
            # group_scores = id_group[id]
            # id_group_score[id] = group_scores[torch.argmax(group_scores).item()].item()
            # id_group[id] = torch.argmax(group_scores).item()
            species_scores = id_species[id]
            top_1_idx = torch.argmax(species_scores).item()
            id_species_score[id] = species_scores[top_1_idx].item()
            id_species[id] = top_1_idx

            top_1_name = level_2_names[top_1_idx]
            top_2_idx = torch.argsort(species_scores)[-2]
            top_2_name = level_2_names[top_2_idx]

            if id not in id_species_score_top2:
                id_species_score_top2[id] = {}

            id_species_score_top2[id][top_1_name] = np.around(species_scores[top_1_idx].item(), 2)
            id_species_score_top2[id][top_2_name] = np.around(species_scores[top_2_idx].item(),2)



        print('Running Flat classification on %s...Done!' % tracking_result)


    # top-2 predictions
    # for id in id_species_score_top2:
    #     print('track_id:',id, id_species_score_top2[id])
    # embed()

    #read tracking csv and add group, species, and 2 total scores
    import csv
    from collections import OrderedDict
    def read_order_as_dict(csv_file):
        csvFile_all = open(csv_file, 'r')
        dict_reader_all = csv.DictReader(csvFile_all)

        track_target = OrderedDict()
        for i, row in enumerate(dict_reader_all):
            track_id = row['id']
            if track_id not in track_target:
                track_target[track_id] = []
            track_target[track_id].append(row)
        csvFile_all.close()

        return track_target

    print('Reading tracking csv and Writing csv: %s ...' %prediction_result)
    track_target = read_order_as_dict(tracking_result)

    # calculate average confidence score and write into csv
    def write_dict_to_csv(track_dict, file_name):
        save_file = open(file_name, "w", newline='')
        fieldnames = ['id', 'filename', 'xmin', 'ymin', 'xmax', 'ymax', 'group', 'group_conf','species', 'species_conf','length','kept' ]
        writer = csv.DictWriter(save_file, fieldnames=fieldnames)
        writer.writeheader()

        for track_id in tqdm(track_dict):
            each_track = track_dict[track_id]
            leng = len(each_track)
            track_id = int(track_id)
            for info in each_track:
                new_info = info.copy()

                for key in info:  #去掉conf class
                    if key not in fieldnames:
                        new_info.pop(key)

                # new_info['group']= level_1_names[id_group[track_id]]
                new_info['group'] = 'NA'
                new_info['species'] = level_2_names[id_species[track_id]]
                # new_info['group_conf'] = id_group_score[track_id]/leng
                new_info['group_conf'] = '1'
                new_info['species_conf'] = id_species_score[track_id]/leng


                # embed()
                writer.writerow(new_info)
        save_file.close()

    write_dict_to_csv(track_target, prediction_result)

    print('Reading tracking csv and Writing csv: %s ... Done!' %prediction_result)