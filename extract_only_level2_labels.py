import csv
from collections import OrderedDict
from tqdm import tqdm
from IPython import embed
from util import level_2_names

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




# calculate average confidence score and write into csv
def filter_out_level2_only(track_dict, file_name):
    save_file = open(file_name, "w", newline='')
    fieldnames = ['id', 'filename', 'xmin', 'ymin', 'xmax', 'ymax', 'conf','class','length','kept' ]
    writer = csv.DictWriter(save_file, fieldnames=fieldnames)
    writer.writeheader()

    for track_id in tqdm(track_dict):
        each_track = track_dict[track_id]

        # filter out level-1 names
        if each_track[0]['class'] not in level_2_names:
            continue
        for info in each_track:
            new_info = info.copy()

            writer.writerow(new_info)
    save_file.close()


def data_distribution(track_dict):
    distribution_track = {}
    distribution_img = {}
    for track_id in track_dict:
        track_info = track_dict[track_id]
        species = track_info[0]['class']

        if species in distribution_track:
            distribution_track[species] += 1
        else:
            distribution_track[species] = 1

        if species in distribution_img:
            distribution_img[species] += len(track_info)
        else:
            distribution_img[species] = len(track_info)

    return distribution_track, distribution_img

valid_gt_path = '/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=JIe%20Mei,volume=homes/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/labels_track_based/fish-rail-valid-plus_sleeper_shark_nonfish.csv'
train_gt_path = '/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=JIe%20Mei,volume=homes/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/labels_track_based/fish-rail-train-plus_sleeper_shark_nonfish.csv'
train_dict = read_order_as_dict(train_gt_path)
valid_dict = read_order_as_dict(valid_gt_path)


level2_train_path = '/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=JIe%20Mei,volume=homes/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/labels_track_based/fish-rail-train-plus_sleeper_shark_nonfish-level2_only.csv'
level2_valid_path = '/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=JIe%20Mei,volume=homes/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/labels_track_based/fish-rail-valid-plus_sleeper_shark_nonfish-level2_only.csv'
filter_out_level2_only(train_dict, level2_train_path)
filter_out_level2_only(valid_dict, level2_valid_path)


train_dict_level2 = read_order_as_dict(level2_train_path)
valid_dict_levle2 = read_order_as_dict(level2_valid_path)


track_distribution_train, img_distribution_train = data_distribution(train_dict_level2)
track_distribution_val, img_distribution_val = data_distribution(valid_dict_levle2)


embed()

