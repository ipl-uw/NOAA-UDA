import os
from util import read_csv_as_dict, read_cls_csv_as_dict
from IPython import embed
import csv
from tqdm import tqdm
import numpy as np
import random

# calculate average confidence score and write into csv
def write_dict_to_csv(track_dict, file_name):
    save_file = open(file_name, "w", newline='')
    fieldnames = ['id', 'filename', 'xmin', 'ymin', 'xmax', 'ymax', 'conf','class','length','kept' ]
    writer = csv.DictWriter(save_file, fieldnames=fieldnames)
    writer.writeheader()
    total_unlabeled_frames = 0
    for track_id in tqdm(track_dict):
        each_track = track_dict[track_id]
        leng = len(each_track)
        total_unlabeled_frames+= leng
        # track_id = int(track_id)
        for info in each_track:
            writer.writerow(info)
    save_file.close()

    print('total_unlabeled_frames: ', total_unlabeled_frames)

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

frames_folder = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/unlabeled_target_data'
tracking_folder = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results'
# combined_file = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_3gt.csv'
# combined_file = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_6gt.csv'
# combined_file = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_10gt.csv'
# combined_file = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_10gt_5%.csv'
combined_file = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_10gt_1%.csv'
# combined_file = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel6-546-250.csv'
# combined_file = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel9-136-072.csv'
# combined_file = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel15-761-819-AK5038.csv'
# combined_file = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel11-953.csv'
# combined_file = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel5-515.csv'
# combined_file = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/combined_unlabeled_target_data_vessel10-593.csv'




hauls = os.listdir(frames_folder)

# have to make sure all ids are unique!!! for later majority vote.
track_id = 1
all_unlabeled_tracks = {}
proba = 0.05

if proba:
    some_unlabeled_tracks = {}

for haul in hauls:

    if proba:
        this_unlabeled_haul = {}

    if haul in [
                'Vessel11-190927_060011-C1H-033-190929_213338_953',
                'Vessel10-190501_171646-C4H-045-190504_175415_593',
                'Vessel9-190723_210339-C2H-022-190725_021911_136',
                'AK-50308-220423_214636-C1H-025-220524_210051_809_1',
                'Vessel 5-210828_174221-C2H-023-210829_170231_515',
                'Vessel 6-210804_212407-C1H-001-210816_211530_546',
                'Vessel15-200501_232738-C2H-034-200504_205344_819',
                'Vessel15-200501_232738-C2H-033-200504_202344_761',
                'Vessel9-190723_210339-C2H-074-190727_190302_072',
                'Vessel 6-210804_212407-C1H-004-210817_160036_250',
    ]:
        tracking_result = os.path.join(tracking_folder, haul, 'Suzanne_correct_gt_including_nonfish','hierarchical_classification_result_processed_SR_GT.csv')
        track_dict = read_cls_csv_as_dict(tracking_result)
    else:
        # tracking_result = os.path.join(tracking_folder, haul, 'tracking_result_with_huber.csv')
        # track_dict=read_csv_as_dict(tracking_result)
        continue

    for track_index in track_dict:

        for frame_info in track_dict[track_index]:
            frame_info['id'] = str(track_id)
            # make the new frame path
            frame_info['filename'] = os.path.join(frames_folder, haul,frame_info['filename'])


        all_unlabeled_tracks[str(track_id)] = track_dict[track_index]
        track_id += 1
        if proba:
            this_unlabeled_haul[str(track_id)] = track_dict[track_index]


    if proba:
        # calculate the species num in this haul
        species_to_num_track = {}
        species_to_num_frames = {}

        for track_idx in this_unlabeled_haul:
            species = this_unlabeled_haul[track_idx][0]['class']
            frame_len = len(this_unlabeled_haul[track_idx])

            if species not in species_to_num_track:
                species_to_num_track[species] = 1
            else:
                species_to_num_track[species] += 1

            if species not in species_to_num_frames:
                species_to_num_frames[species] = frame_len
            else:
                species_to_num_frames[species] += frame_len

        # calculate 5% for each species in this haul
        species_to_num_track_proba = {}
        for species in species_to_num_track:
            num_track = species_to_num_track[species]
            some_num_track = np.round(proba * num_track)
            # embed()
            # if some_num_track < 1:
            #     some_num_track = 1
            species_to_num_track_proba[species] = some_num_track

        # select random 5% tracks to save
        count_species = {}
        this_unlabeled_haul = random_dic(this_unlabeled_haul)
        for track_idx in this_unlabeled_haul:
            species = this_unlabeled_haul[track_idx][0]['class']

            if species not in count_species:
                if species_to_num_track_proba[species]>=1:
                    some_unlabeled_tracks[track_idx] = this_unlabeled_haul[track_idx]
                    count_species[species] = 1
            else:
                if count_species[species] < species_to_num_track_proba[species]:
                    some_unlabeled_tracks[track_idx] = this_unlabeled_haul[track_idx]
                    count_species[species] += 1





# calculate the species num in this combination

species_to_num_track = {}
species_to_num_frames = {}

if proba:
    combined_tracks = some_unlabeled_tracks

else:
    combined_tracks = all_unlabeled_tracks

for track_id in combined_tracks:
    species = combined_tracks[track_id][0]['class']
    frame_len = len(combined_tracks[track_id])

    if species not in species_to_num_track:
        species_to_num_track[species] = 1
    else:
        species_to_num_track[species] += 1

    if species not in species_to_num_frames:
        species_to_num_frames[species] = frame_len
    else:
        species_to_num_frames[species] += frame_len

print('species_to_num_track: ', species_to_num_track)
print('species_to_num_frames: ', species_to_num_frames)

if proba:
    frame_ratio = {}
    track_ratio = {}
    gt_10_all_frames = {'Soft Snout Skates': 1136, 'Spiny Dogfish Shark': 152, 'Pacific Halibut': 6943,
                        'Hard Snout Skates': 1375,
                        'Pacific Sleeper Sharks': 4539, 'Shortraker-Rougheye-BlackSpotted Rockfish': 102,
                        'Sablefish': 3724,
                        'Kamchatka-Arrowtooth': 1146, 'Walleye Pollock': 14, 'Pacific Cod': 1920, 'Sculpin': 181,
                        'Thornyheads': 533,
                        'Yelloweye Rockfish': 65, 'Spotted Ratfish': 120, 'Redbanded Rockfish': 231, 'Starfish': 14}
    gt_10_all_tracks = {'Soft Snout Skates': 33, 'Spiny Dogfish Shark': 11, 'Pacific Halibut': 216, 'Hard Snout Skates': 27,
     'Pacific Sleeper Sharks': 15, 'Shortraker-Rougheye-BlackSpotted Rockfish': 5, 'Sablefish': 183,
     'Kamchatka-Arrowtooth': 51, 'Walleye Pollock': 2, 'Pacific Cod': 73, 'Sculpin': 4, 'Thornyheads': 24,
     'Yelloweye Rockfish': 3, 'Spotted Ratfish': 3, 'Redbanded Rockfish': 4, 'Starfish': 1}

    for species in gt_10_all_frames:
        if species in species_to_num_frames:
            frame_ratio[species] = species_to_num_frames[species] / gt_10_all_frames[species]
            track_ratio[species] = species_to_num_track[species] / gt_10_all_tracks[species]
        else:
            frame_ratio[species] = 0
            track_ratio[species] = 0

    print('total frame_ratio: ', frame_ratio)
    print('total track_ratio: ', track_ratio)

embed()

write_dict_to_csv(combined_tracks, combined_file)

