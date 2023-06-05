
from util import read_csv_as_dict
from IPython import embed
from util import level_2_names
from class_alias import Class_alias

def get_pred(pred, track_id):
    vote = {}
    for frame_info in pred[track_id]:
        species = frame_info['species']
        species_conf = float(frame_info['species_conf'])

        if species not in vote:
            vote[species] = species_conf
        else:
            vote[species] += species_conf

    # embed()
    y = max(zip(vote.values(), vote.keys()))

    return y[1]


# gt_csv = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-001-210816_211530_546/Suzanne_correct_gt_including_nonfish/hierarchical_classification_result_processed_SR_GT.csv'
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-001-210816_211530_546/Suzanne_correct_gt_including_nonfish/flat_classification_result_aug_cutmix_autoaug_on_SR_gt.csv'
# # seperate UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-001-210816_211530_546/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # generic UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-001-210816_211530_546/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--generic_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # seperate UDA - balance sampling
# pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-001-210816_211530_546/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-balance-lre-3-on_SR_gt.csv'
# # 5% finetuned
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-001-210816_211530_546/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_5%.csv'
# # 1% finetuned
# pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-001-210816_211530_546/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_1%.csv'


gt_csv = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-004-210817_160036_250/Suzanne_correct_gt_including_nonfish/hierarchical_classification_result_processed_SR_GT.csv'
# pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-004-210817_160036_250/Suzanne_correct_gt_including_nonfish/flat_classification_result_aug_cutmix_autoaug_on_SR_gt.csv'
# seperate UDA
# pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-004-210817_160036_250/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# generic UDA
# pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-004-210817_160036_250/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--generic_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# seperate UDA - balance sampling
# pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-004-210817_160036_250/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-balance-lre-3-on_SR_gt.csv'
pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-004-210817_160036_250/Suzanne_correct_gt_including_nonfish/flat_2_hierarchy_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-balance-lre-3-on_SR_gt.csv'
# 5% finetuned
# pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-004-210817_160036_250/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_5%.csv'
# 1% finetuned
pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 6-210804_212407-C1H-004-210817_160036_250/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_1%.csv'



# gt_csv = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-022-190725_021911_136/Suzanne_correct_gt_including_nonfish/hierarchical_classification_result_processed_SR_GT.csv'
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-022-190725_021911_136/Suzanne_correct_gt_including_nonfish/flat_classification_result_aug_cutmix_autoaug_on_SR_gt.csv'
# # seperate UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-022-190725_021911_136/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # generic UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-022-190725_021911_136/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--generic_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # seperate UDA - balance sampling
# pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-022-190725_021911_136/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-balance-lre-6-on_SR_gt.csv'
# # 5% finetuned
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-022-190725_021911_136/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_5%.csv'
# # 1% finetuned
# pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-022-190725_021911_136/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_1%.csv'


# gt_csv = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-074-190727_190302_072/Suzanne_correct_gt_including_nonfish/hierarchical_classification_result_processed_SR_GT.csv'
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-074-190727_190302_072/Suzanne_correct_gt_including_nonfish/flat_classification_result_aug_cutmix_autoaug_on_SR_gt.csv'
# # seperate UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-074-190727_190302_072/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # generic UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-074-190727_190302_072/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--generic_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # seperate UDA - balance sampling
# pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-074-190727_190302_072/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-balance-lre-6-on_SR_gt.csv'
# # 5% finetuned
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-074-190727_190302_072/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_5%.csv'
# # 1% finetuned
# pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel9-190723_210339-C2H-074-190727_190302_072/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_1%.csv'


# gt_csv = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-033-200504_202344_761/Suzanne_correct_gt_including_nonfish/hierarchical_classification_result_processed_SR_GT.csv'
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-033-200504_202344_761/Suzanne_correct_gt_including_nonfish/flat_classification_result_aug_cutmix_autoaug_on_SR_gt.csv'
# # seperate UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-033-200504_202344_761/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # generic UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-033-200504_202344_761/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--generic_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # seperate UDA - balance sampling
# pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-033-200504_202344_761/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-balance-lre-3-on_SR_gt.csv'
# # 5% finetuned
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-033-200504_202344_761/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_5%.csv'
# # 1% finetuned
# pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-033-200504_202344_761/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_1%.csv'




# gt_csv = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-034-200504_205344_819/Suzanne_correct_gt_including_nonfish/hierarchical_classification_result_processed_SR_GT.csv'
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-034-200504_205344_819/Suzanne_correct_gt_including_nonfish/flat_classification_result_aug_cutmix_autoaug_on_SR_gt.csv'
# # seperate UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-034-200504_205344_819/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # generic UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-034-200504_205344_819/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--generic_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # seperate UDA - balance sampling
# pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-034-200504_205344_819/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-balance-lre-3-on_SR_gt.csv'
# # 5% finetuned
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-034-200504_205344_819/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_5%.csv'
# # 1% finetuned
# pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel15-200501_232738-C2H-034-200504_205344_819/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_1%.csv'



# gt_csv = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/AK-50308-220423_214636-C1H-025-220524_210051_809_1/Suzanne_correct_gt_including_nonfish/hierarchical_classification_result_processed_SR_GT.csv'
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/AK-50308-220423_214636-C1H-025-220524_210051_809_1/Suzanne_correct_gt_including_nonfish/flat_classification_result_aug_cutmix_autoaug_on_SR_gt.csv'
# # seperate UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/AK-50308-220423_214636-C1H-025-220524_210051_809_1/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # generic UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/AK-50308-220423_214636-C1H-025-220524_210051_809_1/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--generic_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # seperate UDA - balance sampling
# pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/AK-50308-220423_214636-C1H-025-220524_210051_809_1/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-balance-lre-3-on_SR_gt.csv'
# # 5% finetuned
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/AK-50308-220423_214636-C1H-025-220524_210051_809_1/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_5%.csv'
# # 1% finetuned
# pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/AK-50308-220423_214636-C1H-025-220524_210051_809_1/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_1%.csv'


# gt_csv = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel11-190927_060011-C1H-033-190929_213338_953/Suzanne_correct_gt_including_nonfish/hierarchical_classification_result_processed_SR_GT.csv'
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel11-190927_060011-C1H-033-190929_213338_953/Suzanne_correct_gt_including_nonfish/flat_classification_result_aug_cutmix_autoaug_on_SR_gt.csv'
# # seperate UDA
# pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel11-190927_060011-C1H-033-190929_213338_953/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # generic UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel11-190927_060011-C1H-033-190929_213338_953/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--generic_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # 5% finetuned
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel11-190927_060011-C1H-033-190929_213338_953/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_5%.csv'
# # 1% finetuned
# pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel11-190927_060011-C1H-033-190929_213338_953/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_1%.csv'


# gt_csv = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 5-210828_174221-C2H-023-210829_170231_515/Suzanne_correct_gt_including_nonfish/hierarchical_classification_result_processed_SR_GT.csv'
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 5-210828_174221-C2H-023-210829_170231_515/Suzanne_correct_gt_including_nonfish/flat_classification_result_aug_cutmix_autoaug_on_SR_gt.csv'
# # seperate UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 5-210828_174221-C2H-023-210829_170231_515/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # generic UDA
# pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 5-210828_174221-C2H-023-210829_170231_515/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--generic_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # 5% finetuned
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 5-210828_174221-C2H-023-210829_170231_515/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_5%.csv'
# # 1% finetuned
# pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel 5-210828_174221-C2H-023-210829_170231_515/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_1%.csv'


# gt_csv = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel10-190501_171646-C4H-045-190504_175415_593/Suzanne_correct_gt_including_nonfish/hierarchical_classification_result_processed_SR_GT.csv'
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel10-190501_171646-C4H-045-190504_175415_593/Suzanne_correct_gt_including_nonfish/flat_classification_result_aug_cutmix_autoaug_on_SR_gt.csv'
# # seperate UDA
# pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel10-190501_171646-C4H-045-190504_175415_593/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--separate_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # generic UDA
# # pred_csv_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel10-190501_171646-C4H-045-190504_175415_593/Suzanne_correct_gt_including_nonfish/flat_classification_result_lmmd--generic_target_data-aug_cutmix_autoaug-lre-6-on_SR_gt.csv'
# # 5% finetuned
# # pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel10-190501_171646-C4H-045-190504_175415_593/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_5%.csv'
# # 1% finetuned
# pred_csv_no_lmmd = '/home/jiemei/Documents/vit_for_rail/test_sleeper_shark/results/Vessel10-190501_171646-C4H-045-190504_175415_593/Suzanne_correct_gt_including_nonfish/flat_classification_result_finetune_1%.csv'




gt = read_csv_as_dict(gt_csv)
pred_no_lmmd = read_csv_as_dict(pred_csv_no_lmmd)
pred_lmmd = read_csv_as_dict(pred_csv_lmmd)

alias_func = Class_alias()

tp_no_lmmd=0
tp_lmmd=0
total=0


acc_no_lmmd_species={}
acc_lmmd_species={}
total_species={}
total_species_frames={}


for track_id in gt:
    label = gt[track_id][0]['species']
    num_frame = len(gt[track_id])

    label, found = alias_func.convertName(label)

    # embed()
    if label not in level_2_names:
        print(label)
        continue

    if label not in total_species:
        total_species[label] =1
        acc_lmmd_species[label] = 0
        acc_no_lmmd_species[label] = 0

        # embed()
        total_species_frames[label]=num_frame
    else:
        total_species[label] += 1
        total_species_frames[label] += num_frame




    total+=1

    y_no_lmmd=get_pred(pred_no_lmmd, track_id)
    y_lmmd = get_pred(pred_lmmd, track_id)

    # embed()
    if y_lmmd == label:
        tp_lmmd+=1
        acc_lmmd_species[label] += 1

    if y_no_lmmd ==label:
        tp_no_lmmd+=1
        acc_no_lmmd_species[label] += 1

for label in total_species:
    acc_lmmd_species[label]/=total_species[label]
    acc_no_lmmd_species[label] /= total_species[label]

print('total tracks: %d'%total)
print('avg acc no lmmd: %.2f'% (tp_no_lmmd/total *100))
print('avg acc with lmmd: %.2f'% (tp_lmmd/total *100))

print('acc no lmmd - each species: ', acc_no_lmmd_species)
print('acc lmmd - each species: ', acc_lmmd_species)

print('total species: %d' %len(acc_lmmd_species))
embed()
total_species
total_species_frames