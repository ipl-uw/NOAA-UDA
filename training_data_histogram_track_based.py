#这个code 是哟弄个来展示train, evaluation, query data的比例


def sorted_dict(training_eval_data):
 sorted_tuple = sorted(training_eval_data.items(), key  = lambda item:item[1])

 sorted_dict = {}
 for each_tuple in sorted_tuple:
  sorted_dict[each_tuple[0]] = each_tuple[1]

 return sorted_dict


def get_sorted_data(names, dic):
 data_list = []
 for name in names:
  data_list.append(dic[name])

 return np.array(data_list)


##paper用的，相对少的数据
# data_distribution_track_based_total_track_tr={'Sablefish': 720,
#  'Soft Snout Skates': 38,
#  'Pacific Cod': 74,
#  'Hard Snout Skates': 84,
#  'Thornyheads': 155,
#  'Grenadier': 201,
#  'Kamchatka-Arrowtooth': 52,
#  'Starfish': 9,
#  'SRB Rockfish': 74,
#  'Pacific Halibut': 474,
#  'Flatfishes': 4,
#  'Anemones': 12,
#  'Coral': 1,
#  'Spiny Dogfish Shark': 386,
#  'Invertebrates': 4,
#  'Bivalvia': 2,
#  'Sea Urchins': 1,
#  'Snails': 2,
#  'Octopus': 1,
#  'Sponges': 1,
#  'Sculpin': 5,
#  'Yelloweye Rockfish': 12,
#  'Rockfishes': 35,
#  'Spotted Ratfish': 4,
#  'Redbanded Rockfish': 32,
#  'Mollusca': 2,
#  'Blue Shark': 2,
#  'Soupfin Shark': 0.5,
#  'Lingcod': 3,
#  'Silvergray Rockfish': 1,
#  'Canary Rockfish': 0.5,
#  'Quillback Rockfish': 10,
#  'Northern Rockfish': 1,
#  'Walleye Pollock': 1}
# data_distribution_track_based_total_track_val={'Sablefish': 180,
#  'Soft Snout Skates': 10,
#  'Pacific Cod': 19,
#  'Hard Snout Skates': 22,
#  'Thornyheads': 39,
#  'Grenadier': 51,
#  'Kamchatka-Arrowtooth': 13,
#  'Starfish': 3,
#  'SRB Rockfish': 19,
#  'Pacific Halibut': 119,
#  'Flatfishes': 1,
#  'Anemones': 3,
#  'Coral': 1,
#  'Spiny Dogfish Shark': 97,
#  'Invertebrates': 1,
#  'Bivalvia': 1,
#  'Sea Urchins': 1,
#  'Snails': 1,
#  'Octopus': 1,
#  'Sponges': 1,
#  'Sculpin': 2,
#  'Yelloweye Rockfish': 3,
#  'Rockfishes': 9,
#  'Spotted Ratfish': 1,
#  'Redbanded Rockfish': 9,
#  'Mollusca': 1,
#  'Blue Shark': 1,
#  'Soupfin Shark': 0.5,
#  'Lingcod': 1,
#  'Silvergray Rockfish': 1,
#  'Canary Rockfish': 0.5,
#  'Quillback Rockfish': 3,
#  'Northern Rockfish': 1,
#  'Walleye Pollock': 1}
#
# data_distribution_track_based_total_img_tr={'Sablefish': 40981,
#  'Soft Snout Skates': 3090,
#  'Pacific Cod': 3056,
#  'Hard Snout Skates': 11772,
#  'Thornyheads': 10140,
#  'Grenadier': 15890,
#  'Kamchatka-Arrowtooth': 2662,
#  'Starfish': 455,
#  'SRB Rockfish': 5148,
#  'Pacific Halibut': 26254,
#  'Flatfishes': 348,
#  'Anemones': 514,
#  'Coral': 54,
#  'Spiny Dogfish Shark': 22082,
#  'Invertebrates': 114,
#  'Bivalvia': 156,
#  'Sea Urchins': 3,
#  'Snails': 83,
#  'Octopus': 34,
#  'Sponges': 27,
#  'Sculpin': 432,
#  'Yelloweye Rockfish': 1622,
#  'Rockfishes': 1880,
#  'Spotted Ratfish': 164,
#  'Redbanded Rockfish': 1523,
#  'Mollusca': 78,
#  'Blue Shark': 296,
#  'Soupfin Shark': 29,
#  'Lingcod': 742,
#  'Silvergray Rockfish': 47,
#  'Canary Rockfish': 17,
#  'Quillback Rockfish': 342,
#  'Northern Rockfish': 51,
#  'Walleye Pollock': 65}
# data_distribution_track_based_total_img_val={'Sablefish': 9491,
#  'Soft Snout Skates': 1331,
#  'Pacific Cod': 958,
#  'Hard Snout Skates': 2994,
#  'Thornyheads': 2048,
#  'Grenadier': 3279,
#  'Kamchatka-Arrowtooth': 408,
#  'Starfish': 64,
#  'SRB Rockfish': 882,
#  'Pacific Halibut': 7563,
#  'Flatfishes': 47,
#  'Anemones': 114,
#  'Coral': 29,
#  'Spiny Dogfish Shark': 5126,
#  'Invertebrates': 40,
#  'Bivalvia': 28,
#  'Sea Urchins': 54,
#  'Snails': 26,
#  'Octopus': 65,
#  'Sponges': 27,
#  'Sculpin': 155,
#  'Yelloweye Rockfish': 141,
#  'Rockfishes': 394,
#  'Spotted Ratfish': 29,
#  'Redbanded Rockfish': 693,
#  'Mollusca': 5,
#  'Blue Shark': 77,
#  'Soupfin Shark': 29,
#  'Lingcod': 89,
#  'Silvergray Rockfish': 13,
#  'Canary Rockfish': 18,
#  'Quillback Rockfish': 94,
#  'Northern Rockfish': 70,
#  'Walleye Pollock': 60}

##增加了一些数据后
# data_distribution_track_based_total_track_tr = {'Sablefish': 932,
#  'Soft Snout Skates': 55,
#  'Pacific Cod': 90,
#  'Hard Snout Skates': 113,
#  'Thornyheads': 231,
#  'Grenadier': 241,
#  'Kamchatka-Arrowtooth': 70,
#  'Starfish': 22,
#  'SRB Rockfish': 110,
#  'Pacific Halibut': 596,
#  'Flatfishes': 12,
#  'Anemones': 19,
#  'Coral': 3,
#  'Spiny Dogfish Shark': 479,
#  'Invertebrates': 4,
#  'Bivalvia': 2,
#  'Sea Urchins': 1,
#  'Snails': 4,
#  'Octopus': 1,
#  'Sponges': 2,
#  'Sculpin': 11,
#  'Yelloweye Rockfish': 13,
#  'Rockfishes': 101,
#  'Redbanded Rockfish': 75,
#  'Northern Rockfish': 2,
#  'Spotted Ratfish': 4,
#  'Mollusca': 2,
#  'Skates': 0.5,
#  'Lingcod': 5,
#  'Blue Shark': 2,
#  'Soupfin Shark': 0.5,
#  'Silvergray Rockfish': 1,
#  'Canary Rockfish': 0.5,
#  'Quillback Rockfish': 11,
#  'Walleye Pollock': 1}
#
# data_distribution_track_based_total_track_val={'Sablefish': 104,
#  'Soft Snout Skates': 7,
#  'Pacific Cod': 10,
#  'Hard Snout Skates': 13,
#  'Thornyheads': 26,
#  'Grenadier': 27,
#  'Kamchatka-Arrowtooth': 8,
#  'Starfish': 3,
#  'SRB Rockfish': 13,
#  'Pacific Halibut': 67,
#  'Flatfishes': 2,
#  'Anemones': 3,
#  'Coral': 1,
#  'Spiny Dogfish Shark': 54,
#  'Invertebrates': 1,
#  'Bivalvia': 1,
#  'Sea Urchins': 1,
#  'Snails': 1,
#  'Octopus': 1,
#  'Sponges': 1,
#  'Sculpin': 2,
#  'Yelloweye Rockfish': 2,
#  'Rockfishes': 12,
#  'Redbanded Rockfish': 9,
#  'Northern Rockfish': 1,
#  'Spotted Ratfish': 1,
#  'Mollusca': 1,
#  'Skates': 0.5,
#  'Lingcod': 1,
#  'Blue Shark': 1,
#  'Soupfin Shark': 0.5,
#  'Silvergray Rockfish': 1,
#  'Canary Rockfish': 0.5,
#  'Quillback Rockfish': 2,
#  'Walleye Pollock': 1}
#
#
# data_distribution_track_based_total_img_tr={'Sablefish': 53297,
#  'Soft Snout Skates': 4792,
#  'Pacific Cod': 4120,
#  'Hard Snout Skates': 15113,
#  'Thornyheads': 13625,
#  'Grenadier': 19026,
#  'Kamchatka-Arrowtooth': 3508,
#  'Starfish': 912,
#  'SRB Rockfish': 7137,
#  'Pacific Halibut': 34307,
#  'Flatfishes': 684,
#  'Anemones': 800,
#  'Coral': 108,
#  'Spiny Dogfish Shark': 27704,
#  'Invertebrates': 114,
#  'Bivalvia': 156,
#  'Sea Urchins': 3,
#  'Snails': 149,
#  'Octopus': 65,
#  'Sponges': 54,
#  'Sculpin': 571,
#  'Yelloweye Rockfish': 1686,
#  'Rockfishes': 5236,
#  'Redbanded Rockfish': 4176,
#  'Northern Rockfish': 175,
#  'Spotted Ratfish': 164,
#  'Mollusca': 43,
#  'Skates': 34,
#  'Lingcod': 753,
#  'Blue Shark': 296,
#  'Soupfin Shark': 29,
#  'Silvergray Rockfish': 47,
#  'Canary Rockfish': 17,
#  'Quillback Rockfish': 390,
#  'Walleye Pollock': 65}
#
# data_distribution_track_based_total_img_val={'Sablefish': 4810,
#  'Soft Snout Skates': 640,
#  'Pacific Cod': 442,
#  'Hard Snout Skates': 1218,
#  'Thornyheads': 2190,
#  'Grenadier': 1146,
#  'Kamchatka-Arrowtooth': 298,
#  'Starfish': 188,
#  'SRB Rockfish': 661,
#  'Pacific Halibut': 3843,
#  'Flatfishes': 22,
#  'Anemones': 145,
#  'Coral': 104,
#  'Spiny Dogfish Shark': 2150,
#  'Invertebrates': 40,
#  'Bivalvia': 28,
#  'Sea Urchins': 54,
#  'Snails': 27,
#  'Octopus': 34,
#  'Sponges': 31,
#  'Sculpin': 258,
#  'Yelloweye Rockfish': 77,
#  'Rockfishes': 844,
#  'Redbanded Rockfish': 496,
#  'Northern Rockfish': 51,
#  'Spotted Ratfish': 29,
#  'Mollusca': 40,
#  'Skates': 35,
#  'Lingcod': 247,
#  'Blue Shark': 77,
#  'Soupfin Shark': 29,
#  'Silvergray Rockfish': 13,
#  'Canary Rockfish': 18,
#  'Quillback Rockfish': 46,
#  'Walleye Pollock': 60}

# 增加 20180824T162340-0800 sleeper  sharks
# data_distribution_track_based_total_track_tr = {'Sablefish': 932,
#  'Soft Snout Skates': 55,
#  'Pacific Cod': 90,
#  'Hard Snout Skates': 113,
#  'Thornyheads': 231,
#  'Grenadier': 241,
#  'Kamchatka-Arrowtooth': 70,
#  'Starfish': 22,
#  'SRB Rockfish': 110,
#  'Pacific Halibut': 596,
#  'Flatfishes': 12,
#  'Anemones': 19,
#  'Coral': 3,
#  'Spiny Dogfish Shark': 479,
#  'Invertebrates': 4,
#  'Bivalvia': 2,
#  'Sea Urchins': 1,
#  'Snails': 4,
#  'Octopus': 1,
#  'Sponges': 2,
#  'Sculpin': 11,
#  'Yelloweye Rockfish': 13,
#  'Rockfishes': 101,
#  'Redbanded Rockfish': 75,
#  'Northern Rockfish': 2,
#  'Spotted Ratfish': 4,
#  'Mollusca': 2,
#  'Skates': 0.5,
#  'Lingcod': 5,
#  'Blue Shark': 2,
#  'Sleep sharks': 6,
#  'Soupfin Shark': 0.5,
#  'Silvergray Rockfish': 1,
#  'Canary Rockfish': 0.5,
#  'Quillback Rockfish': 11,
#  'Walleye Pollock': 1}
#
# data_distribution_track_based_total_track_val={'Sablefish': 104,
#  'Soft Snout Skates': 7,
#  'Pacific Cod': 10,
#  'Hard Snout Skates': 13,
#  'Thornyheads': 26,
#  'Grenadier': 27,
#  'Kamchatka-Arrowtooth': 8,
#  'Starfish': 3,
#  'SRB Rockfish': 13,
#  'Pacific Halibut': 67,
#  'Flatfishes': 2,
#  'Anemones': 3,
#  'Coral': 1,
#  'Spiny Dogfish Shark': 54,
#  'Invertebrates': 1,
#  'Bivalvia': 1,
#  'Sea Urchins': 1,
#  'Snails': 1,
#  'Octopus': 1,
#  'Sponges': 1,
#  'Sculpin': 2,
#  'Yelloweye Rockfish': 2,
#  'Rockfishes': 12,
#  'Redbanded Rockfish': 9,
#  'Northern Rockfish': 1,
#  'Spotted Ratfish': 1,
#  'Mollusca': 1,
#  'Skates': 0.5,
#  'Lingcod': 1,
#  'Blue Shark': 1,
# 'Sleep sharks': 3,
#  'Soupfin Shark': 0.5,
#  'Silvergray Rockfish': 1,
#  'Canary Rockfish': 0.5,
#  'Quillback Rockfish': 2,
#  'Walleye Pollock': 1}


data_distribution_track_based_total_track_tr = {'Pacific Sleeper Sharks': 43,
 'Pacific Halibut': 611,
 'Kamchatka-Arrowtooth': 77,
 'Pacific Cod': 90,
 'Sablefish': 945,
 'Soft Snout Skates': 55,
 'Hard Snout Skates': 117,
 'Thornyheads': 227,
 'Grenadier': 238,
 'Starfish': 22,
 'Shortraker-Rougheye-BlackSpotted Rockfish': 107,
 'Coral': 15,
 'Anemones': 21,
 'Spiny Dogfish Shark': 472,
 'Bivalvia': 2,
 'Sea Urchins': 1,
 'Snails': 4,
 'Octopus': 1,
 'Sculpin': 12,
 'Yelloweye Rockfish': 13,
 'Sponges': 2,
 'Redbanded Rockfish': 75,
 'Northern Rockfish': 2,
 'Spotted Ratfish': 4,
 'Dover Sole': 1,
 'Mollusca': 2,
 'Lingcod': 4,
 'Blue Shark': 2,
 'Flathead Sole': 6,
 'Soupfin Shark': 1,
 'Silvergray Rockfish': 1,
 'Canary Rockfish': 1,
 'Quillback Rockfish': 11,
 'Walleye Pollock': 1}


data_distribution_track_based_total_track_val = {'Pacific Sleeper Sharks': 15,
 'Sablefish': 207,
 'Grenadier': 53,
 'Thornyheads': 59,
 'Pacific Halibut': 154,
 'Hard Snout Skates': 26,
 'Soft Snout Skates': 12,
 'Bivalvia': 2,
 'Pacific Cod': 20,
 'Kamchatka-Arrowtooth': 17,
 'Snails': 2,
 'Starfish': 6,
 'Redbanded Rockfish': 19,
 'Sculpin': 3,
 'Anemones': 5,
 'Spiny Dogfish Shark': 100,
 'Sponges': 2,
 'Dover Sole': 1,
 'Shortraker-Rougheye-BlackSpotted Rockfish': 27,
 'Mollusca': 1,
 'Soupfin Shark': 1,
 'Blue Shark': 2,
 'Octopus': 1,
 'Yelloweye Rockfish': 4,
 'Silvergray Rockfish': 1,
 'Canary Rockfish': 1,
 'Spotted Ratfish': 1,
 'Flathead Sole': 2,
 'Northern Rockfish': 1,
 'Coral': 4,
 'Walleye Pollock': 1,
 'Sea Urchins': 1,
 'Quillback Rockfish': 2,
 'Lingcod': 2}


# this seems wrong
# data_distribution_track_based_total_img_tr={'Sablefish': 53297,
#  'Soft Snout Skates': 4792,
#  'Pacific Cod': 4120,
#  'Hard Snout Skates': 15113,
#  'Thornyheads': 13625,
#  'Grenadier': 19026,
#  'Kamchatka-Arrowtooth': 3508,
#  'Starfish': 912,
#  'SRB Rockfish': 7137,
#  'Pacific Halibut': 34307,
#  'Flatfishes': 684,
#  'Anemones': 800,
#  'Coral': 108,
#  'Spiny Dogfish Shark': 27704,
#  'Invertebrates': 114,
#  'Bivalvia': 156,
#  'Sea Urchins': 3,
#  'Snails': 149,
#  'Octopus': 65,
#  'Sponges': 54,
#  'Sculpin': 571,
#  'Yelloweye Rockfish': 1686,
#  'Rockfishes': 5236,
#  'Redbanded Rockfish': 4176,
#  'Northern Rockfish': 175,
#  'Spotted Ratfish': 164,
#  'Mollusca': 43,
#  'Skates': 34,
#  'Lingcod': 753,
#  'Blue Shark': 296,
# 'Sleep sharks': 9337,
#  'Soupfin Shark': 29,
#  'Silvergray Rockfish': 47,
#  'Canary Rockfish': 17,
#  'Quillback Rockfish': 390,
#  'Walleye Pollock': 65}
# this seems wrong
# data_distribution_track_based_total_img_val={'Sablefish': 4810,
#  'Soft Snout Skates': 640,
#  'Pacific Cod': 442,
#  'Hard Snout Skates': 1218,
#  'Thornyheads': 2190,
#  'Grenadier': 1146,
#  'Kamchatka-Arrowtooth': 298,
#  'Starfish': 188,
#  'SRB Rockfish': 661,
#  'Pacific Halibut': 3843,
#  'Flatfishes': 22,
#  'Anemones': 145,
#  'Coral': 104,
#  'Spiny Dogfish Shark': 2150,
#  'Invertebrates': 40,
#  'Bivalvia': 28,
#  'Sea Urchins': 54,
#  'Snails': 27,
#  'Octopus': 34,
#  'Sponges': 31,
#  'Sculpin': 258,
#  'Yelloweye Rockfish': 77,
#  'Rockfishes': 844,
#  'Redbanded Rockfish': 496,
#  'Northern Rockfish': 51,
#  'Spotted Ratfish': 29,
#  'Mollusca': 40,
#  'Skates': 35,
#  'Lingcod': 247,
#  'Blue Shark': 77,
# 'Sleep sharks': 5856,
#  'Soupfin Shark': 29,
#  'Silvergray Rockfish': 13,
#  'Canary Rockfish': 18,
#  'Quillback Rockfish': 46,
#  'Walleye Pollock': 60}


# data_distribution_track_based_total_img_tr={'Sablefish': 49939, 'Soft Snout Skates': 4555, 'Hard Snout Skates': 14994, 'Pacific Cod': 3852, 'Thornyheads': 12757, 'Grenadier': 16083, 'Kamchatka-Arrowtooth': 3203, 'Starfish': 818, 'Pacific Halibut': 32397, 'Anemones': 721, 'Coral': 176, 'Shortraker-Rougheye-BlackSpotted Rockfish': 6447, 'Spiny Dogfish Shark': 26639, 'Bivalvia': 156, 'Sea Urchins': 3, 'Snails': 104, 'Octopus': 34, 'Sponges': 54, 'Sculpin': 536, 'Yelloweye Rockfish': 1637, 'Redbanded Rockfish': 3894, 'Spotted Ratfish': 123, 'Mollusca': 78, 'Blue Shark': 296, 'Soupfin Shark': 29, 'Lingcod': 831, 'Silvergray Rockfish': 13, 'Canary Rockfish': 17, 'Quillback Rockfish': 383, 'Northern Rockfish': 121, 'Walleye Pollock': 65, 'Sleeper sharks': 9337}
#
# data_distribution_track_based_total_img_val={'Soft Snout Skates': 738, 'Pacific Cod': 465, 'Sablefish': 4436, 'Grenadier': 1276, 'Thornyheads': 1521, 'Shortraker-Rougheye-BlackSpotted Rockfish': 773, 'Pacific Halibut': 3439, 'Anemones': 174, 'Coral': 25, 'Starfish': 139, 'Kamchatka-Arrowtooth': 427, 'Spiny Dogfish Shark': 1824, 'Bivalvia': 28, 'Hard Snout Skates': 996, 'Redbanded Rockfish': 428, 'Northern Rockfish': 21, 'Sculpin': 142, 'Sponges': 31, 'Mollusca': 5, 'Lingcod': 169, 'Soupfin Shark': 29, 'Blue Shark': 77, 'Octopus': 65, 'Yelloweye Rockfish': 126, 'Silvergray Rockfish': 47, 'Canary Rockfish': 18, 'Quillback Rockfish': 53, 'Spotted Ratfish': 55, 'Walleye Pollock': 60, 'Snails': 57, 'Sea Urchins': 54, 'Sleeper sharks': 5856}

data_distribution_track_based_total_img_tr = {'Pacific Sleeper Sharks': 23375,
 'Pacific Halibut': 36276,
 'Kamchatka-Arrowtooth': 3727,
 'Pacific Cod': 4132,
 'Sablefish': 52883,
 'Soft Snout Skates': 4436,
 'Hard Snout Skates': 14805,
 'Thornyheads': 14058,
 'Grenadier': 17768,
 'Starfish': 984,
 'Shortraker-Rougheye-BlackSpotted Rockfish': 6813,
 'Coral': 1204,
 'Anemones': 898,
 'Spiny Dogfish Shark': 23597,
 'Bivalvia': 100,
 'Sea Urchins': 3,
 'Snails': 119,
 'Octopus': 34,
 'Sculpin': 745,
 'Yelloweye Rockfish': 1658,
 'Sponges': 58,
 'Redbanded Rockfish': 4410,
 'Northern Rockfish': 156,
 'Spotted Ratfish': 138,
 'Dover Sole': 13,
 'Mollusca': 78,
 'Lingcod': 987,
 'Blue Shark': 178,
 'Flathead Sole': 238,
 'Soupfin Shark': 29,
 'Silvergray Rockfish': 47,
 'Canary Rockfish': 17,
 'Quillback Rockfish': 399,
 'Walleye Pollock': 65}



data_distribution_track_based_total_img_val = {'Pacific Sleeper Sharks': 7844,
 'Sablefish': 13520,
 'Grenadier': 4589,
 'Thornyheads': 3880,
 'Pacific Halibut': 9935,
 'Hard Snout Skates': 4427,
 'Soft Snout Skates': 1868,
 'Bivalvia': 112,
 'Pacific Cod': 977,
 'Kamchatka-Arrowtooth': 1001,
 'Snails': 83,
 'Starfish': 203,
 'Redbanded Rockfish': 987,
 'Sculpin': 195,
 'Anemones': 175,
 'Spiny Dogfish Shark': 9331,
 'Sponges': 58,
 'Dover Sole': 26,
 'Shortraker-Rougheye-BlackSpotted Rockfish': 1943,
 'Mollusca': 10,
 'Soupfin Shark': 58,
 'Blue Shark': 272,
 'Octopus': 130,
 'Yelloweye Rockfish': 336,
 'Silvergray Rockfish': 26,
 'Canary Rockfish': 36,
 'Spotted Ratfish': 110,
 'Flathead Sole': 35,
 'Northern Rockfish': 140,
 'Coral': 570,
 'Walleye Pollock': 120,
 'Sea Urchins': 108,
 'Quillback Rockfish': 74,
 'Lingcod': 182}


import numpy as np
import matplotlib.pyplot as plt
from IPython import embed


data_list = [(data_distribution_track_based_total_track_tr, data_distribution_track_based_total_track_val),(data_distribution_track_based_total_img_tr,data_distribution_track_based_total_img_val)]
xlables = ["#Tracks", "#Frames"]
# save_names = ['./data distribution track-based-more_plus_sleeper_shark', './data distribution img-based-more_plus_sleeper_shark']
save_names = ['./track-based-level2_only',
              './frame-based-level2_only']

txt_loc = [600, 30000]
txt=['total #tracks', 'total #frames']
cof=[1.5,1.5]
for i, tuple in enumerate(data_list):
 tr = tuple[0]
 val = tuple[1]
 #排序作图
 sorted_track_tr = sorted_dict(tr)
 all_cls_names = list(sorted_track_tr.keys())
 sorted_track_tr_data = np.array(list(sorted_track_tr.values()))
 sorted_track_val_data = get_sorted_data(all_cls_names,val)
 # embed()

 X = np.arange(len(all_cls_names))
 X_legend = all_cls_names
 bar_width = 0.7
 fig = plt.figure(figsize=(7, 6))
 plt.barh(X, sorted_track_tr_data, bar_width, color = 'darkorange',label='Train')
 plt.barh(X, sorted_track_val_data, bar_width, color = 'gray',left=sorted_track_tr_data,label='Valid ')

 plt.title('Data Distribution')
 plt.xlabel(xlables[i])
 plt.legend(loc = 'lower left',fontsize='small')
 plt.yticks(X, X_legend)

 for a, b in zip(X, sorted_track_tr_data):
  if b<1:
   plt.text( txt_loc[i], a-0.35,'%.1f ' % b, ha='center', va='bottom', fontsize=8)
  else:
   plt.text(txt_loc[i], a - 0.35, '%.0f ' % b, ha='center', va='bottom', fontsize=8)
 for a, b in zip(X, sorted_track_val_data):
  if b < 1:
   plt.text(txt_loc[i]*1.5, a-0.35,  '%.1f ' % b, ha='center', va='bottom', fontsize=8,color = "darkgreen")
  else:
   plt.text(txt_loc[i]*1.5, a - 0.35, '%.0f ' % b, ha='center', va='bottom', fontsize=8,color = "darkgreen")
 plt.tight_layout()
 total_num = np.sum(sorted_track_tr_data) + np.sum(sorted_track_val_data)
 plt.text(txt_loc[i] * cof[i], 3, txt[i]+': %.0f ' % total_num, ha='center', va='bottom', fontsize=8)
 # from IPython import embed
 # embed()
 plt.savefig(save_names[i], dpi=300)
 plt.show()
embed()










#第二种画图方式
plt.figure(figsize=(10,6))

#设置x轴柱子的个数
x=np.arange(len(all_cls_names))+1 #课程品类数量已知为14，也可以用len(ppv3.index)

#设置y轴的数值，需将numbers列的数据先转化为数列，再转化为矩阵格式
y=train
y1 = eval
xticks1=all_cls_names #构造不同课程类目的数列

#画出柱状图
# plt.bar(x,y,width = 0.35,align='center',color = 'deeppink',alpha=0.8, label = 'trian')
# plt.bar(x,y1,width = 0.35,align='center',color = 'darkblue',alpha=0.8, label = 'eval', bottom=y)
plt.barh(x,y,color = 'darkorange',alpha=0.8, label = 'trian')
plt.barh(x,y1,color = 'gray',alpha=0.8, label = 'eval', left=y)


#设置x轴的刻度，将构建的xticks代入，同时由于课程类目文字较多，在一块会比较拥挤和重叠，因此设置字体和对齐方式
# plt.xticks(x,xticks1,size='small',rotation=90)
plt.yticks(x,xticks1,size='small',rotation=0)

#x、y轴标签与图形标题
plt.ylabel('species')
plt.xlabel('#images/labels')
plt.title('Data Distribution')
plt.legend(loc = 'lower left')

#设置数字标签
for a,b,c in zip(x,y,y+y1):
 # plt.text(c+5000, a-0.5,'%.0f' % b, ha='center', va= 'bottom',fontsize=10)
 plt.text(20000, a - 0.5, '%.0f' % b, ha='center', va='bottom', fontsize=10 )

for a,b,c in zip(x,y1, y+y1):
 # plt.text(c+15000,a-0.5, '%.0f' % b, ha='center', va= 'bottom',fontsize=10)
 plt.text(30000, a - 0.5, '%.0f' % b, ha='center', va='bottom', fontsize=10, color = "darkgreen")


total_num = 0
for each in training_eval_data:
 total_num +=training_eval_data[each]
plt.text(40000, 3, 'total #images: %d' % total_num, ha='center', va='bottom', fontsize=10  )
#设置y轴的范围
# plt.ylim(0,20000)
plt.tight_layout()
plt.savefig('./data distribution.png', dpi=300)
plt.show()

embed()