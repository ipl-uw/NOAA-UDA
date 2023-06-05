### define alias of classes if there are any

# default_alias = {
#     'Spiny Dogfish Shark': ['Dogfish Shark', 'Spiny Dogfish', 'Spinydogfish Shark', 'Spiny Dogfish Shark_Discard'],
#     'Pacific Halibut': ['Halibut', 'Halibuts', 'Halibut_Discard', 'Pacific Halibut_Discard'],
#     'Kamchatka/Arrowtooth/Turbot complex': ['Kamchatka/Arrowtooth_Flounder_UnID', 'Kamchatka_Flounder/Arrowtooth_complex'],
#     'Kamchatka Flounder': ['Kamchatka'], # from Predator_2018
#     'Shortraker/Rougheye/BlackSpotted Rockfish unidentified': ['Shortraker/Rougheye_Rockfish', 'Shortraker/Rougheye_unidentified', 'Shortraker/Rougheye Rockfish UNID'],
#     'Soft Snout Skates': ['Bathy Raja UNID', 'BathyRajaUNID', 'Bathyraja', 'Bathyraja UnID'],
#     'Anemones': ['Anemone', 'Sea Anemone'],
#     'Arrowtooth Flounder': ['Arrowtooth'],
#     'Bait_other': ['Bait_Other'],
#     'Corals': ['Coral on Rock', 'CoralUNID'],
#     'Fish': ['FishUNID', 'Fish Unidentified'],
#     'Fishing Gear': ['Fishing Gear_other'],
#     'Grenadier': ['Grenadier UNID', 'GrenadierUNID'],
#     'Grenadiers and Rattails Other': ['Grenadiers and Rattails'],
#     'Invertebrates': ['Invertabrate unidentified', 'nonfish_unid'],
#     #'Hard Snout Skates': ['Longnose Skate'],
#     'Rockfishes and Thornyheads': ['Rockfish UNID'],
#     'Sablefish': ['Sablefish_Discard'],
#     'Shortspine Thornyhead': ['Short Spine Thornyhead'],
#     'Starfish': ['StarfishUNID'], # not in species.yml
#     'Thornyhead Unidentified': ['Thornyhead', 'Thornyhead UnID', 'Thornyhead Unidentified', 'ThornyheadUNID'],
#
#     'Bird': ['Northern Fulmar_Bird On Water', 'Northern Fulmar_On Water', 'Northern Fulmar_Bird in Air'], # from Predator_2018
#     'Misc_Other': ['Misc Other']
# }

# query_learning_alias = {
#
#     'Sablefish': ['Sablefish_Discard','Sablefish Head'],
#
#     'Rockfishes': ['Rockfish UNID','Redbanded Rockfish','Red Rockfish','Red Rockfish_Discard','Yelloweye Rockfish',
#                    'Thornyhead Unidentified','ThornyheadUNID','Thornyhead','Thornyhead UNID','Thornyhead UNID Head','Thornyhead UnID','Thornyhead UnId','Shortspined Thornyhead',
#                    'Short Spine Thornyhead', 'Shortspine Thornyhead','Shortspined Thornyhead','Shortspine Thornyhead',
#                    'Blackspotted-Rougheye-Shortraker','Shortraker/Rougheye/BlackSpotted Rockfish unidentified', 'Shortraker/Rougheye unidentified', 'Shortraker/Rougheye Rockfish UNID','Shortraker Rockfish', 'Blackspotted/Rougheye/Shortraker'],
#
#     'Grenadier': ['Grenadier UNID', 'GrenadierUNID','Grenadiers and Rattails','Giant Grenadier','Grenadiers and Rattails Head','Grenadier_Head','Giant Grenadier_Discard'],
#
#     'Kamchatka-Arrowtooth-Turbot' :['Arrowtooth','Arrowtooth Flounder','Kamchatka Flounder/Arrowtooth complex','Kamchatka/Arrowtooth Flounder UnId','Kamchatka/Arrowtooth Flounder UnID','Kamchatka/Arrowtooth Flounder UNID','Kamchatka/Arrowtooth Flounder Head UNID','Kamchatka/Arrowtooth Flounde Head UNID'],
#
#     'Pacific Halibut': ['Halibut'],
#
#     'Pacific Cod' : ['Pacific Cod Head'],
#
#     'Skates': ['Bathy Raja UNID','Bathyraja UnId', 'BathyRajaUNID', 'Bathyraja', 'Bathyraja UnID','Big Skate','Longnose Skate','Soft Snout Skates'],
#
#     'Spiny Dogfish Shark': ['Dogfish Shark','Spiny dogfish shark', 'Spiny Dogfish', 'Spinydogfish Shark', 'Spiny Dogfish Shark_Discard'],
#
#     'Spotted Ratfish':['Spotted Ratfish'],
#
#     #200多label
#     # 'Bird': ['Bird_Bird on Water','Bird_Bird in Air'],
#
#     #10000多label
#     'Northern Fulmar' :['Northern Fulmar_Bird in Air','Northern Fulmar_Bird on Water'],
#
#     #10000多label
#     'Gulls':['Gulls_Bird in Air','Gull_Bird on Water','Gull_Bird in Air',],
#
#     #3000多label
#     'Albatrosses' : ['Black-footed Albatross_Bird on Water','Black-footed Albatross-Bird on Water']
#
#     # 不是目前关注的对象，属于Invertebrates，而且类别有点杂
#     # 'Corals': ['Coral on Rock', 'CoralUNID'],
#
#     # 不是目前关注的对象，而且类别有点杂
#     # 'Fishing Gear': ['Anchor','Gangion','Hook','Mag Knot','Pole Gaff','Snap Clip','Weight','Fishing Gear_other'],
#
#     # 不是目前关注的对象，而且类别有点杂
#     # 'Bait': ['Bait_Other','Bait_Discard','Bait_Cod','Bait_Combo','Bait_Herring','Bait_Octopus','Bait_Salmon','Bait_Sardines','Bait_Squid'],
#
#     # label太少了，89+25
#     # 'Invertebrates': ['Invertabrate unidentified', 'nonfish_unid'],
#
#     #  类别太杂了
#     # 'Misc_Other': ['Misc Other']
# }


hierarchy_alias = {



    'Skates':['Skate'],
    'Hard Snout Skates':['Longnose Skate','Longnose Skate_discard','Longnose Skate_Discard','Big Skate','Hard Snout Skate'],
    'Soft Snout Skates': ['Bathy Raja UNID','Bathyraja UnId', 'BathyRajaUNID', 'Bathyraja', 'Bathyraja UnID','Soft Snout Skates',
                          'Soft Snout Skates_discard','Soft Snout Skates_Unidentified','Bathyraja Unidentified','Bathyraja Unidentifiedd',
                          'Bathyraja unidentified','Soft Snout Skate','Soft Snout Snake'],


    'Blue Shark':[],
    'Spiny Dogfish Shark': ['Dogfish Shark', 'Spiny dogfish shark', 'Spiny Dogfish', 'Spinydogfish Shark',
                            'Spiny Dogfish Shark_Discard','Spiny Dogfish Shark_discard','Dogfish Sharks'],
    'Spotted Ratfish': ['Spotted Ratfish'],
    'Soupfin Shark':[],



    'Pacific Cod' : ['Pacific Cod Head','Pacific Halibut_Discard','Pacific Cod_discard','Pacific Cod_Discard'],
    'Grenadier': ['Grenadier UNID', 'GrenadierUNID','Grenadiers and Rattails','Giant Grenadier','Grenadiers and Rattails Head',
                  'Grenadier_Head','Giant Grenadier_Discard','Giant Grenadier_discard','Grenadier_Discard','Grenadier Unidentified',
                  'Greanadier','Grenadies'],
    'Sablefish': ['Sablefish_Discard','Sablefish Head','Sablefish_discard','Sabelfish','Sablefish_Head','sablefish','Sablefish_1\''],
    'Lingcod':[],
    'Sculpin':['Yellow Irish Lord','Yellow Irish Lord_discard','Myoxocephalus unidentified','Myoxocephalus Unidentified',
               'Great Sculpin','Myoxocephalus','Sculpins','Irish Lord'],
    'Walleye Pollock':['Walleye Pollock_discard'],


    'Flatfishes':['Flatfishes Unidentified','Flatfishes','Flathfish Unidentified','Flatfish Unidentified','Flatfish_Discard'],
    'Kamchatka-Arrowtooth' :['Arrowtooth','Arrowtooth Flounder','Kamchatka Flounder/Arrowtooth complex','Kamchatka/Arrowtooth Flounder UnId',
                             'Kamchatka/Arrowtooth Flounder UnID','Kamchatka/Arrowtooth Flounder UNID','Kamchatka/Arrowtooth Flounder Head UNID',
                             'Kamchatka/Arrowtooth Flounde Head UNID','Kamchatka Flounder/Arrowtooth complex_discard','Kamchatka/Arrowtooth Unidentified',
                             'Kamchatka Flounder/Arrowtooth Unidentified','Kamchatka Flounder/Arrowtooth complex_Head','Kamchatka Flounder/Arrowtooth flounder',
                             'Kamchatka Flounder/Arrowtooth Flounder','Kamchatka/Arrowtooth/Turbot','Kamchatka/Arrowtooth/Turbot_Head','Kamchatka Flounder/Arrowtooth',
                             'Kamchatka/Arrowtooth/Turbot complex', 'Kamchatka/Arrowtooth/Turbot_Discard'],
    'Pacific Halibut': ['Halibut','Halibut Damaged','Halibut Head','Pacific Halibut_discard','Halibuit','halibut','Halibut_Discard'],
    'Flathead Sole':[],
    'Dover Sole':[],





    'Rockfishes':['Rockfish UNID','Red Rockfish','Red Rockfish_Discard', 'Rockfishes and Thornyheads_discard','Rockfish Unidentified',
                  'Rockfish Unidentified_Discard','Rockfish_Unidentified','Rockfish','Rock','Rockfish_Discard','Rock_Discard','Rockfishes and Thornyheads'],

    'Thornyheads':['Thornyhead Unidentified','ThornyheadUNID','Thornyhead','Thornyhead UNID','Thornyhead UNID Head','Thornyhead UnID','Thornyhead UnId','Shortspined Thornyhead',
                   'Short Spine Thornyhead', 'Shortspine Thornyhead','Shortspined Thornyhead','Shortspine Thornyhead','Shortspined Thornyhead_Discard'],
    'Redbanded Rockfish':['Redbanded Rockfish','Redbanded Rockfish_Discard'],
    'Shortraker-Rougheye-BlackSpotted Rockfish':['Blackspotted-Rougheye-Shortraker','Shortraker/Rougheye/BlackSpotted Rockfish unidentified',
                                                 'Shortraker/Rougheye unidentified', 'Shortraker/Rougheye Rockfish UNID','Shortraker Rockfish',
                                                 'Blackspotted/Rougheye/Shortraker', 'Shortraker/ Rougheye Rockfish Complex','Blackspotted/Rougheye/Shortraker_Unidentified_Discard',
                                                 'Blackspotted/Rougheye/Shortraker Unidentified_1_Discard','Blackspotted/Rougheye/Shortraker Unidentified_Discard','Blackspotted/Rougheye/Shortraker Unidentified_2_Discard',
                                                 'Blackspotted/Rougheye/Shortraker Unidentified_3_Discard','Shortraker/Rougheye Rockfish Complex','Shortraker/ Rougheye Complex',
                                                 'Shortraker/ Rougheye complex','Shortraker/Rougheye Complex','Blackspotted/ Rougheye/ Shortraker',
                                                 'Blackspotted/Rougheye/Shortraker Unidentified','Blackspotted/Rougheye/Shortraker Rockfish','Shortraker/Rougheye/Blackspotted Rockfish'],
    'Silvergray Rockfish':[],
    'Canary Rockfish':[],
    'Yelloweye Rockfish':[],
    'Northern Rockfish':[],
    'Quillback Rockfish':[],


    'Invertebrates': ['Invertabrate unidentified', 'nonfish_unid','Marine Invertabrate Unidentified'],
    'Anemones': ['Anemone', 'Sea Anemone','Sea amemones'],
    'Octopus':[],
    'Coral':['Coral on Rock', 'CoralUNID','Sea Whip','Sea Pens and Sea Whips','Hydrocorals','Coral on Rock', 'CoralUNID','Corals','Seaweed'],
    'Starfish':['StarfishUNID', 'Basket Stars','Starfish Unidentified','Sea Stars','Sea Star','Sea Stars_Discard'],
    'Bivalvia':[],
    'Snails':['Snail'],
    'Sea Urchins':[],
    'Sponges':['Sponge'],
    'Mollusca':[],


    'Bird': ['Bird_Bird on Water','Bird_Bird in Air','Bird_Bird on Water','Shearwater','Northern Fulmar',
             'Black-footed Albatross','Black-footed Albatross_Bird On Water','Black-footed Albatross_Bird on Water',
             'Black-footed Albatross-Bird on Water','Blackfooted Albatross_Bird on Water','Laysan Albatross',
             'Common Murre','King Eider','Large Immature Gull','Gull unidentified','Black-legged Kittiwake',
             'Fork-tailed Storm-petrel','Leach\'s Storm-petrel','Tufted Puffin','Parakeet Auklet',
             'Cassin\'s Auklet','Emperor Goose','Crested Auklet','Whiskered Auklet','Northern Fulmar_Bird on Water',
             'Northern Fulmar_Bird on Water','Gulls_Bird on Water','Gull Unidentified_Bird On Water','Gull Unidentified_Bird In Air',
             'Gulls_Bird in Air','Northern Fulmar_Bird On Water','Northern Fulmar_BirdOnWater','Northen Fulmar_Bird on Water','Northern Fulmar_Bird in Air',
             'Northern Fulmar _Bird on Water','Northern Fulmar_Bird On water','Norther Fulmar_Bird on Water',
             'Gull_Bird in Air','Gull_Bird on Water'],


    'Pacific Sleeper Sharks':['Sleeper sharks', 'Pacific Sleeper Sharks_Discard', 'Pacific Sleeper Shark'],



    # 不是目前关注的对象，而且类别有点杂
    'Fishing Gear': ['Anchor','Gangion','Hook','Mag Knot','Pole Gaff','Snap Clip','Weight','Fishing Gear_other', 'Chain',
                     'Weights', 'Fishing Gear_Line','Weight_ChainLink','Quick Link','Glove','Lead Weight','Buoy','Buoys','Flag',
                     'Hook and Gangion','Line'
                     ],

    # 不是目前关注的对象，而且类别有点杂
    'Bait': ['Bait_Other','Bait_Discard','Bait_Cod','Bait_Combo','Bait_Herring','Bait_Octopus','Bait_Salmon','Bait_Sardines','Bait_Squid','Bait_discard'],


    #  类别太杂了
    'Misc_Other': ['Misc Other','Miscellaneous','Marine Debris']
}




class Class_alias:
    def __init__(self, filename = ''):
        if filename == '':
            self.alias = hierarchy_alias
        else:
            # read from file (not implemented yet)
            raise NotImplementedError

    def readAlias(self, filename):
        raise NotImplementedError

    def convertName(self, input):
        input = input.strip()
        if input[-8:].lower().replace('_', ' ') == ' discard':
            input = input[:-8]
        input = input.strip()
        if input == '':
            return 'null'

        Found_alias=False
        for key, value in self.alias.items():
            if key.lower().replace("_", " ") == input.lower().replace("_", " "): # case and underline insensitive
                Found_alias=True
                return key, Found_alias
            for a in value:
                if a.lower().replace("_", " ") == input.lower().replace("_", " "): # case and underline insensitive
                    Found_alias= True
                    return key, Found_alias

        return input, Found_alias