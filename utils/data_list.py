from glob import glob
import numpy as np
import json

def main(test=False):
    # ----------------------------------------- modify the path relative to your machine
    data_path = '../TexturePaired'        
    # self.data_path = '/source/jihyeon/Sample-Refine2/TexturePaired/'
    # ----------------------------------------------------------------------------------

    if test:
        data_list = [
            [96, 'MGN_digital_wardrobe_96_2048*']\
        ]
    else:
        data_list = [
            [300, 'eNgine_300__512*'],
            [451, 'surreal_female_451__512*'],
            [478, 'surreal_male_478__512*'],
            [212,  'AIHUB_212__512*'],
            [741,  'renderpeople_741__512*'],
        ]
    
    interval = 10

    txtr_map  = {}
    mask      = {}
    part_map  = {}
    norm_map  = {}

    # i: 0,1,2,3,4
    # data: eNgine_300__512*, ...
    for i, (_, data) in enumerate(data_list): 
        norm_map[i] = sorted(glob(f'{data_path}/{data}/normal_obj/*.png'))
        txtr_map[i] = sorted(glob(f'{data_path}/{data}/texture/*.jpg')) \
                            + sorted(glob(f'{data_path}/{data}/texture/*.png'))
        mask[i]     = {}
        part_map[i] = {}  
        for a in np.arange(-90, 100, interval): # a : -90~90
            mask[i][int(a)] = sorted(glob(f'{data_path}/{data}/mask_symm/{a}/*.png'))
            part_map[i][int(a)] = sorted(glob(f'{data_path}/{data}/symmetry/{a}/*.png'))

    json_dict={'data_list':data_list,'txtr_map':txtr_map, 'mask':mask, 'part_map':part_map, 'norm_map':norm_map, 'interval':interval}
    
    if test:
        file_name = "data_list_test.json"
    else:
        file_name = "data_list_train.json"
        
    with open(file_name, "w") as file: 
        json.dump(json_dict, file, indent=4)

if __name__ == "__main__":
    main(True)
    main(False)