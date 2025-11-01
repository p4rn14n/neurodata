# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:39:13 2025

@author: MurphyLab
"""

############### 1ROI data
dic = {
    '505970m3': {'target': (-0.02, 0.08),
                 'seeds': (-0.04, 0.12),
                 'maps': (-0.04, 0.04)},
    '505970m5': {'target': (-0.02, 0.12),
                 'seeds': (-0.04, 0.14),
                 'maps': (-0.1, 0.1)},
    '506554m1': {'target': (-0.08, 0.08),
                 'seeds': (-0.05, 0.17),
                 'maps': (-0.1, 0.1)},
    '506554m3': {'target' : (-0.04, 0.12),
                 'seeds': (-0.05, 0.26),
                 'maps': (-0.12, 0.12)}
}

expt1_data_list = []
# expt1_data_list.append(['506554m1', 'male', [
#     [1, '20250516114407'], # 60tr no-audio random-reward
#     [2, '20250517180237'], # 60tr no-audio random-reward
#     [3, '20250518162012'], # 60tr no-audio random-reward
#     [4, '20250519161048'], # 60tr no-audio random-reward
#     [5, '20250520152233'], # 60tr no-audio random-reward
#     [6, '20250521141936'], # 60tr RFLL-CFLL @0.07
#     [7, '20250522143242'], # 60tr RFLL-CFLL @0.07
#     [8, '20250523141233'], # 60tr RFLL-CFLL @0.07
#     [10, '20250525162719'], # 60tr RFLL-CFLL @0.07
#     [11, '20250526123718'], # 60tr no-audio random-reward
#     [12, '20250527111858'], # 60tr no-audio random-reward
#     [13, '20250528124426'], # 60tr no-audio random-reward
#     [14, '20250529144436'], # 60tr no-audio random-reward
#     [15, '20250530120739'], # 60tr no-audio random-reward
#     [16, '20250531154629'], # 60tr RFLL-CFLL @0.07
#     [18, '20250602134033'], # 60tr RFLL-CFLL @0.07
#     [19, '20250603145046'], # 60tr RFLL-CFLL @0.07
#     # [21, '20250605120449'], # 60tr RFLL-CFLL @0.07
#                           ], '2ROI', 'grp1', 'exp1'])


expt1_data_list.append(['505970m3', 'male', [
    [1, '20250516135106'], # 60tr no-audio random-reward
    [2, '20250517190930'], # 60tr no-audio random-reward
    [3, '20250518180737'], # 60tr no-audio random-reward
    [4, '20250519174810'], # 60tr no-audio random-reward
    [5, '20250520173410'], # 60tr no-audio random-reward
    [6, '20250521163059'], # 60tr RFLL-CFLL @0.07
    [7, '20250522155959'], # 60tr RFLL-CFLL @0.07
    [8, '20250523161538'], # 60tr RFLL-CFLL @0.07
    [9, '20250525174300'], # 60tr RFLL-CFLL @0.07
    [10, '20250526143521'], # 60tr no-audio random-reward
    [11, '20250527130352'], # 60tr no-audio random-reward
    [12, '20250528142855'], # 60tr no-audio random-reward
    [13, '20250529162208'], # 60tr no-audio random-reward
    [14, '20250530132755'], # 60tr no-audio random-reward
    [15, '20250531170236'], # 60tr RFLL-CFLL @0.07
    [16, '20250602151749'], # 60tr RFLL-CFLL @0.07
    # [19, '20250603154929'] # 60tr RFLL-CFLL @0.07
    # [21, '20250605134951'], # 60tr RFLL-CFLL @0.07
                          ], '2ROI'])

expt1_data_list.append(['506554m3', 'male', [
    [1, '20250516125850'], # 60tr no-audio random-reward
    [2, '20250517183212'], # 60tr no-audio random-reward
    [3, '20250518165954'], # 60tr no-audio random-reward
    [4, '20250519165910'], # 60tr no-audio random-reward
    [5, '20250520161240'], # 60tr no-audio random-reward
    [6, '20250521150951'], # 60tr RFLL-CFLL @0.07
    [7, '20250522152304'], # 60tr RFLL-CFLL @0.07
    [8, '20250523151222'], # 60tr RFLL-CFLL @0.07
    [9, '20250525170958'], # 60tr RFLL-CFLL @0.07
    [10, '20250526132638'], # 60tr no-audio random-reward
    [11, '20250527120518'], # 60tr no-audio random-reward
    [12, '20250528133151'], # 60tr no-audio random-reward
    [13, '20250529152657'], # 60tr no-audio random-reward
    [14, '20250530125259'], # 60tr no-audio random-reward
    [15, '20250531163045'], # 60tr RFLL-CFLL @0.07/
    [16, '20250602142243'], # 60tr RFLL-CFLL @0.07
    # [19, '20250603152054'] # 60tr RFLL-CFLL @0.07
    # [21, '20250605125323'], # 60tr RFLL-CFLL @0.07
                          ], '2ROI'])

expt1_data_list.append(['506554m1', 'male', [
    [1, '20250516114407'], # 60tr no-audio random-reward
    [2, '20250517180237'], # 60tr no-audio random-reward
    [3, '20250518162012'], # 60tr no-audio random-reward
    [4, '20250519161048'], # 60tr no-audio random-reward
    [5, '20250520152233'], # 60tr no-audio random-reward
    [6, '20250521141936'], # 60tr RFLL-CFLL @0.07
    [7, '20250522143242'], # 60tr RFLL-CFLL @0.07
    [8, '20250523141233'], # 60tr RFLL-CFLL @0.07
    [9, '20250525162719'], # 60tr RFLL-CFLL @0.07
    [10, '20250526123718'], # 60tr no-audio random-reward
    [11, '20250527111858'], # 60tr no-audio random-reward
    [12, '20250528124426'], # 60tr no-audio random-reward
    [13, '20250529144436'], # 60tr no-audio random-reward
    [14, '20250530120739'], # 60tr no-audio random-reward
    [15, '20250531154629'], # 60tr RFLL-CFLL @0.07
    [16, '20250602134033'], # 60tr RFLL-CFLL @0.07
    # [19, '20250603145046'] # 60tr RFLL-CFLL @0.07
    # [21, '20250605120449'] # 60tr RFLL-CFLL @0.07
                          ], '2ROI'])



expt1_data_list.append(['505970m5', 'male', [
    [1, '20250516152400'], # 60tr no-audio random-reward
    [2, '20250517194131'], # 60tr no-audio random-reward
    [3, '20250518185235'], # 60tr no-audio random-reward
    [4, '20250519184224'], # 60tr no-audio random-reward
    [5, '20250520183117'], # 60tr no-audio random-reward
    [6, '20250521173020'], # 60tr RFLL-CFLL @0.07
    [7, '20250522165959'], # 60tr RFLL-CFLL @0.07
    [8, '20250523171616'], # 60tr RFLL-CFLL @0.07
    [9, '20250525182809'], # 60tr RFLL-CFLL @0.07
    [10, '20250526153200'], # 60tr no-audio random-reward
    [11, '20250527135747'], # 60tr no-audio random-reward
    [12, '20250528152523'], # 60tr no-audio random-reward
    [13, '20250529170928'], # 60tr no-audio random-reward
    [14, '20250530141541'], # 60tr no-audio random-reward
    [15, '20250531175155'], # 60tr RFLL-CFLL @0.07
    [16, '20250602160149'], # 60tr RFLL-CFLL @0.07
    # [19, '20250603161604'] # 60tr RFLL-CFLL @0.07
    # [21, '20250605143108'] # 60tr RFLL-CFLL @0.07 
                          ], '2ROI'])
