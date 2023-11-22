from __future__ import print_function, absolute_import
import numpy as np
"""Cross-Modality ReID"""
import pdb


import numpy as np
import torch

import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def eval_vision_H(distmat, q_pids, g_pids,queryset,gallset,picure,trial, max_rank = 20):
    H=392
    W=196
    #distmat.shape:(3803, 301)
    #q_pids.shape:(3803,):q_camids
    #g_pids.shape:(301,):g_camids
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    #indices:(3803, 301),argsort
    indices = np.argsort(distmat, axis=1)
    #pred_label.shape:(3803, 301)
    #g_pids.shape:(301,)
    pred_label = g_pids[indices]
    #matches:(3803, 301)，q->g
    #q_pids[:, np.newaxis].shape:(3803, 1)  #########  g_pids[indices].shape:(3803, 301)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

###################################################################################
    mm=matches
    nn=indices
    # mm=np.flip(mm, axis=1)
    # nn=np.flip(nn, axis=1)
    rank=10
    

    for i in range(0,3803,30):
        begin=i
        end=i+30
        if end>3803:
            end=3803
        q_index = range(begin, end)
        fig_per_num = len(q_index)
        g_index = nn[q_index]
        match = mm[q_index]
        fig = plt.figure(constrained_layout=True, figsize=(22,4 * fig_per_num))
        #fig.suptitle("VISISUL")
        subfigs = fig.subfigures(nrows=fig_per_num, ncols=1)
        for row in range(fig_per_num):
            #grandtrues##########################
            #subfigs[row].suptitle("")
            axs = subfigs[row].subplots(nrows=1, ncols=rank + 2)
            image = Image.fromarray(queryset.test_image[q_index[row]])
            image = ImageOps.expand(image, border=10, fill='blue').resize((W, H))
            imageid=queryset.test_label[q_index[row]]
            axs[1].imshow(image)
            axs[1].axis('off')
            axs[0].text(0.38,0.45,str(imageid),fontsize=50)
            axs[0].axis('off')
            #last ten pit
            for col in range(1+1, rank + 1+1):
                image=Image.fromarray(gallset.test_image[g_index[row, col - 1]] )
                if match[row, col - 1]:
                    image = ImageOps.expand(image, border=10, fill='green').resize((W, H))
                else: image = image.resize((W, H))
                axs[col].imshow(image)
                axs[col].axis('off')
        if not os.path.isdir('/home/gml/HXC/.eval/'+str(begin)+'_'+str(end)):
            os.makedirs('/home/gml/HXC/.eval/'+str(begin)+'_'+str(end))

        output_dir = '/home/gml/HXC/.eval/'+str(begin)+'_'+str(end)+'/tmp'+str(trial)+'.png'
        fig.savefig(output_dir)
        print("saved at: " + output_dir)
        if end==3803:
            break
####################################################################################