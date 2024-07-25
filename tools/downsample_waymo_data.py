import os 
import numpy as np 

from sklearn.cluster import KMeans
from sklearn import cluster
import cv2
from torch.nn import functional as F
import torch
import pdb
import os.path as osp
from tqdm import tqdm 
from multiprocessing import Pool
from functools import partial
import concurrent.futures as futures
import argparse

os.environ["OPENBLAS_MAIN_FREE"] = "1"

def compute_angles(pc_np):
    tan_theta = pc_np[:, 2] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    theta = np.arctan(tan_theta)
    theta = (theta / np.pi) * 180

    sin_phi = pc_np[:, 1] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi_ = np.arcsin(sin_phi)
    phi_ = (phi_ / np.pi) * 180

    cos_phi = pc_np[:, 0] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi = np.arccos(cos_phi)
    phi = (phi / np.pi) * 180

    phi[phi_ < 0] = 360 - phi[phi_ < 0]
    phi[phi == 360] = 0

    return theta, phi

def beam_label(theta, beam):
    estimator=KMeans(n_clusters=beam)
    res=estimator.fit_predict(theta.reshape(-1, 1))
    label=estimator.labels_
    centroids=estimator.cluster_centers_
    return label, centroids[:,0]


def generate_mask(phi, beam, label, idxs, beam_ratio, bin_ratio):
    mask = np.zeros((phi.shape[0])).astype(np.bool)

    for i in range(0, beam, beam_ratio):
        phi_i = phi[label == idxs[i]]
        idxs_phi = np.argsort(phi_i)
        mask_i = (label == idxs[i])
        mask_temp = np.zeros((phi_i.shape[0])).astype(np.bool)
        mask_temp[idxs_phi[::bin_ratio]] = True
        mask[mask_i] = mask_temp

    return mask

def downsample(inputpath, savepath):
    beam = 64
   
    # filepath = osp.join(inputpath, file)
    # savepath = osp.join(savepath, file)

    filepath = inputpath 
    savepath = savepath 

    points = np.load(filepath)
    pc_np = points[:, :3]
    theta, phi = compute_angles(pc_np)

    label, centroids = beam_label(theta, beam)

    idxs = np.argsort(centroids)

    mask = generate_mask(phi, beam, label, idxs, beam_ratio=2, bin_ratio=2)
    save_downsample = points[mask]
    # save_path = self.root_split_path / 'modes' / '32' / ('%s.bin' % sample_idx)
     
    np.save(savepath, save_downsample)
    # save_downsample.tofile(save_path)

# mask = generate_mask(phi, beam, label, idxs, beam_ratio=2, bin_ratio=2)
# save_downsample = points[mask]
# save_path = self.root_split_path / 'modes' / '32^' / ('%s.bin' % sample_idx)
# save_downsample.tofile(save_path)

def process_single_sequence(folder, input_folder, output_folder): 
    ipath = osp.join(input_folder, folder)
    opath = osp.join(output_folder, folder)
    if not osp.exists(opath): 
            os.makedirs(opath)
    files = os.listdir(ipath)
    for file in files: 
        input_file = osp.join(ipath, file)
        output_file = osp.join(opath, file)
        if input_file[-3:] != 'npy': 
            continue    
        downsample(file, input_file, output_file)

def create_infos(input_folder, output_folder, intlist):
    folders = os.listdir(input_folder)[intlist[0]:intlist[1]]

    
    for folder in tqdm(folders): 
        ipath = osp.join(input_folder, folder)
        opath = osp.join(output_folder, folder)
        if not osp.exists(opath): 
            os.makedirs(opath)
        files = os.listdir(ipath)
     
        # files_npy = [i for i in files if i.endswith('.npy')]
        # downsampling = partial(
        #     downsample,
        #     inputpath=ipath,
        #     savepath=opath
        # )
    #     with Pool(16) as p: 
    #         infos = list(tqdm(p.imap(downsampling, files_npy),
    #                                    total=len(files_npy)))

        for file in files: 
            input_file = osp.join(ipath, file)
            output_file = osp.join(opath, file)
            if input_file[-3:] != 'npy': 
                continue
            
            downsample(input_file, output_file)

def list_of_ints(arg):
    return list(map(int, arg.split(',')))                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--int-list', type=list_of_ints)
    args = parser.parse_args()
    int_list = args.int_list
    input_folder = '../data/waymo/waymo_processed_data_v0_5_0'
    output_folder = '../data/waymo/waymo_32^'
    if not osp.exists(output_folder): 
        os.makedirs(output_folder)
    create_infos(input_folder, output_folder, int_list)
    
