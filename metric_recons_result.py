from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree
import numpy as np
import os
import torch
import glob
import pandas as pd
import h5py

from xml_reader import parseXml
import trimesh
from trimesh.voxel import creation
from trimesh import Trimesh

from kbr_slice import read_kbr_plane, read_kbr_mesh

from skimage.metrics import hausdorff_distance
from utils_seg import cal_score, cal_asd

from skimage import measure
import copy
import SimpleITK as sitk

def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=int)
    hdf5_file.close()

    return image

def read_obj_file(pred_file):
    obj = trimesh.load(pred_file)

    # print(type(obj))
    verts = obj.vertices
    # faces = obj.faces

    voxels = creation.local_voxelize(obj, obj.centroid, pitch=0.005, radius=128, fill=True)
    matrix = voxels.matrix
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 2)
    # print(matrix.shape)

    return verts, matrix

def read_xml_file(gt_name, mode, sid):

    info_list, cali_info, mesh_text = parseXml(gt_name)
    verts, faces = read_kbr_mesh(mesh_text[mode])
    verts -= np.mean(verts, axis=0)
    scale = 1.1
    verts /= scale * np.max(np.absolute(verts))

    hdf5_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5/'
    hdf5_file = glob.glob(hdf5_path+'/*'+sid+'.hdf5')[0]
    # print(hdf5_file)
    matrix = hdf5_reader(hdf5_file,'image')

    return verts, matrix


def metric_recons_hd(pred_matrix, gt_matrix):
    '''
    Args:
        pred_points, gt_points: np arrays
    Returns:
          hausdorff distance
    '''

    hausdorff_d = hausdorff_distance(pred_matrix, gt_matrix)

    return hausdorff_d

def metric_recons_cd(pred_points, gt_points):
    '''
    Args:
        pred_points, gt_points: np arrays
    Returns:
        chamfer distance
    '''

    tree = KDTree(gt_points)
    dist_pred_points = tree.query(pred_points)[0]
    tree = KDTree(pred_points)
    dist_gt_points = tree.query(gt_points)[0]

    return np.mean(dist_pred_points) + np.mean(dist_gt_points)

def metric_recons_IOU(preds_matrix, gt_matrix):
    '''
    Args:
		preds (np.array) voxels
		gt (np,array) voxels

	Returns:
		float: IoU
    '''
    intersec = np.logical_and(preds_matrix, gt_matrix)
    union = np.logical_or(preds_matrix, gt_matrix)
    voxels_IOU = np.sum(intersec) / np.sum(union)

    return voxels_IOU


def metric_recons_DICE(preds_matrix, gt_matrix):
    '''
    Args:
		preds (np.array) voxels
		gt (np,array) voxels

	Returns:
		float: DICE
    '''
    count_pred_true = np.sum(preds_matrix)
    count_gt_true = np.sum(gt_matrix)
    intersec = np.logical_and(preds_matrix, gt_matrix)
    voxels_DICE = 2*np.sum(intersec) / (count_pred_true+count_gt_true)

    return voxels_DICE

if __name__ == '__main__':
    
    # ## Test

    # pred_filepath = '/staff/ydli/projects/OReX/output/directory/kbr'
    # gt_filepath = '/staff/ydli/projects/OReX/Data/kbr_patient_backup'
    # mode_list = ['ed', 'es']

    # for mode in mode_list: 
    #     for pred_name in os.listdir(pred_filepath):
    #         if mode in pred_name:
    #             # print(pred_name)

    #             sid = pred_name.split('_')[-1]
    #             sid = 'SID' + '_' + pred_name.split('_')[-2] + '_' + sid  
    #             # print(sid)
    #             sid_mode = sid+'_'+mode
    #             # sid_list.append(sid_mode)
    #             pred_file = pred_filepath+'/'+pred_name+'/mesh_last_300.obj'
    #             pred_points, preds_matrix = read_obj_file(pred_file)

    #             gt_file = glob.glob(gt_filepath+'/*'+sid+'.xml')[0]
    #             # print(gt_name)
    #             gt_points, gt_matrix = read_xml_file(gt_file, mode, sid_mode)

    #             hd = metric_recons_hd(preds_matrix, gt_matrix)

    #             cd = metric_recons_cd(pred_points, gt_points)

    #             iou = metric_recons_IOU(preds_matrix, gt_matrix)

    #             dice = metric_recons_DICE(preds_matrix, gt_matrix)
                
    #             predict = copy.deepcopy(preds_matrix*1)
    #             target = copy.deepcopy(gt_matrix)
    #             predict = sitk.GetImageFromArray(predict)
    #             target = sitk.GetImageFromArray(target)
    #             predict = sitk.Cast(predict,sitk.sitkUInt8)
    #             target = sitk.Cast(target,sitk.sitkUInt8)   

    #             roi_pred = (torch.from_numpy(preds_matrix)==1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0) # 11HWD
    #             roi_true = (torch.from_numpy(gt_matrix)==1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0)
    #             # print(roi_pred.size())
    #             # print(roi_true.size())

    #             # preds_matrix = preds_matrix==1
    #             # gt_matrix = gt_matrix==1

    #             # print(type(preds_matrix))
    #             # print(preds_matrix.shape)
    #             # print(type(gt_matrix))
    #             # print(gt_matrix.shape)

    #             result = cal_score(predict, target)
    #             asd = cal_asd(roi_pred,roi_true)

    #         break
    #     break

    # print(result)
    # print(asd)
    # print(hd)
    # print(cd)
    
    ## Metrics ORex pred&gt
    
    pred_filepath = '/staff/ydli/projects/OReX/output/directory/kbr'
    gt_filepath = '/staff/ydli/projects/OReX/Data/kbr_patient_backup'
    mode_list = ['ed', 'es']

    sid_list = []
    pred_filelist = []
    gt_filelist = []

    hausdorff_dis = []
    chamfer_dis = []
    iou_3d = []
    dice_3d = []

    asd_list = []
    jaccard = []
    volume_sim = []
    false_negative = []
    false_positive = []


    for mode in mode_list: 
        for pred_name in os.listdir(pred_filepath):
            if mode in pred_name:
                # print(pred_name)

                sid = pred_name.split('_')[-1]
                sid = 'SID' + '_' + pred_name.split('_')[-2] + '_' + sid  
                # print(sid)
                sid_mode = sid+'_'+mode
                # sid_list.append(sid_mode)
                try:
                    pred_file = pred_filepath+'/'+pred_name+'/mesh_last_300.obj'
                    pred_points, preds_matrix = read_obj_file(pred_file)

                    gt_file = glob.glob(gt_filepath+'/*'+sid+'.xml')[0]
                    # print(gt_name)
                    gt_points, gt_matrix = read_xml_file(gt_file, mode, sid_mode)

                    hd = metric_recons_hd(preds_matrix, gt_matrix)

                    cd = metric_recons_cd(pred_points, gt_points)

                    iou = metric_recons_IOU(preds_matrix, gt_matrix)

                    dice = metric_recons_DICE(preds_matrix, gt_matrix)
                    
                    predict = copy.deepcopy(preds_matrix*1)
                    target = copy.deepcopy(gt_matrix)
                    predict = sitk.GetImageFromArray(predict)
                    target = sitk.GetImageFromArray(target)
                    predict = sitk.Cast(predict,sitk.sitkUInt8)
                    target = sitk.Cast(target,sitk.sitkUInt8)   
                    roi_pred = (torch.from_numpy(preds_matrix)==1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0) # 11HWD
                    roi_true = (torch.from_numpy(gt_matrix)==1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0)
                    # preds_matrix = preds_matrix==1
                    # gt_matrix = gt_matrix==1

                    # print(type(preds_matrix))
                    # print(preds_matrix.shape)
                    # print(type(gt_matrix))
                    # print(gt_matrix.shape)

                    result = cal_score(predict, target)
                    asd = cal_asd(roi_pred,roi_true)

                    print('metrics '+sid_mode+' successfully!')
                except:
                    print('metrics '+sid_mode+' failed!')
                else:
                    sid_list.append(sid_mode)
                    pred_filelist.append(pred_file)
                    gt_filelist.append(gt_file)
                    hausdorff_dis.append(hd)
                    chamfer_dis.append(cd)
                    iou_3d.append(iou)
                    dice_3d.append(dice)
                    asd_list.append(asd)
                    jaccard.append(result['Jaccard'])
                    volume_sim.append(result['VolumeSimilarity'])
                    false_negative.append(result['FalseNegativeError'])
                    false_positive.append(result['FalsePositiveError'])

    print(len(hausdorff_dis))
    print(hausdorff_dis[0])
    print(len(chamfer_dis))
    print(chamfer_dis[0])
    print(len(iou_3d))
    print(iou_3d[0])
    print(len(dice_3d))
    print(dice_3d[0])
    print(len(jaccard))
    print(jaccard[0])
    print(len(volume_sim))
    print(volume_sim[0])
    print(len(false_negative))
    print(false_negative[0])
    print(len(false_positive))
    print(false_positive[0])
    print(len(asd_list))
    print(asd_list[0])

    csv_dict = {'sid_mode':sid_list, 'pred_obj':pred_filelist, 'gt_xml':gt_filelist, 
                'hausdorff_dis':hausdorff_dis, 'chamfer_dis':chamfer_dis, 'iou_3d':iou_3d, 'dice_3d':dice_3d, 
                'jaccard':jaccard, 'VolumeSimilarity':volume_sim, 'FalseNegativeError':false_negative, 'FalsePositiveError':false_positive, 'asd':asd_list}
    df = pd.DataFrame(csv_dict)
    df.to_csv('/staff/ydli/projects/OReX/output/directory/kbr/metrics.csv')