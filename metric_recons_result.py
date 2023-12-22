from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree
import numpy as np
import os
import glob
import pandas as pd
import h5py

from xml_reader import parseXml
import trimesh
from trimesh.voxel import creation
from trimesh import Trimesh

from kbr_slice import read_kbr_plane, read_kbr_mesh

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
    matrix = matrix*1

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


def metric_recons_hd(pred_points, gt_points):
    '''
    Args:
        pred_points, gt_points: np arrays
    Returns:
          hausdorff distance
    '''

    hausdorff_d = max(directed_hausdorff(pred_points, gt_points)[0], directed_hausdorff(gt_points, pred_points)[0])

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
    # points_a = np.random.rand(10,3)
    # points_b = np.random.rand(20,3)

    # hd = metric_recons_hd(points_a, points_b)
    # cd = metric_recons_cd(points_a, points_b) 

    # print(hd)
    # print(cd)
    
    # Metrics ORex pred&gt
    
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

                    hd = metric_recons_hd(pred_points, gt_points)

                    cd = metric_recons_cd(pred_points, gt_points)

                    iou = metric_recons_IOU(preds_matrix, gt_matrix)

                    dice = metric_recons_DICE(preds_matrix, gt_matrix)
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

    print(len(hausdorff_dis))
    print(hausdorff_dis[0])
    print(len(chamfer_dis))
    print(chamfer_dis[0])
    print(len(iou_3d))
    print(iou_3d[0])
    print(len(dice_3d))
    print(dice_3d[0])

    csv_dict = {'sid_mode':sid_list, 'pred_obj':pred_filelist, 'gt_xml':gt_filelist, 
                'hausdorff_dis':hausdorff_dis, 'chamfer_dis':chamfer_dis, 'iou_3d':iou_3d, 'dice_3d':dice_3d}
    df = pd.DataFrame(csv_dict)
    df.to_csv('/staff/ydli/projects/OReX/output/directory/kbr/metrics.csv')