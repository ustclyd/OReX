import numpy as np
import os
import pandas as pd
import pickle
import glob
import random
import h5py

import pyvista as pv
import open3d as o3d

from xml_reader import parseXml
import trimesh
from trimesh.voxel import creation
from trimesh import Trimesh
from Dataset.CSL import CSL

from scipy.spatial import ConvexHull

from kbr_mesh import KBRMesh
from kbr_slice import read_kbr_plane, read_kbr_mesh



def make_kbr_gt_mesh2voxels(filename, mode):
    '''
    read kbr mesh ground truth from a xml file, mode record 'ed', 'es'
    then output a matrix which is the dense matrix voxels of this mesh (N*N*N) filled with True, False
    '''
    
    info_list, cali_info, mesh_text = parseXml(filename)
    verts, faces = read_kbr_mesh(mesh_text[mode])
    verts -= np.mean(verts, axis=0)
    scale = 1.1
    verts /= scale * np.max(np.absolute(verts))
    # print(verts)
    mesh = Trimesh(verts, faces)
    voxels = creation.local_voxelize(mesh, mesh.centroid, pitch=0.005, radius=128, fill=True)
    # print(voxels)
    # voxels.show()
    matrix = voxels.matrix
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 2)
    # print(matrix.shape)

    # output matrix with shape:(512, 512, 512) filled with True&False

    return matrix

def make_kbr_train_mesh2voxels(filename):
    '''
    read kbr plane mesh(train data) from ply,
    create a convex mesh for this mesh,
    then output a np.array (N, 3) which represents the voxels of this mesh
    '''
    
    # pcd = o3d.io.read_point_cloud(filename)
    # hull, _ = pcd.compute_convex_hull()
    # # print(type(hull))
    # # hull.scale(1 / np.max(hull.get_max_bound() - hull.get_min_bound()),
    # #        center=hull.get_center())
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(hull,
    #                                                           voxel_size=0.005)
    # voxels = voxel_grid.get_voxels()
    # indices = np.stack(list(vx.grid_index for vx in voxels)) # 
    # # print(indices)
    # # o3d.visualization.draw_geometries([voxel_grid])

    pcd = o3d.io.read_point_cloud(filename)
    pcd.estimate_normals()

    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 1.5 * avg_dist   

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd,
            1)

    # print(np.asarray(mesh.vertices))

    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                          vertex_normals=np.asarray(mesh.vertex_normals))
    
    convex_hull = trimesh.convex.convex_hull(tri_mesh, qhull_options='QbB Pp Qt', repair=True)
    voxels = creation.local_voxelize(convex_hull, convex_hull.centroid, pitch=0.005, radius=128, fill=True)
    # voxels.show()
    matrix = voxels.matrix
    matrix = np.delete(matrix, 0, 0)
    # print(matrix.shape)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 2)

    # output matrix with shape:(512, 512, 512) filled with True&False

    return matrix




def make_gt_pickle_txt_file(dump_filepath, xml_file, mode):

    txt_file = open(dump_filepath, 'wb')
    matrix = make_kbr_gt_mesh2voxels(xml_file, mode)
    pickle.dump(matrix, txt_file)
    txt_file.close()

    return

def make_train_pickle_txt_file(dump_filepath, ply_file):

    txt_file = open(dump_filepath, 'wb')
    matrix = make_kbr_train_mesh2voxels(ply_file)
    pickle.dump(matrix, txt_file)
    txt_file.close()

    return


def make_txt_file_for_UNet(gt_filepath, train_filepath, mode_list, data_path):

    # create data path
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print("UNet Folder created")
    else:
        print("UNet Folder already exists")
    
    # create es&ed data path
    for mode in mode_list:
        mode_data_path = data_path + '/' + mode + '_data'
        if not os.path.exists(mode_data_path):
            os.makedirs(mode_data_path)
            print('UNet '+mode+ '_data Folder created')
        else:
            print('UNet '+mode+ '_data Folder already exists')

    # print(len(os.listdir(gt_filepath)))
    for filename in os.listdir(gt_filepath):

        xml_file = gt_filepath+'/'+filename

        # filename is like 'VpStudy_SID_3042_10280.xml'
        sid = filename.split('_')[-1].split('.')[0]
        # print(sid) # sid = '10280'
        sid = 'SID' + '_' + filename.split('_')[-2] + '_' + sid   
        # print(sid) # sid = 'SID_3042_10280'

        for mode in mode_list: # mode 'es'&'ed'
            mode_data_path = data_path + '/' + mode + '_data'
            sid_data_path = mode_data_path + '/' + sid
            if not os.path.exists(sid_data_path):
                os.makedirs(sid_data_path)
                print(sid+'_'+mode+' Folder created')
            else:
                print(sid+'_'+mode+' Folder already exists')

            sid_gt_data_path = mode_data_path + '/' + sid + '/' + 'gt'
            if not os.path.exists(sid_gt_data_path):
                os.makedirs(sid_gt_data_path)
                print(sid+'_'+mode+' gt Folder created')
            else:
                print(sid+'_'+mode+' gt Folder already exists')

            sid_train_data_path = mode_data_path + '/' + sid + '/' + 'train'
            if not os.path.exists(sid_train_data_path):
                os.makedirs(sid_train_data_path)
                print(sid+'_'+mode+' train Folder created')
            else:
                print(sid+'_'+mode+' train Folder already exists')

            # make es&ed gt data .txt file
            try:

                dump_filepath = sid_gt_data_path+ '/' + sid+'_'+mode+'_gt_data.txt'
                make_gt_pickle_txt_file(dump_filepath, xml_file, mode)
                print(sid+'_'+mode+' gt txt file created')
            except:
                print(' error: '+sid+'_'+mode+' gt txt file not created')


            id_file = train_filepath +'/*' +sid+'.ply'
            # len_file=len(glob.glob(id_file))
            # print(len_file)
            for file in glob.glob(id_file):
                if mode in file:
                    ply_file = file
                    try:
                        if len(os.listdir(sid_gt_data_path)) != 0:
                            dump_filepath = sid_train_data_path+ '/' + sid+'_'+mode+'_train_data.txt'
                            make_train_pickle_txt_file(dump_filepath, ply_file)
                            print(sid+'_'+mode+' train txt file created')
                            
                        else:
                            print(' error: '+sid+'_'+mode+' gt txt file does not exist')
                    except:
                        print(' error: '+sid+'_'+mode+' train txt file not created')
    
    return



def make_csv_file_for_UNet(mode_list, data_path):

    # make the .csv file
    sid_list = []
    train_filelist = []
    gt_filelist = []
    split_list = []
    ratio = 0.25

    for mode in mode_list:
        mode_data_path = data_path + '/' + mode + '_data'
        for sid in os.listdir(mode_data_path):
            # print(sid)
            train_path = mode_data_path + '/' + sid + '/' + 'train'
            gt_path = mode_data_path + '/' + sid + '/' + 'gt'

            if len(os.listdir(train_path)) != 0:

                sid_mode = sid + '_' + mode
                sid_list.append(sid_mode)

                train_filepath = train_path + '/' + sid+'_'+mode+'_train_data.txt'
                train_filelist.append(train_filepath)

                gt_filepath = gt_path + '/' + sid+'_'+mode+'_gt_data.txt'
                gt_filelist.append(gt_filepath)

                if random.uniform(0, 1) > ratio :

                    split_list.append('training')
                else:
                    split_list.append('validation')

    # print(sid_list[0])
    print(len(sid_list))
    # print(train_filelist[0])
    # print(len(train_filelist))
    # print(gt_filelist[0])
    # print(len(gt_filelist))
    # print(split_list)
    csv_dict = {'data_name':sid_list, 'train_filepath':train_filelist, 'gt_filepath':gt_filelist, 'split':split_list}
    df = pd.DataFrame(csv_dict)
    
    # save dataframe as .csv file
    df.to_csv('/staff/ydli/projects/OReX/Data/UNet/UNet_data.csv')

def read_pickle_saveas_hdf5(mode_list, data_path):

    # make the .csv file
    # sid_list = []
    # train_filelist = []
    # gt_filelist = []
    # split_list = []
    # ratio = 0.25
    unet_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5'

    for mode in mode_list:
        sid_list = []
        train_filelist = []
        gt_filelist = []
        mode_data_path = data_path + '/' + mode + '_data'
        for sid in os.listdir(mode_data_path):
            # print(sid)
            train_path = mode_data_path + '/' + sid + '/' + 'train'
            gt_path = mode_data_path + '/' + sid + '/' + 'gt'

            if len(os.listdir(train_path)) != 0:

                sid_mode = sid + '_' + mode
                sid_list.append(sid_mode)

                train_filepath = train_path + '/' + sid+'_'+mode+'_train_data.txt'
                train_filelist.append(train_filepath)

                gt_filepath = gt_path + '/' + sid+'_'+mode+'_gt_data.txt'
                gt_filelist.append(gt_filepath)

                # if random.uniform(0, 1) > ratio :

                #     split_list.append('training')
                # else:
                #     split_list.append('validation')
        # print(sid_list[0])
        length = len(sid_list)
        # print(train_filelist)
        for index in range(length):
            # print(index)
            sid  = sid_list[index]
            train_file = train_filelist[index]
            # print(train_file)
            gt_file = gt_filelist[index]

            train_pickle_file = open(train_file,'rb')
            image = pickle.load(train_pickle_file)
            train_pickle_file.close()

            gt_pickle_file = open(gt_file,'rb')
            label = pickle.load(gt_pickle_file)
            gt_pickle_file.close()

            hdf5_path = unet_path+'/'+sid+'.hdf5'
            hdf5_file = h5py.File(hdf5_path, 'w')
            hdf5_file.create_dataset('image', data=image.astype(np.int16))
            hdf5_file.create_dataset('label', data=label.astype(np .uint8))
            hdf5_file.close()
            print('create '+sid+' hdf5 file success!')
 
    # print(sid_list[0])
    # print(len(sid_list))



if __name__ == '__main__':
    
    # ## Test count the number of voxels in our voxel data

    # filename = '/staff/ydli/projects/OReX/Data/kbr_patient_backup/VpStudy_SID_3042_10280.xml' # '/staff/ydli/projects/OReX/trash/kbr_ed_heart_SID_3042_10280.ply' # '/staff/ydli/projects/OReX/Data/kbr_patient_backup/VpStudy_SID_3042_10280.xml'
    
    
    # matrix = make_kbr_gt_mesh2voxels(filename, 'es')
    # print(matrix.shape)
    # count_true = np.sum(matrix) / 134217728
    # print(count_true)

    # ## Test glob

    # train_filepath = '/staff/ydli/projects/OReX/Data/kbr_backup'
    # sid = 'SID_3042_10280'

    # id_file = train_filepath +'/*' +sid+'.ply'
    # # len_file=len(glob.glob(id_file))
    # # print(len_file)
    # for file in glob.glob(id_file):
    #     print(file)




    ## Run this line to make the .csv file which record the .txt file(voxel file)'s path

    gt_filepath = '/staff/ydli/projects/OReX/Data/kbr_patient_backup'
    train_filepath = '/staff/ydli/projects/OReX/Data/kbr_backup'
    mode_list = ['es', 'ed']

    # create UNet data path
    data_path = '/staff/ydli/projects/OReX/Data/UNet'

    # # make the .txt file
    # make_txt_file_for_UNet(gt_filepath, train_filepath, mode_list, data_path)


    # make the .csv file
    # print(len(os.listdir(gt_filepath)))
    # print(os.listdir(gt_filepath))
    read_pickle_saveas_hdf5(mode_list, data_path)
