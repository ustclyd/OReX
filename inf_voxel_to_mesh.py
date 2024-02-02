import numpy as np
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import h5py
import open3d as o3d
import os
import glob
from tqdm import tqdm
from natsort import natsorted

from meshcut import cross_section

import trimesh
import pymesh

import pyvista as pv
import fast_simplification

from Dataset.Helpers import plane_origin_from_params
from kbr_slice import read_kbr_mesh, read_kbr_plane
from kbr_plane import make_plane_transform
from xml_reader import parseXml
from Slicer import _get_kbr_planes

def read_vpt(vpt_path):
    with open(vpt_path, 'rb') as vpt_file:
        data = np.frombuffer(vpt_file.read(), np.uint8)
    return data

def read_scan_vpt(scan_dir, width, height):
    if not os.path.exists(scan_dir):
        return None
    frame_path_list = os.listdir(scan_dir)
    frame_path_list = [os.path.join(scan_dir, p) for p in frame_path_list]
    frame_path_list = natsorted(frame_path_list)
    frame_list = [read_vpt(frame_path).reshape(height, width) for frame_path in frame_path_list]
    return np.array(frame_list)

def get_mesh_and_paras_from_xml(xml_file):
    info_list, cali_info, mesh_text = parseXml(xml_file)
    # print(info_list)
    gt_origin_mesh_verts, gt_origin_mesh_faces, = read_kbr_mesh(mesh_text[mode])
    plane_vector = np.mean(gt_origin_mesh_verts, axis=0)
    # plane_vector = np.zeros((3, ))
    gt_origin_mesh_verts -= plane_vector
    # print(plane_vector)
    plane_scale = 1.1
    plane_scale *= np.max(np.absolute(gt_origin_mesh_verts))
    # plane_scale = 1
    # # print(plane_scale)
    gt_origin_mesh_verts /= plane_scale
    # print(verts)
    gt_origin_mesh = trimesh.Trimesh(vertices=gt_origin_mesh_verts, faces=gt_origin_mesh_faces)

    plane_normals, ds, matrix_list = _get_kbr_planes(info_list, cali_info, plane_scale, plane_vector, mode)
    plane_origins = [plane_origin_from_params((*n, d)) for n, d in zip(plane_normals, ds)]

    return gt_origin_mesh, plane_normals, plane_origins, matrix_list, plane_scale, plane_vector

def get_mesh_from_voxel(voxel):
    verts, faces, _, _ = measure.marching_cubes(voxel, level=0, spacing=(0.016, 0.016, 0.016))

    voxel_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    v_vector = np.array([0.016*64, 0.016*64, 0.016*64])

    voxel_mesh = voxel_mesh.apply_translation(-v_vector)

    return voxel_mesh

def read_hdf5_voxel(hdf5_file):
    hdf5_file = h5py.File(hdf5_file, 'r')
    data_voxel = np.asarray(hdf5_file['data'])
    inf_voxel = np.asarray(hdf5_file['inf_image'])
    gt_voxel = np.asarray(hdf5_file['gt_label'])

    return data_voxel, inf_voxel, gt_voxel 

def get_2d_cross_section_list(sid, xml_file_path, hdf5_file):
    xml_id = xml_file_path+'/*' +sid+'.xml'
    xml_file = glob.glob(xml_id)[0]

    inf_section_list = []
    # inf_2dsection_list = []
    gt_section_list = []
    # gt_2dsection_list = []
    # gt_cross_section_list = []
    image_gt_section_points_list = []
    image_inf_section_points_list = []

    data_voxel, inf_voxel, gt_voxel = read_hdf5_voxel(hdf5_file)

    inf_voxel_mesh = get_mesh_from_voxel(inf_voxel)
    gt_voxel_mesh = get_mesh_from_voxel(gt_voxel)


    gt_origin_mesh, plane_normals, plane_origins, matrix_list, plane_scale, plane_vector = get_mesh_and_paras_from_xml(xml_file)

    for o, n in zip(plane_origins, plane_normals):

        # gt_cross_section = cross_section(gt_origin_mesh.vertices, gt_origin_mesh.faces, plane_orig=o, plane_normal=n)
    
        inf_section: trimesh.path.Path3D = inf_voxel_mesh.section(plane_origin=o, plane_normal=n)
        # inf_section_2d: trimesh.path.Path2D = inf_section.to_planar()[0]

        gt_section: trimesh.path.Path3D = gt_voxel_mesh.section(plane_origin=o, plane_normal=n)
        # gt_section_2d: trimesh.path.Path2D = gt_section.to_planar()[0]

        # gt_cross_section_list.append(gt_cross_section)

        inf_section_list.append(inf_section)
        # inf_2dsection_list.append(inf_section_2d)
        gt_section_list.append(gt_section)
        # gt_2dsection_list.append(gt_section_2d)

        # # Create a trimesh.Scene object
        # scene = trimesh.Scene()

        # # Add path1 to the scene
        # scene.add_geometry(inf_section_2d, geom_name='inf_section_2d')

        # # Add path2 to the scene
        # scene.add_geometry(gt_section_2d, geom_name='gt_section_2d')

        # # Show the combined scene
        # scene.show()
    
        # plt.scatter(inf_section_2d.vertices[:, 0], inf_section_2d.vertices[:, 1], color='red', label='inf_section_2d')
        # plt.scatter(gt_section_2d.vertices[:, 0], gt_section_2d.vertices[:, 1], color='blue', label=' gt_section_2d')

        # # Show the plot
        # plt.savefig('cross_sections.png')
        # # plt.show()

    for inf_section, gt_section, matrix in zip(inf_section_list, gt_section_list, matrix_list):
        
        inf_section_verts = inf_section.vertices
        gt_section_verts = gt_section.vertices
        # print(gt_section_verts[0])

        trans_matrix = np.linalg.inv(matrix)

        # print(np.dot(trans_matrix, matrix))
        expend_gt_points = np.ones((len(gt_section_verts), 4))
        expend_gt_points[:, :3] = gt_section_verts
        gt_2d_points = np.dot(trans_matrix, expend_gt_points.T).T
        # print(gt_2d_points.shape)
        # print(gt_2d_points[0]) 
        image_gt_section_points = np.ones((len(gt_2d_points), 2))
        image_gt_section_points = gt_2d_points[:, :2]
        # print(image_gt_section_points[0])

        expend_inf_points = np.ones((len(inf_section_verts), 4))
        expend_inf_points[:, :3] = inf_section_verts
        inf_2d_points = np.dot(trans_matrix, expend_inf_points.T).T
        # print(gt_2d_points.shape)
        # print(gt_2d_points[0]) 
        image_inf_section_points = np.ones((len(inf_2d_points), 2))
        image_inf_section_points = inf_2d_points[:, :2]
        # print(image_inf_section_points[0])
        
        image_gt_section_points_list.append(image_gt_section_points)
        image_inf_section_points_list.append(image_inf_section_points)
        # gt_3d_points = np.dot(matrix, gt_2d_points.T).T
        # print(gt_3d_points[0])

    # # plot the mesh

    # poly_faces = [[3, a, b, c ] for a, b, c in gt_mesh_faces]
    # gt_origin_mesh = pv.PolyData(gt_mesh_verts, poly_faces)

    # inf_mesh_pyvista = pv.wrap(inf_mesh)
    # # gt_mesh_pyvista = pv.wrap(gt_origin_mesh)

    # # Create a plotter instance
    # plotter = pv.Plotter()

    # # Add the mesh to the plotter
    # plotter.add_mesh(inf_mesh_pyvista, show_edges=True, color='blue', opacity=0.5)
    # plotter.add_mesh(gt_origin_mesh, show_edges=True, color='red', opacity=0.5)
    # plotter.add_axes_at_origin()

    # # Display the plotter window
    # plotter.show()

    return image_inf_section_points_list, image_gt_section_points_list

def get_2d_image_list(sid, xml_file_path):
    study_dir = '/staff/wangzhaohui/ultrasound/data/newdata/vpt_data'
    sid_csv_path = '/staff/ydli/projects/OReX/Data/UNet/origin_sid_map.csv'
    
    df = pd.read_csv(sid_csv_path)

    ed_image_list = []
    es_image_list = []

    xml_id = xml_file_path+'/*' +sid+'.xml'
    xml_file = glob.glob(xml_id)[0]

    info_list, cali_info, mesh_text = parseXml(xml_file)
    # print(info_list)
    scan_name_list, ed_idx_list, es_idx_list = [plane['scan_name'] for plane in info_list], [plane['ed'] for plane in info_list], [plane['es'] for plane in info_list]
    height_list, width_list = [plane['height'] for plane in info_list], [plane['width'] for plane in info_list]

    for csv_sid, csv_origin_sid in zip(df['sid'], df['origin_sid']):
        if sid == csv_sid:
            origin_sid = csv_origin_sid
            origin_sid = origin_sid.strip()

    for scan_name, ed_idx, es_idx, height, width in zip(scan_name_list, ed_idx_list, es_idx_list, height_list, width_list):
        vpt_path = os.path.join(study_dir, origin_sid, scan_name)
        print(origin_sid)
        seq = read_scan_vpt(vpt_path, width, height)

        ed_image = seq[ed_idx]
        es_image = seq[es_idx]
        print(ed_idx)
        print(scan_name)

        ed_image_list.append(ed_image)
        es_image_list.append(es_image)

        # plt.imshow(ed_image, cmap='gray')
        # plt.savefig('/staff/ydli/projects/OReX/Data/UNet/figure.png')  # Save the figure as a PNG file

    return ed_image_list, es_image_list


if __name__ == "__main__":

    # Test inf voxel Data

    # data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_128_backup'
    data_path = '/staff/ydli/projects/Med_Seg/seg/new_inf/kbr/v1.4.0/fold3/hdf5_eval'
    xml_file_path = '/staff/ydli/projects/OReX/Data/kbr_patient_backup'


    hdf5_file = data_path+'/SID_3042_10000_es.hdf5'
    file = 'SID_3042_10000_es.hdf5'

    sid = file.split('.')[-2][:-3]
    mode = file.split('.')[-2].split('_')[-1]
    sid_mode = sid+'_'+mode
    
    image_inf_section_points_list, image_gt_section_points_list = get_2d_cross_section_list(sid, xml_file_path, hdf5_file)
    ed_image_list, es_image_list = get_2d_image_list(sid, xml_file_path)

    for i, (ed_image, image_inf_section_points, image_gt_section_points) in enumerate(zip(ed_image_list, image_inf_section_points_list, image_gt_section_points_list)):
        
        plt.imshow(ed_image, cmap='gray')

        # Plot the points on top of the image
        # plt.scatter(image_inf_section_points[:, 0], image_inf_section_points[:, 1], color='red', marker='x', s=1)
        plt.scatter(image_gt_section_points[:, 0], image_gt_section_points[:, 1], color='blue', marker='x', s=1)

        # Show the plot
        plt.savefig('/staff/ydli/projects/Med_Seg/seg/new_inf/kbr/v1.4.0/fold3/2d_png_eval/'+sid_mode+'_'+str(i)+'_figure.png')
        plt.close()
        print('save '+str(i)+' image success')

    # #  Test make 2d Data
    # study_dir = '/staff/wangzhaohui/ultrasound/data/newdata/vpt_data'
    # sid_csv_path = '/staff/ydli/projects/OReX/Data/UNet/origin_sid_map.csv'
    
    # df = pd.read_csv(sid_csv_path)

    # sid_list = []
    # scan_name_list = []
    # ed_idx_list = []
    # es_idx_list = []
    # height_list = []
    # width_list = []

    # ed_image_list = []
    # es_image_list = []

    # ed_label_list = []
    # es_label_list = []

    # for file in os.listdir(data_path):
    #     print(file)
    #     sid = file.split('.')[-2][:-3]
    #     mode = file.split('.')[-2].split('_')[-1]
    #     sid_mode = sid+'_'+mode

    #     hdf5_file = data_path+'/'+file
    #     try:
    #         image_inf_section_points_list, image_gt_section_points_list = get_2d_cross_section_list(sid, xml_file_path, hdf5_file)

    #         xml_id = xml_file_path+'/*' +sid+'.xml'
    #         xml_file = glob.glob(xml_id)[0]

    #         info_list, cali_info, mesh_text = parseXml(xml_file)
    #         # print(info_list)
    #         scan_name_list_sid, ed_idx_list_sid, es_idx_list_sid = [plane['scan_name'] for plane in info_list], [plane['ed'] for plane in info_list], [plane['es'] for plane in info_list]
    #         height_list_sid, width_list_sid = [plane['height'] for plane in info_list], [plane['width'] for plane in info_list]

    #         for csv_sid, csv_origin_sid in zip(df['sid'], df['origin_sid']):
    #             if sid == csv_sid:
    #                 origin_sid = csv_origin_sid
    #                 origin_sid = origin_sid.strip()

    #         # print(len(scan_name_list_sid))
    #         for scan_name, ed_idx, es_idx, height, width in zip(scan_name_list_sid, ed_idx_list_sid, es_idx_list_sid, height_list_sid, width_list_sid):

    #             scan_index = int(scan_name[-1])
    #             # print(scan_index)

    #             if mode == 'ed':
    #                 vpt_path = os.path.join(study_dir, origin_sid, scan_name)
    #                 # print(origin_sid)
    #                 seq = read_scan_vpt(vpt_path, width, height)

    #                 ed_image = seq[ed_idx]
    #                 # print(ed_idx)
    #                 # print(scan_name)

    #                 ed_label = np.zeros((width, height))

    #                 # Set points to 1 in the array
    #                 # print(len(image_gt_section_points_list))
    #                 for point in image_gt_section_points_list[scan_index]:
    #                         # print(point.shape)
    #                         x = round(point[0])
    #                         y = round(point[1])
    #                         ed_label[x, y] = 1

    #                 sid_list.append(sid_mode)
    #                 ed_image_list.append(ed_image)
    #                 ed_label_list.append(ed_label)

    #             else:
    #                 vpt_path = os.path.join(study_dir, origin_sid, scan_name)
    #                 # print(origin_sid)
    #                 seq = read_scan_vpt(vpt_path, width, height)

    #                 es_image = seq[es_idx]
    #                 # print(type(es_image))
    #                 # print(es_image.shape)
    #                 # print(es_idx)
    #                 # print(scan_name)

    #                 es_label = np.zeros((width, height))

    #                 # Set points to 1 in the array
    #                 for point in image_gt_section_points_list[scan_index]:
    #                         # print(point)
    #                         x = round(point[0])
    #                         y = round(point[1])
    #                         es_label[x, y] = 1

    #                 sid_list.append(sid_mode)
    #                 es_image_list.append(es_image)
    #                 es_label_list.append(es_label)
    #             # plt.imshow(ed_image, cmap='gray')
    #             # plt.savefig('/staff/ydli/projects/OReX/Data/UNet/figure.png')  # Save the figure as a PNG file
                    
    #                 # print('len', len(es_image_list))
    #                 # print(type(es_image_list[0]))
    #                 # print(es_image_list[0].shape)
    #     except:
    #         print('create '+sid_mode+' data failed!')
    #     else:
    #         scan_name_list.extend(scan_name_list_sid)
    #         ed_idx_list.extend(ed_idx_list_sid)
    #         es_idx_list.extend(es_idx_list_sid)
    #         height_list.extend(height_list_sid)
    #         width_list.extend(width_list_sid)

    #         unet_path = '/staff/ydli/projects/OReX/Data/UNet/2d_hdf5_backup'

    #         for scan_id in scan_name_list_sid:
    #             if mode == 'ed':
    #                 hdf5_path = unet_path+'/'+sid_mode+'_'+scan_id+'.hdf5'
    #                 hdf5_file = h5py.File(hdf5_path, 'w')
    #                 hdf5_file.create_dataset('image', data=ed_image.astype(np.int16))
    #                 hdf5_file.create_dataset('label', data=ed_label.astype(np.uint8))
    #                 hdf5_file.close()
    #                 print('create '+sid+'_'+scan_id+' hdf5 file success!')
    #             else:
    #                 hdf5_path = unet_path+'/'+sid_mode+'_'+scan_id+'.hdf5'
    #                 hdf5_file = h5py.File(hdf5_path, 'w')
    #                 hdf5_file.create_dataset('image', data=es_image.astype(np.int16))
    #                 hdf5_file.create_dataset('label', data=es_label.astype(np.uint8))
    #                 hdf5_file.close()
    #                 print('create '+sid+'_'+scan_id+' hdf5 file success!')

    # #  Make 2d Data
    # study_dir = '/staff/wangzhaohui/ultrasound/data/newdata/vpt_data'
    # sid_csv_path = '/staff/ydli/projects/OReX/Data/UNet/origin_sid_map.csv'
    
    # df = pd.read_csv(sid_csv_path)

    # sid_list = []
    # scan_name_list = []
    # ed_idx_list = []
    # es_idx_list = []
    # height_list = []
    # width_list = []

    # ed_image_list = []
    # es_image_list = []

    # ed_label_list = []
    # es_label_list = []


    # for file in os.listdir(data_path):
    #     print(file)
    #     sid = file.split('.')[-2][:-3]
    #     mode = file.split('.')[-2].split('_')[-1]
    #     sid_mode = sid+'_'+mode

    #     hdf5_file = data_path+'/'+file

    #     image_inf_section_points_list, image_gt_section_points_list = get_2d_cross_section_list(sid, xml_file_path, hdf5_file)

    #     xml_id = xml_file_path+'/*' +sid+'.xml'
    #     xml_file = glob.glob(xml_id)[0]

    #     info_list, cali_info, mesh_text = parseXml(xml_file)
    #     # print(info_list)
    #     scan_name_list_sid, ed_idx_list_sid, es_idx_list_sid = [plane['scan_name'] for plane in info_list], [plane['ed'] for plane in info_list], [plane['es'] for plane in info_list]
    #     height_list_sid, width_list_sid = [plane['height'] for plane in info_list], [plane['width'] for plane in info_list]

    #     scan_name_list.extend(scan_name_list_sid)
    #     ed_idx_list.extend(ed_idx_list_sid)
    #     es_idx_list.extend(es_idx_list_sid)
    #     height_list.extend(height_list_sid)
    #     width_list.extend(width_list_sid)



    #     for csv_sid, csv_origin_sid in zip(df['sid'], df['origin_sid']):
    #         if sid == csv_sid:
    #             origin_sid = csv_origin_sid
    #             origin_sid = origin_sid.strip()

    #     for scan_name, ed_idx, es_idx, height, width in zip(scan_name_list_sid, ed_idx_list_sid, es_idx_list_sid, height_list_sid, width_list_sid):

    #         scan_index = int(scan_name[-1])
    #         print(scan_index)

    #         if mode == 'ed':
    #             vpt_path = os.path.join(study_dir, origin_sid, scan_name)
    #             print(origin_sid)
    #             seq = read_scan_vpt(vpt_path, width, height)

    #             ed_image = seq[ed_idx]
    #             print(ed_idx)
    #             print(scan_name)

    #             ed_label = np.zeros((width, height))

    #             # Set points to 1 in the array
    #             for point in image_gt_section_points_list:

    #                     x = round(point[0])
    #                     y = round(point[0])
    #                     ed_label[x, y] = 1

    #             sid_list.append(sid_mode)
    #             ed_image_list.append(ed_image)
    #             ed_label_list.append(ed_label)

    #         else:
    #             vpt_path = os.path.join(study_dir, origin_sid, scan_name)
    #             print(origin_sid)
    #             seq = read_scan_vpt(vpt_path, width, height)

    #             es_image = seq[es_idx]
    #             # print(type(es_image))
    #             # print(es_image.shape)
    #             print(es_idx)
    #             print(scan_name)

    #             es_label = np.zeros((width, height))

    #             # Set points to 1 in the array
    #             for point in image_gt_section_points_list[scan_index]:
    #                     # print(point)
    #                     x = round(point[0])
    #                     y = round(point[0])
    #                     es_label[x, y] = 1

    #             sid_list.append(sid_mode)
    #             es_image_list.append(es_image)
    #             es_label_list.append(es_label)
    #         # plt.imshow(ed_image, cmap='gray')
    #         # plt.savefig('/staff/ydli/projects/OReX/Data/UNet/figure.png')  # Save the figure as a PNG file
    #     break
                
    # print('len', len(es_image_list))
    # print(type(es_image_list[0]))
    # print(es_image_list[0].shape)
    
    # unet_path = '/staff/ydli/projects/OReX/Data/UNet/2d_hdf5_backup'

    # for scan_id in scan_name_list_sid:
    #     if mode == 'ed':
    #         hdf5_path = unet_path+'/'+sid+'_'+scan_id+'.hdf5'
    #         hdf5_file = h5py.File(hdf5_path, 'w')
    #         hdf5_file.create_dataset('image', data=ed_image.astype(np.int16))
    #         hdf5_file.create_dataset('label', data=ed_label.astype(np.uint8))
    #         hdf5_file.close()
    #         print('create '+sid+'_'+scan_id+' hdf5 file success!')
    #     else:
    #         hdf5_path = unet_path+'/'+sid+'_'+scan_id+'.hdf5'
    #         hdf5_file = h5py.File(hdf5_path, 'w')
    #         hdf5_file.create_dataset('image', data=es_image.astype(np.int16))
    #         hdf5_file.create_dataset('label', data=es_label.astype(np.uint8))
    #         hdf5_file.close()
    #         print('create '+sid+'_'+scan_id+' hdf5 file success!')