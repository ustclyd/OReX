import argparse
import os

import shutil
import pyvista as pv
import numpy as np
import trimesh
from matplotlib.path import Path
from meshcut import cross_section
from shapely.geometry import LinearRing
from stl import mesh as mesh2
from tqdm import tqdm
import glob

from Dataset.CSL import CSL

def read_csl(csl_file):
    csl =  CSL.from_csl_file(csl_file)

    plane_number = len(csl.planes)
    plane_para_list = []
    plane_verts_list = []
    plane_edges_list = []

    for plane in csl.planes:
        plane_paras = plane.plane_params
        # print(type(plane_paras))
        plane_para_list.append(plane_paras)

        plane_verts = plane.vertices
        # print(type(plane_verts))
        plane_verts_list.append(plane_verts)
        
        plane_edges = []
        plane_connected_components = plane.connected_components[0].vertices_indices.tolist()
        # print(plane_connected_components)
        # print(type(plane_connected_components))
        plane_start_v = plane_connected_components[0:]
        plane_end_v = plane_connected_components[1:]
        plane_end_v.append(0)
        # print(plane_end_v)
        # print(type(plane_connected_components))
        # plane_edges_list.append(plane_edges)
        for start_v, end_v in zip(plane_start_v, plane_end_v):
            edge = []
            edge.append(start_v)
            edge.append(end_v)
            edge.append(0)
            edge.append(1)
            plane_edges.append(edge)
        plane_edges_list.append(plane_edges)


    return plane_number, plane_para_list, plane_verts_list, plane_edges_list

def to_contour_file(file_path, plane_number, plane_para_list, plane_verts_list, plane_edges_list):

    f = open(file_path, "w")

    f.write(str(plane_number))

    for index in range(plane_number):
        plane_para = list(plane_para_list[index])
        plane_verts = plane_verts_list[index]
        plane_edges = plane_edges_list[index]
        para_str = '{0[0]} {0[1]} {0[2]} {0[3]}'.format(plane_para)
        f.write(para_str)

        verts_str = ' '.join(['{:.10f} {:.10f} {:.10f}'.format(*vert) + '\n' for vert in plane_verts])
        f.write(verts_str)
    
        edge_str = ' '.join(['{0[0]} {0[1]} {0[2]} {0[3]}'.format(*edge) + '\n' for edge in plane_edges])
        f.write(edge_str)

    f.close()

    return


def read_and_write_contour_file(csl_file, contour_file):

    csl =  CSL.from_csl_file(csl_file)

    plane_number = len(csl.planes)
    plane_para_list = []
    plane_verts_list = []
    plane_edges_list = []

    f = open(contour_file, "w")

    f.write(str(plane_number)+ '\n')

    for plane in csl.planes:
        plane_paras = plane.plane_params
        # print(type(plane_paras))
        para_str = '{0[0]} {0[1]} {0[2]} {0[3]}'.format(plane_paras)+ '\n'
        f.write(para_str)

        plane_verts = plane.vertices
        print(type(plane_verts))
        print(plane_verts.shape)
        print(plane_verts[0])
    
        plane_edges = []
        plane_connected_components = plane.connected_components[0].vertices_indices.tolist()
        # print(plane_connected_components)
        # print(type(plane_connected_components))
        plane_start_v = plane_connected_components[0:]
        plane_end_v = plane_connected_components[1:]
        plane_end_v.append(0)
        # print(plane_end_v)
        # print(type(plane_connected_components))
        # plane_edges_list.append(plane_edges)

        number_str = str(len(plane_verts))+' '+str(len(plane_start_v))+'\n'
        f.write(number_str)

        for vert in plane_verts:
            verts_str ='{:.10f} {:.10f} {:.10f}'.format(*vert) + '\n'
            f.write(verts_str)

        for start_v, end_v in zip(plane_start_v, plane_end_v):
            edge = []
            edge.append(start_v)
            edge.append(end_v)
            edge.append(0)
            edge.append(1)
            # print(edge)
            edge_str = str(edge[0])+' '+str(edge[1])+' '+str(edge[2])+' '+str(edge[3])+'\n'
            f.write(edge_str)

    f.close()

    return

if __name__ == '__main__':

    # Test read csl file
    csl_path = '/staff/ydli/projects/OReX/Data/kbr_without_ref_backup'
    contour_path = '/staff/ydli/projects/OReX/Data/kbr_contour_backup'

    csl_id_file = csl_path +'/*.csl'
    # plane_number, plane_para_list, plane_verts_list, plane_edges_list = read_csl(csl_file)
    # print(plane_number)
    # # print(plane_para_list)
    # print(type(plane_para_list[0]))
    # print(len(plane_para_list))
    # # print(plane_verts_list)
    # print(type(plane_verts_list[0]))
    # print(len(plane_verts_list))
    # # print(plane_edges_list)
    # print(type(plane_edges_list[0]))
    # print(len(plane_edges_list))
    for csl_file in glob.glob(csl_id_file):
        id_name = csl_file.split('/')[-1].split('.')[0]
        print(id_name)
        contour_file = contour_path+'/'+ id_name+'.contour'
        read_and_write_contour_file(csl_file, contour_file)
        break