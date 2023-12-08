import numpy as np
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
    mesh = Trimesh(verts, faces)
    voxels = creation.local_voxelize(mesh, mesh.centroid, pitch=0.5, radius=256, fill=True)
    # print(voxels)
    # voxels.show()
    matrix = voxels.matrix
    # print(type(matrix))

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

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist   

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius, radius * 2]))


    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                          vertex_normals=np.asarray(mesh.vertex_normals))
    
    convex_hull = trimesh.convex.convex_hull(tri_mesh, qhull_options='QbB Pp Qt', repair=True)
    voxels = creation.local_voxelize(convex_hull, convex_hull.centroid, pitch=0.05, radius=256, fill=True)
    # voxels.show()
    matrix = voxels.matrix

    return matrix


if __name__ == '__main__':
    

    filename = '/staff/ydli/projects/OReX/Data/kbr_patient_backup/VpStudy_SID_3042_10280.xml' # '/staff/ydli/projects/OReX/trash/kbr_ed_heart_SID_3042_10280.ply' # '/staff/ydli/projects/OReX/Data/kbr_patient_backup/VpStudy_SID_3042_10280.xml'
    
    # pcd = o3d.io.read_point_cloud(filename)
    # pcd.estimate_normals()

    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 1.5 * avg_dist   

    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #         pcd,
    #         o3d.utility.DoubleVector([radius, radius * 2]))


    # tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
    #                       vertex_normals=np.asarray(mesh.vertex_normals))
    
    # convex_hull = trimesh.convex.convex_hull(tri_mesh, qhull_options='QbB Pp Qt', repair=True)
    # voxels = creation.local_voxelize(convex_hull, convex_hull.centroid, pitch=0.05, radius=256, fill=True)
    # # voxels.show()


    matrix = make_kbr_gt_mesh2voxels(filename)
    count_true = np.sum(matrix)
    print(count_true)

    # import pandas as pd 
    # import openpyxl
    # # df = pd.DataFrame(matrix)
    # df = pd.DataFrame(matrix.reshape(-1, matrix.shape[-1]), columns=[f'Col_{i}' for i in range(matrix.shape[-1])])
    # df.to_excel(f'1.xlsx', index=False)
    # # print(matrix[0][3])
    
    
