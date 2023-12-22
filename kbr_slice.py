import numpy as np
import pyvista as pv
import open3d as o3d
from xml_reader import parseXml
from trimesh import Trimesh
from trimesh.sample import sample_surface
from Dataset.CSL import CSL

from scipy.spatial import ConvexHull

from kbr_mesh import KBRMesh

def read_kbr_plane(transform):
    '''
    transform: np.array, 4x4 matrix
    
    return:
    -------
    normal: normalized
    d
    '''
    n = np.array([0, 0, 1, 0])
    o = np.array([0, 0, 0, 1])
    normal = np.dot(transform, n.T)[:3] # (A, B, C)
    normal = normal / np.linalg.norm(normal)
    o = np.dot(transform, o.T)[:3]
    d = -np.dot(normal, o.T)    # D

    return normal, d

def read_kbr_mesh(mesh_text):
    '''
    return:
    -------
    verts, np.array, (n_v, 3)
    faces, np.array, (n_f, 3)
    '''

    kbr_mesh = KBRMesh(mesh_text)

    verts = kbr_mesh.points
    # print(verts)
    faces = kbr_mesh.faces
    # print(faces)

    return verts, faces

if __name__ == '__main__':
    # file_name = r'/staff/ydli/projects/OReX/mesh_text'
    '''
    px, py, pz = np.array([])
    ox, oy, oz, ow = np.array([])

    verts, faces = read_kbr_mesh(file_name)
    edges = ... # for f in faces....
    edges = np.unique(edges, axis=0)

    # new_points, new_edges, new_faces = loop(verts, edges, faces)

    poly_edges = [[2, e[0], e[1]] for e in edges]
    poly_faces = [[3, f[0], f[1], f[2]] for f in faces]
    poly = pv.PolyData(verts, lines=poly_edges, faces=poly_faces)
    subed_poly = pv.PolyData(verts+20, lines=poly_edges, faces=poly_faces)
    plotter = pv.Plotter()
    plotter.add_mesh(subed_poly, color='red', style='wireframe')
    plotter.add_mesh(poly, color='b', style='wireframe')
    plotter.show()
    
    '''
    filename =  '/staff/ydli/projects/OReX/Data/kbr_patient_backup/VpStudy_SID_3042_10280.xml' # '/staff/ydli/projects/OReX/trash/kbr_ed_heart_SID_3042_10280.ply' # '/staff/ydli/projects/OReX/VpStudy_bak.xml'

    # reader = pv.get_reader(filename)
    # mesh = reader.read()

    info_list, cali_info, mesh_text = parseXml(filename)
    verts, faces = read_kbr_mesh(mesh_text['es'])
    # pcd = o3d.io.read_point_cloud(filename)
    # verts -= np.mean(verts, axis=0)
    # scale = 1.1
    # verts /= scale * np.max(np.absolute(verts))
    # verts += 1
    # reader = pv.get_reader(filename)

    # obj_name = '/staff/ydli/projects/OReX/mesh_last_300.obj'
    # reader_obj = pv.get_reader(obj_name)
    # mesh = o3d.io.read_triangle_mesh(obj_name)# reader_obj.read()
    # mesh = Trimesh(verts, faces) # reader.read() # 
    # print(mesh)
    # mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
    #        center=mesh.get_center())
    # o3d.visualization.draw_geometries([mesh])
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
    #                                                           voxel_size=0.005)
    # print(voxel_grid)
    # voxels = voxel_grid.get_voxels()
    # indices = np.stack(list(vx.grid_index for vx in voxels))
    # o3d.visualization.draw_geometries([voxel_grid])
    # # pl = pv.Plotter()
    # pl.add_mesh(mesh, show_edges=True, color= 'red')
    # pl.add_mesh(mesh_obj, show_edges=True, color= 'red', style='wireframe')
    # pl.show()

    # hull = ConvexHull(verts * (1 + 0.05))
    # mesh = Trimesh(hull.points, hull.simplices)
    # verts = sample_surface(Trimesh(hull.points, hull.simplices), 2 ** 14)
    pv.plot(
    verts,
    scalars=verts[:, 2],
    render_points_as_spheres=True,
    point_size=20,
    show_scalar_bar=False,
)
    # pl = pv.Plotter()
    # pl.add_mesh(mesh, show_edges=True, color= 'white')
    # pl.show()
    # boundary_xyzs = np.array(sample_surface(Trimesh(hull.points, hull.simplices), args.n_samples_boundary)[0])
    # return boundary_xyzs