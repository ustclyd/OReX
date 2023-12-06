import numpy as np
import pandas as pd
import pyvista as pv

STRUCT_LABELS = ['Apical_subTA_freewall', 'FourC_basal_bulge', 'Sub_Tricuspid', 'FourC_freewallTA', 'endo_point_surface', 'RVOT_freewall', 'septal_edge', 'basal_bulge_rv', 'sub_tricuspid', 'super1_RV', 'sub_aortic_dip', 'unspecified', 'SAX_subTA_freewall', 'Septum_Boundary', 'FourC_basal_freewall', 'FourC_septalTA', 'Tricuspid_Annulus', 'SAX_septum', 'sub_pulmonic', 'RV_Septum', 'RV_super8', 'RV_super4', 'RVOT_Posterior', 'apex_freewall', 'Pulmonic_Valve', 'rv_angle_freewall', 'FourC_RX', 'SAX_subTA_septum', 'RV_super2', 'RVOT_Anterior', 
'sub_pulmonic2', 'RVOT_Ao', 'SAX_subPV_freewall', 'Septum_Between_Valves', 'apex_edge', 'FourC_freewall', 'PV_Point_Ao', 'SAX_mid_freewall', 'Sub_Pulmonary', 'is_sharp', 'FourC_mid_freewall', 'Apical_subPV_freewall', 'ApicalRX_subTA', 'FourC_RXfreewall', 'ApicalRX_subPV', 'RX_lowest', 'RV_super1', 'apex', 'not_RV_Septum', 'SAX_subPV_septum', 'BasalSAX_dip']

STRUCTS = [
    'endo_point_surface',
    'rv_septum',
    'pulmonic_valve',
    'Sub_Pulmonary',
    'tricuspid_annulus',
    'basal_bulge_rv',
    'FourC_RX',
    'septal_edge',
]
class StructLable():
    pass

class KBRMesh():
    # TODO: different mesh only points coordinates are different. All Edges Faces and labels are the same.
    def __init__(self, mesh_text):
        self.read_str(mesh_text)
        
    def read_str(self, mesh_text):
        mesh_text = mesh_text.strip()
        lines = mesh_text.split('\n')
        v_list = []
        e_list = []
        f_list = []
        v_labels = dict([(k.lower(), []) for k in STRUCT_LABELS])
        e_labels = dict([(k.lower(), []) for k in STRUCT_LABELS])
        f_labels = dict([(k.lower(), []) for k in STRUCT_LABELS])
        labels_set = set()
        normal_list = []
        
        for line in lines:
            line = line.strip()
            items = line.split(' ')
            # print(line)
            if line.startswith('#'):
                # print('continue')
                continue

            if line.startswith('Vertex'): # vertex
                vid = int(items[1][1:])
                x, y, z = items[2:5]
                v = (float(x), float(y), float(z))
                v_list.append(v)
                labels = items[5:]
                for l in labels:
                    ll = l.lower()
                    labels_set.add(ll)               
                    v_labels[ll].append(vid)
                        
            elif line.startswith('Edge'): # edge 
                eid = int(items[1][1:])
                p1, p2 = line.split(' ')[2:4]
                e = (int(p1[1:]), int(p2[1:]))
                e_list.append(e)
                labels = items[4:]
                for l in labels:
                    ll = l.lower()
                    labels_set.add(ll)                    
                    e_labels[ll].append(eid)
                    
            elif line.startswith('Face'): # face
                fid = int(items[1][1:])
                p1, p2, p3 = items[2:5]
                f = (int(p1[1:]), int(p2[1:]), int(p3[1:]))
                f_list.append(f)
                labels = items[5:]
                for l in labels:
                    ll = l.lower()
                    labels_set.add(ll)                  
                    f_labels[ll].append(fid)
                    
            else: # normal
                nx, ny, nz = items[4:]
                norm = (float(nx[1:]), float(ny[1:]), float(nz[1:]))
                normal_list.append(norm)
        self.points = np.array(v_list)
        self.edges = np.array(e_list)
        self.faces = np.array(f_list)
        # self.normals = np.array(normal_list) # TODO:
        self.v_labels = v_labels
        self.e_labels = e_labels
        self.f_labels = f_labels
        self.labels = list(labels_set)
        
    def to_poly(self):
        poly_edges = [[2, e[0], e[1]] for e in self.edges]
        poly_faces = [[3, f[0], f[1], f[2]] for f in self.faces]
        poly = pv.PolyData(self.points, lines=poly_edges, faces=poly_faces)
        return poly
    
    def make_point_table(self):
        num_struct = len(STRUCTS)
        num_points = self.points.shape[0]
        point_table = np.zeros((num_points, 3+num_struct))
        point_table[:,:3] = self.points
        for struct_id in range(num_struct):
            struct_name_low = STRUCTS[struct_id].lower()
            point_table[self.v_labels[struct_name_low],3+struct_id] = 1
        return point_table
    
    def make_face_table(self):
        num_struct = len(STRUCTS)
        num_faces = self.faces.shape[0]
        face_table = np.zeros((num_faces, 3+num_struct))
        face_table[:,:3] = self.faces
        for struct_id in range(num_struct):
            struct_name_low = STRUCTS[struct_id].lower()
            face_table[self.f_labels[struct_name_low],3+struct_id] = 1
        return face_table
    

class KBRPoint():
    def __init__(self, x, y, z, label):
        self.x = x
        self.y = y
        self.z = z
        self.label = label
    
class KBRPointCloud:
    def __init__(self):
        pass


def select_poly(kbr_mesh, vid_list, eid_list, fid_list):
    vid_map = {}
    for new_vid, orig_vid in enumerate(vid_list):
        vid_map[orig_vid] = new_vid
    vid_map_f = lambda x: vid_map[x]

    sub_points = kbr_mesh.points[vid_list]
    
    if len(eid_list) > 0:
        sub_edges = kbr_mesh.edges[eid_list]
        sub_edges = np.vectorize(vid_map_f)(sub_edges)
    else:
        sub_edges = np.array([])
    if len(fid_list) > 0:
        sub_faces = kbr_mesh.faces[fid_list]
        sub_faces = np.vectorize(vid_map_f)(sub_faces)
    else:
        sub_faces = np.array([])
    
    poly_edges = [[2, e[0], e[1]] for e in sub_edges] if len(sub_edges) > 0 else None
    poly_faces = [[3, f[0], f[1], f[2]] for f in sub_faces] if len(sub_faces) > 0 else None
    poly = pv.PolyData(sub_points, lines=poly_edges, faces=poly_faces)

    return poly


def sub_struct(kbr_mesh, label): 
    sub_vid = kbr_mesh.v_labels[label]
    sub_eid = kbr_mesh.e_labels[label]
    sub_fid = kbr_mesh.f_labels[label]
    sub_poly = select_poly(kbr_mesh, sub_vid, sub_eid, sub_fid)
    return sub_poly

def test_mesh_reading():
    
    file_path = r'/staff/ydli/projects/OReX/mesh_text'
    with open(file_path, 'r') as f:
        mesh_text = f.read()
    
    kbr_mesh = KBRMesh(mesh_text)
    print(len(kbr_mesh.points))
    print(len(kbr_mesh.edges))
    print(len(kbr_mesh.faces))
    # print(len(kbr_mesh.normals))
    print(kbr_mesh.labels)
    for k, v in kbr_mesh.v_labels.items():
        print(k, len(v))
        
    for k, v in kbr_mesh.e_labels.items():
        print(k, len(v))        
        
    for k, v in kbr_mesh.f_labels.items():
        print(k, len(v))
    # print(kbr_mesh.v_labels)
    
def test_sub_poly():
    file_path = r'/staff/ydli/projects/OReX/mesh_text'
    with open(file_path, 'r') as f:
        mesh_text = f.read()
    kbr_mesh = KBRMesh(mesh_text)
    
    label = 'tricuspid_annulus'
    sub_str = sub_struct(kbr_mesh, label)
    print(sub_str.points.shape)
    
    plotter = pv.Plotter()
    plotter.add_mesh(kbr_mesh.to_poly(), color='w', opacity=0.5)
    plotter.add_mesh(sub_str, color='b')
    plotter.show()
    
    
def test_make_point_table():
    
    file_path = r'/staff/ydli/projects/OReX/mesh_text'
    with open(file_path, 'r') as f:
        mesh_text = f.read()
    kbr_mesh = KBRMesh(mesh_text)
    point_table = kbr_mesh.make_point_table()
    pt_df = pd.DataFrame(point_table)
    pt_df.to_csv('./point_table.csv')
    

def read_test_point_cloud():
    struct_inv = dict([(struct_name, i) for i, struct_name in enumerate(STRUCTS)])
    pc_path = 'pointcloud_3042_10240_ed'
    with open(pc_path) as pcf:
        pc_lines = pcf.readlines()
        
    pc_list = []
    for pc_line in pc_lines:
        struct_name, pos_str = pc_line.split(': ')
        pos = pos_str.strip().strip('(').strip(')').split(',')
        row = pos + [struct_inv[struct_name]]
        pc_list.append(row)

    return np.array(pc_list, dtype=np.float)


if __name__ == '__main__':
    test_sub_poly()

    
