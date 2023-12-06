import numpy as np
import quaternion
from xml_reader import parseXml



def patient_movement_correction(patient_sensor_orientation, patient_initial_orientation, sensor_orientation, patient_sensor_position, patient_initial_position, sensor_posiotion):
    orien_matrix = patient_orein_correction(patient_sensor_orientation, patient_initial_orientation, sensor_orientation)
    move_vector = patient_vector_correction(patient_sensor_position, patient_initial_position, sensor_posiotion)
    # print(orien_matrix)
    # print(move_vector)
    correct_v = move_vector - patient_initial_position
    # print("the shape of orien_matrix is " + str(orien_matrix.shape))
    # print("the shape of correct_v is " + str(correct_v.shape))
    correct_v = np.dot(orien_matrix, correct_v.T)
    correct_v = correct_v + patient_initial_position
    # print(correct_v)
    return orien_matrix, correct_v

def patient_orein_correction(patient_sensor_orientation, patient_initial_orientation, sensor_orientation):
    """
    imput: 
        patient_initial_orientation, 
        patient_sensor_orientation
    return
        patient orientation correction matrix

    """
    patientq = quaternion.from_float_array(patient_sensor_orientation)
    patientq0 = quaternion.from_float_array(patient_initial_orientation)
    # print(patientq)
    probq = quaternion.from_float_array(sensor_orientation)
    # print(np.linalg.norm(quaternion.as_float_array(patientq)))

    q1 = patientq.inverse()
    # print(q1)
    # print(patientq0)
    qdelta = patientq0*q1
    # print(qdelta)
    qout = qdelta*probq
    # print(qout)
    
    orien_matrix = quaternion.as_rotation_matrix(qout)
    # print(orien_matrix)
    
    return orien_matrix

def patient_vector_correction(patient_sensor_position, patient_initial_position, sensor_posiotion):
    vdelta = patient_sensor_position - patient_initial_position
    vout = sensor_posiotion - vdelta

    return vout


def make_44matrix(orien_matrix, pos_vector):
    trans_matrix = np.zeros((4, 3))

    # set_rotate_matrix
    r, c = 0, 0
    # print(orien_matrix.shape[0])
    trans_matrix[r:r+orien_matrix.shape[0], c:c+orien_matrix.shape[1]] += orien_matrix
    # print(trans_matrix)
    # set_pos_vector
    # print(pos_vector)
    pos_vector = np.insert(pos_vector, 3, 1)
    # print(pos_vector)
    _pos_vector = pos_vector.T
    #print(_pos_vector)
    trans_matrix = np.insert(trans_matrix, 3, _pos_vector, axis=1)
    # print(trans_matrix)
    # print("the shape of make_matrix is " + str(trans_matrix.shape))
    return trans_matrix



def make_plane_transform(enablePatientMovementCorrection, 
                        patient_sensor_orientation, patient_initial_orientation, sensor_orientation, patient_sensor_position, patient_initial_position, sensor_posiotion, 
                        calibration_rotation_matrix, calibration_trans_vector, 
                        depth_XMillimetersPerPixel, depth_YMillimetersPerPixel, depth_OriginPixel_X, depth_OriginPixel_Y, 
                        ):
    '''
    return: np.array, 4x4 matrix

    '''

    # 0. read the file para
    #TODO
    c_orien_matrix = calibration_rotation_matrix
    c_pos_vector = calibration_trans_vector

    # 1. if enable patient correction
    if enablePatientMovementCorrection:
        orien_matrix,  pos_vector = patient_movement_correction(patient_sensor_orientation, patient_initial_orientation, sensor_orientation, patient_sensor_position, patient_initial_position, sensor_posiotion)
    
    # 2. patient correction
    else:
        orien_q = quaternion.from_float_array(sensor_orientation)
        orien_matrix = quaternion.as_rotation_matrix(orien_q)
        # print(orien_matrix)
        pos_vector = sensor_posiotion
        # print(pos_vector)

    # 3.make_44matrix: sensor_matrix & calibration_matrix
    # print(orien_matrix)
    # print("the shape of sensor_orien_matrix is " + str(orien_matrix.shape))
    sensor_matrix = make_44matrix(orien_matrix, pos_vector)
    # print(sensor_matrix)
    calibration_matrix = make_44matrix(c_orien_matrix, c_pos_vector)
    # print(calibration_matrix)

    # 4. make_44matrix: scale & image origin translation
    scale_matrix = np.eye(4)
    scale_matrix[0][0], scale_matrix[1][1] = depth_XMillimetersPerPixel, depth_YMillimetersPerPixel
    # print(scale_matrix)
    origin_matrix = np.eye(4)
    origin_matrix[0][3], origin_matrix[1][3] = -depth_OriginPixel_X, -depth_OriginPixel_Y
    # print(origin_matrix)

    # 4. output transform
    tans_matrix = np.dot(sensor_matrix, calibration_matrix)
    tans_matrix = np.dot(tans_matrix, scale_matrix)
    tans_matrix = np.dot(tans_matrix, origin_matrix)
    # tans_matrix[2] = -tans_matrix[2]
    
    
    return tans_matrix

if __name__ == '__main__':
    """
    # each plane
    enablePatientMovementCorrection = 0 # bool
    patient_initial_position = np.array([]) # x, y, z
    patient_initial_orientation = np.array([]) # x, y, z, w
    patient_sensor_position = np.array([]) # x, y, z ed/es
    patient_sensor_orientation = np.array([]) # x, y, z, w
    sensor_posiotion = np.array([]) # x, y, z
    sensor_orientation = np.array([]) # x, y, z, w

    # calibration
    calibration_trans_vector = np.array([])
    calibration_rotation_matrix = np.array([
        [],
        [],
        [],
        []
    ])
    
    #scale
    depth_XMillimetersPerPixel = 0
    depth_YMillimetersPerPixel = 0

    #image origin translation
    depth_OriginPixel_X = -0
    depth_OriginPixel_Y = -0

    transform = make_plane_transform()
    """

    #TEST
    #scan0
    # enablePatientMovementCorrection = 0
    # patient_sensor_orientation = np.array([0.274661034, -0.141114742, 0.4680224, 0.828011453]) # w, x, y, z
    # patient_initial_orientation = np.array([0.274661034, -0.141114742, 0.4680224, 0.828011453]) # w, x, y, z
    # sensor_orientation = np.array([0.595256, -0.4525459,-0.289692879, 0.5974534]) # w, x, y, z,

    # patient_sensor_position = np.array([ 204.8247, 203.931732, 486.667969 ])
    # patient_initial_position = np.array([204.936325, 203.931732, 486.667969])
    # sensor_posiotion = np.array([ 310.0834, -70.9910, -189.0861 ])

    # c_orien_matrix = np.array([[0.01515140877822618, 0.99987812569996293, -0.0037641146056174293], [-0.0089131071713035564, 0.0038994578400193261, 0.99995267425469025], [0.99984548372865489, -0.015117141769628525, 0.0089711031723531658]])
    # c_pos_vector = np.array([58.776005261429063, 0.022962416538761781, 28.80773985780079])
    # depth_XMillimetersPerPixel = 0.25156177156177156
    # depth_YMillimetersPerPixel = 0.25239202657807308
    # depth_OriginPixel_X = 628.5
    # depth_OriginPixel_Y = 86


    #scan3
    # enablePatientMovementCorrection = 0
    # patient_sensor_orientation = np.array([0.274661034, -0.141114742, 0.4680224, 0.828011453]) # w, x, y, z
    # patient_initial_orientation = np.array([0.274661034, -0.141114742, 0.4680224, 0.828011453]) # w, x, y, z
    # sensor_orientation = np.array([0.633468151, -0.397489369,-0.06360318, 0.660813868]) # w, x, y, z,

    # patient_sensor_position = np.array([ 203.820114, 203.262009, 486.667969 ])
    # patient_initial_position = np.array([ 203.820114, 203.262009, 486.667969 ])
    # sensor_posiotion = np.array([ 300.260742, -71.66074, -186.853714 ])

    # c_orien_matrix = np.array([[0.01515140877822618, 0.99987812569996293, -0.0037641146056174293], [-0.0089131071713035564, 0.0038994578400193261, 0.99995267425469025], [0.99984548372865489, -0.015117141769628525, 0.0089711031723531658]])
    # c_pos_vector = np.array([58.776005261429063, 0.022962416538761781, 28.80773985780079])
    # depth_XMillimetersPerPixel = 0.25156177156177156
    # depth_YMillimetersPerPixel = 0.25239202657807308
    # depth_OriginPixel_X = 628.5
    # depth_OriginPixel_Y = 86



    # orien_matrix,  pos_vector = patient_movement_correction(patient_sensor_orientation, patient_initial_orientation, sensor_orientation, patient_sensor_position, patient_initial_position, sensor_posiotion)

    # print(orien_matrix)
    # print(pos_vector)

    # sensor_matrix = make_44matrix(orien_matrix, pos_vector)

    # print(sensor_matrix)
    # print("the shape of sensor_matrix is " + str(sensor_matrix.shape))

    # calibration_matrix = make_44matrix(c_orien_matrix, c_pos_vector)
    # print(calibration_matrix)
    # print("the shape of calibration_matrix is " + str(calibration_matrix.shape))

    file_name = 'VpStudy_SID_3042_10240.xml'
    input_path = '/staff/ydli/projects/OReX/Data/kbr_patient_backup/' + file_name
    out_path = '/staff/ydli/projects/OReX/trash/'

    info_list, cali_info, mesh_text = parseXml(input_path)
    
    trans_matrixs_list = []
    normal_list = []
    d_list = []
    mode_list = ['ed']

    for mode in mode_list:
        for plane in info_list:
            depth_label = plane['depth_label']
            enablePatientMovementCorrection = 0
            patient_sensor_orientation = np.array(plane[mode+'_patient_orientation'])[[3,0,1,2]]
            patient_initial_orientation = np.array(plane['init_patient_orientation'])[[3,0,1,2]]
            sensor_orientation = np.array(plane[mode+'_sensor_orientation'])[[3,0,1,2]]
            patient_sensor_position = np.array(plane[mode+'_patient_location'])
            patient_initial_position = np.array(plane['init_patient_location'])
            sensor_posiotion = np.array(plane[mode+'_sensor_location'])
            c_orien_matrix = np.reshape(cali_info['calibration_rotation_matrix'], (3,3))
            c_pos_vector = np.array(cali_info['calibration_translation_vector'])
            for depth in cali_info['depth_list']:
                if depth['depth_label'] == depth_label:
                    depth_XMillimetersPerPixel = depth['x_millmeter_per_pixel']
                    depth_YMillimetersPerPixel = depth['y_millmeter_per_pixel']
                    depth_OriginPixel_X = depth['origin_x']
                    depth_OriginPixel_Y = depth['origin_y']
        # print(plane_scale)
            trans_matrix = make_plane_transform(enablePatientMovementCorrection, patient_sensor_orientation, patient_initial_orientation, sensor_orientation, patient_sensor_position, patient_initial_position, sensor_posiotion, c_orien_matrix, c_pos_vector, depth_XMillimetersPerPixel, depth_YMillimetersPerPixel, depth_OriginPixel_X, depth_OriginPixel_Y)
            trans_matrixs_list.append(trans_matrix)
    
    
    print("the len of trans_matrix is ")
    print(len(trans_matrixs_list))
    print(trans_matrixs_list)

    # point = np.array([636, 218, 0, 1])
    # new_point = np.dot(trans_matrix, point.T)
    # print(new_point)
