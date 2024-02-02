import os
import pandas as pd
import subprocess

if __name__ == '__main__':
    input_dir = './Data/kbr'
    output_dir = './output/directory/kbr'
    script_path = './Main.py'
    csv_filepath = './output/directory/kbr/metrics.csv'
    data_df = pd.read_csv(csv_filepath, encoding='gbk', usecols=[1])
    data = data_df.values.tolist()
    # print(data)
    # idd = 'SID_3039_10620_ed'
    # stri_list = [idd]
    # print(stri_list in data)
    for filename in os.listdir(input_dir):
        mode = filename.split('_')[1]
        sid = filename.split('_')[-1].split('.')[0]
        sid = 'SID' + '_' + filename.split('_')[-2] + '_' + sid
        sid_mode = sid+'_'+mode
        sid_mode_list = [sid_mode]
        # print(sid_mode)
        if sid_mode_list in data:
            print(sid_mode+'already reconed')
        else:
            print(sid_mode+'reconing')
            input_csl_path = os.path.join(input_dir, filename)
            output_mesh_dir = os.path.join(output_dir)
            script_args=[
                        "python3",
                        script_path,
                        output_mesh_dir,
                        input_csl_path,
                        "--cuda_device=6"
                        ]
            p = subprocess.call(script_args)