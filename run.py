import os

import subprocess

if __name__ == '__main__':
    input_dir = './Data/kbr'
    output_dir = './output/directory/kbr'
    script_path = './Main.py'
    for filename in os.listdir(input_dir):
        input_csl_path = os.path.join(input_dir, filename)
        output_mesh_dir = os.path.join(output_dir)
        script_args=[
                    "python3",
                    script_path,
                    output_mesh_dir,
                    input_csl_path,
                    "--cuda_device=1"
                    ]
        p = subprocess.call(script_args)