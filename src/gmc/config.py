import pathlib
import os

source_dir = os.path.dirname(__file__)
data_base_path = pathlib.Path(f"{source_dir}/../../data")

eval_slize_size = 1024 * 1024 * 100
eval_img_n_sample_points = 50 * 50
