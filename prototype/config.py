import pathlib
data_base_path = pathlib.Path("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data")
data_base_path = pathlib.Path("/home/madam/temp/prototype/")
num_dataloader_workers = 0
batch_size = 10

mnist_n_in_g = 25
mnist_n_layers_1 = 5
mnist_n_out_g_1 = 24
mnist_n_layers_2 = 6
mnist_n_out_g_2 = 12
mnist_n_out_g_3 = 6
mnist_n_kernel_components = 5
eval_slize_size = 1024 * 1024 * 100
eval_img_n_sample_points = 50 * 50
eval_pc_n_sample_points = 1000 #set to 1000!
debug_image_size = (480, 270)
#debug_image_size = (531, 698)