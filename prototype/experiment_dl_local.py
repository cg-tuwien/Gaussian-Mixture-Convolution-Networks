from experiment_gm_fitting import test_dl_fitting

n_iterations_r = 200002
n_iterations_c = 1000002
n_iterations_c = 550002


## local
#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 1024, 750], fully_layer_sizes=[128, 128, 64, 64, 32, 32], n_agrs=3, batch_norm=True, convolved_input=False, n_iterations=n_iterations_r)
#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 1024, 750], fully_layer_sizes=[128, 128, 64, 64, 32, 32], n_agrs=3, batch_norm=True, convolved_input=True, n_iterations=n_iterations_c)

#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 1024, 750], fully_layer_sizes=[128, 128, 128, 128, 128, 32], n_agrs=3, batch_norm=True, convolved_input=False, n_iterations=n_iterations_r)
#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 1024, 750], fully_layer_sizes=[128, 128, 128, 128, 128, 32], n_agrs=3, batch_norm=True, convolved_input=True, n_iterations=n_iterations_c)

#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 1024, 1024, 750], fully_layer_sizes=[128, 128, 128, 128, 128, 32], n_agrs=3, batch_norm=True, convolved_input=False, n_iterations=n_iterations_r)
#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 1024, 1024, 750], fully_layer_sizes=[128, 128, 128, 128, 128, 32], n_agrs=3, batch_norm=True, convolved_input=True, n_iterations=n_iterations_c)

#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 1024, 1024, 750], fully_layer_sizes=[256, 256, 256, 256, 256, 128], n_agrs=3, batch_norm=True, convolved_input=False, n_iterations=n_iterations_r)
test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 1024, 1024, 750], fully_layer_sizes=[256, 256, 256, 256, 256, 128], n_agrs=3, batch_norm=True, convolved_input=True, n_iterations=n_iterations_c)
