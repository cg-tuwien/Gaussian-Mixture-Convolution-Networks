from experiment_gm_fitting import test_dl_fitting

n_iterations_r = 50001
n_iterations_c = 50001

## local
# test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 250], fully_layer_sizes=[128, 128, 64, 32], n_agrs=1, batch_norm=True, convolved_input=False, n_iterations=n_iterations_r)
# test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 250], fully_layer_sizes=[128, 128, 64, 32], n_agrs=1, batch_norm=True, convolved_input=True, n_iterations=n_iterations_c)
# test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 750], fully_layer_sizes=[128, 128, 64, 32], n_agrs=3, batch_norm=True, convolved_input=False, n_iterations=n_iterations_r)
# test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 750], fully_layer_sizes=[128, 128, 64, 32], n_agrs=3, batch_norm=True, convolved_input=True, n_iterations=n_iterations_c)

## dl 2
#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 250], fully_layer_sizes=[128, 128, 64, 32], n_agrs=1, batch_norm=False, convolved_input=False, n_iterations=n_iterations_r, device='cuda:0')
#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 250], fully_layer_sizes=[128, 128, 64, 32], n_agrs=1, batch_norm=False, convolved_input=True, n_iterations=n_iterations_c, device='cuda:0')
#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 750], fully_layer_sizes=[128, 128, 64, 32], n_agrs=3, batch_norm=False, convolved_input=False, n_iterations=n_iterations_r, device='cuda:1')
#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 750], fully_layer_sizes=[128, 128, 64, 32], n_agrs=3, batch_norm=False, convolved_input=True, n_iterations=n_iterations_c, device='cuda:1')

test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 1000], fully_layer_sizes=[128, 128, 64, 32], n_agrs=1, batch_norm=True, convolved_input=False, n_iterations=n_iterations_r, device='cuda:2')
test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 1000], fully_layer_sizes=[128, 128, 64, 32], n_agrs=1, batch_norm=True, convolved_input=True, n_iterations=n_iterations_c, device='cuda:2')
#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 3000], fully_layer_sizes=[128, 128, 64, 32], n_agrs=3, batch_norm=True, convolved_input=False, n_iterations=n_iterations_r, device='cuda:3')
#test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 3000], fully_layer_sizes=[128, 128, 64, 32], n_agrs=3, batch_norm=True, convolved_input=True, n_iterations=n_iterations_c, device='cuda:3')

## dl_4
# test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 1000], fully_layer_sizes=[128, 128, 64, 32], n_agrs=1, batch_norm=False, convolved_input=False, n_iterations=n_iterations_r, device='cuda:0')
# test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 1000], fully_layer_sizes=[128, 128, 64, 32], n_agrs=1, batch_norm=False, convolved_input=True, n_iterations=n_iterations_c, device='cuda:0')
# test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 3000], fully_layer_sizes=[128, 128, 64, 32], n_agrs=3, batch_norm=False, convolved_input=False, n_iterations=n_iterations_r, device='cuda:1')
# test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 3000], fully_layer_sizes=[128, 128, 64, 32], n_agrs=3, batch_norm=False, convolved_input=True, n_iterations=n_iterations_c, device='cuda:1')
#
# test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 5000], fully_layer_sizes=[128, 128, 64, 32], n_agrs=1, batch_norm=True, convolved_input=False, n_iterations=n_iterations_r, device='cuda:2')
# test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 5000], fully_layer_sizes=[128, 128, 64, 32], n_agrs=1, batch_norm=True, convolved_input=True, n_iterations=n_iterations_c, device='cuda:2')
# test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 15000], fully_layer_sizes=[128, 128, 64, 32], n_agrs=3, batch_norm=True, convolved_input=False, n_iterations=n_iterations_r, device='cuda:3')
# test_dl_fitting(g_layer_sizes=[64, 128, 256, 512, 15000], fully_layer_sizes=[128, 128, 64, 32], n_agrs=3, batch_norm=True, convolved_input=True, n_iterations=n_iterations_c, device='cuda:3')
