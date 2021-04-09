
import pcfitting.data_loading as data_loading
import gmc.mixture
import gmc.inout
import gmc.cpp.gm_vis.gm_vis as gm_vis
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # Backend Qt5Agg makes problems when using GMVis, so rather use TkAgg or something else
import time

# Example Data. One pointcloud and one mixture, although the same could be done with batches.
pointcloud = data_loading.load_pc_from_off("sample-pc.off")
mixture = gmc.inout.read_gm_from_ply("sample-gm.ply")
negmixture = gmc.inout.read_gm_from_ply("sample-neg-gm.ply")

###############################################################
#
# This file demonstrates the usage of gm_vis in different ways.
# The onetime functions provide the easiest access, however they
# take significantly longer than reusing an existing GMVisualizer-object.
# Here are taken measurements for each example-operation:
# - onetime_density_render (normal):        ~180ms
# - onetime_density_render (logarithmic):   ~180ms
# - onetime_ellipsoids_render:              ~175ms
# - creation of reusable GMVis-object:      ~117ms
# - manual density rendering (logarithmic): ~ 32ms
# - manuel ellipsoids rendering:            ~ 17ms
# - double rendering of pc/positions+dens:  ~ 45ms
# - finalization of GMvis-object:           ~ 32ms
# The first time we use the gm_vis-functionality, we have an
# additional time overhead of ~270ms.
#
###############################################################

# Extra measurement for first
# start = time.time()
# gmv = gm_vis.GMVisualizer(False, 500, 500)
# end = time.time()
# print("Time for first vis creation: ", (end-start))
# gmv.finish()

###
# In the first part of this file, the usage of GMVis is displayed.
# The second part displays the results.
# All render functions return an np.array of size (batch_size, n_rendermethods, height, width, 4).
###

# Onetime Density Render
# start = time.time()
res_onetime_density = gm_vis.onetime_density_render(500, 500, mixture)[0, 0]
# end = time.time()
# print("Time for calling res_onetime_density: ", (end-start))

# Onetime Density Render (Log-Scale)
# start = time.time()
res_onetime_denslog = gm_vis.onetime_density_render(500, 500, mixture, True)[0, 0]
# end = time.time()
# print("Time for calling res_onetime_denslog: ", (end-start))

# Onetime Ellipsoids Render
# start = time.time()
res_onetime_ellipsoids = gm_vis.onetime_ellipsoids_render(500, 500, mixture)[0, 0]
# end = time.time()
# print("Time for calling res_onetime_ellipsoids: ", (end-start))

# GMVisualizer object for non-onetime examples
# start = time.time()
vis = gm_vis.GMVisualizer(False, 500, 500)
# end = time.time()
# print("Time for creating GMVisualizer: ", (end-start))

# Logarithmic Density Rendering on vis-object
# start = time.time()
vis.set_camera_auto(True)
vis.set_density_rendering(True)
vis.set_density_logarithmic(True)
vis.set_gaussian_mixtures(mixture.detach().cpu())
res_manual_density = vis.render()[0, 0]
# end = time.time()
# print("Time for manual density", (end-start))

# Ellipsoid Rendering on vis-object
# start = time.time()
vis.set_density_rendering(False) # camera position has been set in previous example
vis.set_ellipsoids_pc_rendering(True, True, True)
vis.set_ellipsoids_colormode(gm_vis.GmVisColoringRenderMode.COLOR_WEIGHT)
vis.set_ellipsoids_rangemode(gm_vis.GmVisColorRangeMode.RANGE_MINMAX)
vis.set_gaussian_mixtures(mixture.detach().cpu()) # We could skip this, as we've done this in the previous example
res_manual_ellipsoids = vis.render()[0, 0]
# end = time.time()
# print("Time for manual ellipsoids", (end-start))

vis.set_ellipsoids_pc_rendering(False, True, False)
vis.set_pointclouds(pointcloud.detach().cpu())
res_pc_only = vis.render()[0, 0]

# Rendering of GM-Positions (colored by amp) + Pointcloud
# and Density (manual minmax values) together, manual camera position
# start = time.time()
vis.set_camera_lookat((33.4439, 18.4841, 60.3351), (0, -1e-05, 0.0076), (0, 1, 0))
vis.set_ellipsoids_pc_rendering(False, False, True)
vis.set_positions_rendering(True, pointcloud=True)
vis.set_positions_colormode(gm_vis.GmVisColoringRenderMode.COLOR_AMPLITUDE)
vis.set_positions_rangemode(gm_vis.GmVisColorRangeMode.RANGE_MINMAX)
vis.set_density_rendering(True)
vis.set_density_logarithmic(False)                  # False per default, but we used it before
vis.set_density_range_manual(0.005e-2, 0.0137e-2)   # reactivate auto with set_density_range_auto
vis.set_pointclouds(pointcloud.detach().cpu())
vis.set_gaussian_mixtures(mixture.detach().cpu())   # We could skip this, as we've done this in the previous example
rendering = vis.render()
res_pc_positions_manual_camera = rendering[0, 0]
res_density_manual_camera = rendering[0, 1]
# end = time.time()
# print("Time for manual double Rendering: ", (end-start))

# When being done using a GMVisualizer object, finish() should be called for clean exit!
# start = time.time()
vis.finish()
# end = time.time()
# print("Time for calling finish(): ", (end-start))

# Bonus: Rendering of Mixture with negative Gaussians
res_negative_density = gm_vis.onetime_density_render(500, 500, negmixture)[0, 0]

# Display of Renderings

f1 = plt.figure("Onetime Density Render")
f1.figimage(res_onetime_density)
plt.show()

f2 = plt.figure("Onetime Density Render (Log-Scale)")
f2.figimage(res_onetime_denslog)
plt.show()

f3 = plt.figure("Onetime Ellipsoids Render")
f3.figimage(res_onetime_ellipsoids)
plt.show()

f4 = plt.figure("Manual Density Render")
f4.figimage(res_manual_density)
plt.show()

f5 = plt.figure("Manual Ellipsoids Render")
f5.figimage(res_manual_ellipsoids)
plt.show()

fX = plt.figure("Only PC Render")
fX.figimage(res_pc_only)
plt.show()

f6 = plt.figure("Manual Positions Render (manual camera)")
f6.figimage(res_pc_positions_manual_camera)
plt.show()

f7 = plt.figure("Manual Density Render (manual camera)")
f7.figimage(res_density_manual_camera)
plt.show()

f8 = plt.figure("Negative Density Renderer")
f8.figimage(res_negative_density)
plt.show()
