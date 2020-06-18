import cpp.gm_vis.pygmvis as api
import torch

def density(mixture: torch.Tensor, width: int, height: int) -> None:
    mixture = mixture.detach().cpu()
    v = api.create_visualizer(False, width, height, isgmm=False)
    v.set_camera_auto(True)
    v.set_density_rendering(True, api.GMDensityRenderMode.ADDITIVE_ACC_PROJECTED)
    v.set_gaussian_mixtures(mixture, isgmm=False)
    rendering = v.render(0)
    v.finish()
    return rendering

