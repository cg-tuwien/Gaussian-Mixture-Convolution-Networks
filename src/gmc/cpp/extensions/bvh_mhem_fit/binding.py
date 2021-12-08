from torch.utils.cpp_extension import load
import os
import torch.autograd

from gmc.cpp.extensions.compile_flags import *
import gmc.inout

source_dir = os.path.dirname(__file__)


source_files = [source_dir + '/bindings.cpp', source_dir + '/implementation_dispatch.cpp', source_dir + '/../lbvh/bvh.cu', source_dir + '/../CpuSynchronisationPoint.cpp', source_dir + '/../pieces/pieces.cpp', source_dir + '/../pieces/matrix_inverse.cu', source_dir + '/../pieces/symeig.cu']
for dtype in ['float', 'double']:
    for reduction_n in [2, 4, 8, 16]:
        for ndims in [2, 3]:
            for direction in ['forward', 'backward']:
                source_files.append(source_dir + f"/implementation_{direction}_instances/template_instance_implementation_{direction}_{reduction_n}_{dtype}_{ndims}.cu")

cpp_binding = load('bvh_mhem_fit', source_files,
                   extra_include_paths=extra_include_paths,
                   verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags, extra_ldflags=["-lpthread"])


class BvhMhemFit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, target_mixture: torch.Tensor, n_components_fitting: int, reduction_n: int):
        if not target_mixture.is_contiguous():
            target_mixture = target_mixture.contiguous()

        fitting_mixture, bvh_nodes, bvh_attribs = cpp_binding.forward(target_mixture, n_components_fitting, reduction_n)
        ctx.save_for_backward(fitting_mixture, target_mixture, bvh_nodes, bvh_attribs, torch.tensor(n_components_fitting), torch.tensor(reduction_n))
        return fitting_mixture

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        fitting_mixture, target_mixture, bvh_nodes, bvh_attribs, n_components_fitting, reduction_n = ctx.saved_tensors
        grad_target_mixture = cpp_binding.backward(grad_output, fitting_mixture, target_mixture, bvh_nodes, bvh_attribs, n_components_fitting.item(), reduction_n.item())

        if torch.any(torch.isnan(grad_target_mixture)):
            print(f"grad_target_mixture: {grad_target_mixture}")
            print(f"target_mixture has nans: {torch.any(torch.isnan(target_mixture)).item()}")
            print(f"grad_output has nans: {torch.any(torch.isnan(grad_output)).item()}")
            print(f"number of nans in torch.isnan(grad_target_mixture): {torch.isnan(grad_target_mixture).sum(dim=-1).sum(dim=-1)}")
            print(f"grad_target_mixture.shape = {grad_target_mixture.shape}")
            print(f"target_mixture.shape = {target_mixture.shape}")
            print(f"grad_output.shape = {grad_output.shape}")
            print(f"reduction_n = {reduction_n}")
            print(f"n_components_fitting = {n_components_fitting}")
            gmc.inout.save(grad_output, "./bad_mixture_gradient.torch")
            gmc.inout.save(target_mixture, "./bad_mixture.torch")
            print(f"ahhh")
            exit(1)

        # assert not torch.any(torch.isinf(mixture))

        return grad_target_mixture, None, None


apply = BvhMhemFit.apply
