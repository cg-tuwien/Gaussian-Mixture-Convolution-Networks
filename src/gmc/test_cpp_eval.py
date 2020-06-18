import unittest
import time
import torch.autograd

import cpp.gm_evaluate.gm_evaluate_inversed as cpp_inversed_eval
import mixture as gm

enable_output = True
test_precision_places = 5;

class CppEvalTest(unittest.TestCase):
    def _test_forward(self, mixture, xes, reference_values, test_fun, test_name):
        test_result = test_fun(mixture, xes)
        rmse = ((reference_values - test_result)**2).mean().sqrt().item()
        self.assertAlmostEqual(rmse, 0, places=test_precision_places, msg=f"RMSE {test_name}_cpu")

        test_result = test_fun(mixture.cuda(), xes.cuda()).cpu()
        rmse = ((reference_values - test_result)**2).mean().sqrt().item()
        self.assertAlmostEqual(rmse, 0, places=test_precision_places, msg=f"RMSE {test_name}_cuda")

    def test_forward(self):
        for n_batch in (1, 5, 10):
            for n_layers in (1, 4, 7):
                for n_components in (1, 6, 17, 50):
                    for n_xes in (1, 8, 23, 50):
                        for n_dims in (2, 3):
                            mixture = gm.generate_random_mixtures(n_batch, n_layers, n_components, n_dims)
                            mixture = gm.pack_mixture(gm.weights(mixture), gm.positions(mixture), gm.covariances(mixture).inverse().transpose(-2, -1))

                            xes = torch.rand([n_batch, n_layers, n_xes, n_dims])
                            reference = gm.old_evaluate_inversed(mixture, xes)
                            self._test_forward(mixture, xes, reference, gm.old_evaluate_inversed, "python");
                            self._test_forward(mixture, xes, reference, cpp_inversed_eval.apply, "cpp");

                            xes = torch.rand([1, 1, n_xes, n_dims])
                            reference = gm.old_evaluate_inversed(mixture, xes)
                            self._test_forward(mixture, xes, reference, gm.old_evaluate_inversed, "python");
                            self._test_forward(mixture, xes, reference, cpp_inversed_eval.apply, "cpp");

    def _test_backward(self, mixture, xes, reference_mixture_grad, reference_xes_grad, test_fun, test_name):
        mixture.requires_grad = True
        xes.requires_grad = True
        mixture.grad = None
        xes.grad = None
        forward_result = test_fun(mixture, xes)
        forward_result.sum().backward()
        rmse_mixture_grad = ((mixture.grad, reference_mixture_grad)**2).mean().sqrt().item()
        rmse_xes_grad = ((xes.grad, reference_xes_grad) ** 2).mean().sqrt().item()
        self.assertAlmostEqual(rmse_mixture_grad, 0, places=test_precision_places, msg=f"RMSE mixtures grad {test_name}_cpu")
        self.assertAlmostEqual(rmse_xes_grad, 0, places=test_precision_places, msg=f"RMSE xes grad {test_name}_cpu")

        forward_result = test_fun(mixture.cuda(), xes.cuda())
        forward_result.sum().backward()
        rmse_mixture_grad = ((mixture.grad.cpu(), reference_mixture_grad)**2).mean().sqrt().item()
        rmse_xes_grad = ((xes.grad.cpu(), reference_xes_grad) ** 2).mean().sqrt().item()
        self.assertAlmostEqual(rmse_mixture_grad, 0, places=test_precision_places, msg=f"RMSE mixtures grad {test_name}_cpu")
        self.assertAlmostEqual(rmse_xes_grad, 0, places=test_precision_places, msg=f"RMSE xes grad {test_name}_cpu")

    #todo: finish testing backward
    def test_backward(self):
        n_batch = 10
        n_layers = 5
        n_dims = 2
        mixture = gm.generate_random_mixtures(n_batch, n_layers, 60, n_dims)
        mixture = gm.pack_mixture(gm.weights(mixture), gm.positions(mixture), gm.covariances(mixture).inverse().transpose(-2, -1))
        xes = torch.rand([n_batch, n_layers, 50, n_dims])
        cuda_mixture = mixture.cuda()
        cuda_xes = xes.cuda()
        mixture.requires_grad = True;
        xes.requires_grad = True;
        cuda_mixture.requires_grad = True;
        cuda_xes.requires_grad = True;

        print("python cpu started")
        python_cpu = gm.old_evaluate_inversed(mixture, xes)
        python_cpu.sum().backward()
        python_cpu_mixture_grad = mixture.grad.clone()
        python_cpu_xes_grad = xes.grad.clone()

        print("python cuda started")
        python_cuda = gm.old_evaluate_inversed(cuda_mixture, cuda_xes)
        torch.cuda.synchronize()
        python_cuda.sum().backward()
        torch.cuda.synchronize()
        python_cuda_mixture_grad = cuda_mixture.grad.clone()
        python_cuda_xes_grad = cuda_xes.grad.clone()

        mixture.grad = None
        xes.grad = None
        cuda_mixture.grad = None
        cuda_xes.grad = None

        print("cpp cpu started")
        cpp_cpu = cpp_inversed_eval.apply(mixture, xes)
        cpp_cpu.sum().backward()
        cpp_cpu_mixture_grad = mixture.grad.clone()
        cpp_cpu_xes_grad = xes.grad.clone()

        print("cpp cuda started")
        cpp_cuda = cpp_inversed_eval.apply(cuda_mixture, cuda_xes)
        torch.cuda.synchronize()
        cpp_cuda.sum().backward()
        torch.cuda.synchronize()
        cpp_cuda_mixture_grad = cuda_mixture.grad.clone()
        cpp_cuda_xes_grad = cuda_xes.grad.clone()

        self.assertAlmostEqual(((python_cpu_mixture_grad - python_cuda_mixture_grad.cpu())**2).mean().sqrt().item(), 0, places=test_precision_places, msg = "RMSE mixture grad python_cpu vs python_cuda")
        self.assertAlmostEqual(((python_cpu_mixture_grad - cpp_cpu_mixture_grad)**2).mean().sqrt().item(), 0, places=test_precision_places, msg = "RMSE mixture grad python_cpu vs cpp_cpu")
        self.assertAlmostEqual(((python_cpu_mixture_grad - cpp_cuda_mixture_grad.cpu())**2).mean().sqrt().item(), 0, places=test_precision_places, msg = "RMSE mixture grad python_cpu vs cpp_cuda")
        self.assertAlmostEqual(((cpp_cpu_mixture_grad - cpp_cuda_mixture_grad.cpu())**2).mean().sqrt().item(), 0, places=test_precision_places, msg = "RMSE mixture grad cpp_cpu vs cpp_cuda")

        self.assertAlmostEqual(((python_cpu_xes_grad - python_cuda_xes_grad.cpu())**2).mean().sqrt().item(), 0, places=test_precision_places, msg = "RMSE xes grad python_cpu vs python_cuda")
        self.assertAlmostEqual(((python_cpu_xes_grad - cpp_cpu_xes_grad)**2).mean().sqrt().item(), 0, places=test_precision_places, msg = "RMSE xes grad python_cpu vs cpp_cpu")
        self.assertAlmostEqual(((python_cpu_xes_grad - cpp_cuda_xes_grad.cpu())**2).mean().sqrt().item(), 0, places=test_precision_places, msg = "RMSE xes grad python_cpu vs cpp_cuda")
        self.assertAlmostEqual(((cpp_cpu_xes_grad - cpp_cuda_xes_grad.cpu())**2).mean().sqrt().item(), 0, places=test_precision_places, msg = "RMSE xes grad cpp_cpu vs cpp_cuda")


# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
#
# mixture = gm.generate_random_mixtures(1, 1, 60, 3).to(torch.float64).cuda()
# xes = torch.rand([1, 1, 60, 3]).to(torch.float64).cuda()
#
# mixture.requires_grad = False;
# xes.requires_grad = True;
#
# exit()
# print(f"====== torch.autograd.gradcheck ======")
# test = torch.autograd.gradcheck(cpp_inversed_eval.apply, (mixture, xes), eps=1e-6, atol=1e-5, nondet_tol=1e-5)
# print(test)
