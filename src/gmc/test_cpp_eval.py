import unittest
import time
import torch.autograd

import gmc.cpp.extensions.evaluate_inversed.evaluate_inversed as cpp_inversed_eval
import gmc.mixture as gm

enable_output = True
test_precision_places = 4;

position_radius = 10
covariance_radius = 10

class CppEvalTest(unittest.TestCase):
    def _test_forward(self, mixture, xes, reference_values, test_fun, test_name):
        test_result = test_fun(mixture, xes)
        rmse = ((reference_values - test_result)**2).mean().sqrt().item()
        self.assertAlmostEqual(rmse, 0, places=test_precision_places, msg=f"RMSE {test_name}_cpu")

        test_result = test_fun(mixture.cuda(), xes.cuda()).cpu()
        rmse = ((reference_values - test_result)**2).mean().sqrt().item()
        self.assertAlmostEqual(rmse, 0, places=test_precision_places, msg=f"RMSE {test_name}_cuda")

    def test_forward(self):
        mixture = torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0]).view(1, 1, 1, -1)
        xes = torch.zeros(1, 1, 1, 2)
        reference = gm.old_evaluate_inversed(mixture, xes)
        self._test_forward(mixture, xes, reference, cpp_inversed_eval.apply, "cpp")
        for n_batch in (1, 5, 10):
            for n_layers in (1, 4, 7):
                for n_components in (1, 6, 17, 50):
                    for n_xes in (1, 8, 23, 50):
                        for n_dims in (2, 3):
                            mixture = gm.generate_random_mixtures(n_batch, n_layers, n_components, n_dims,
                                                                  pos_radius=position_radius, cov_radius=covariance_radius)
                            mixture = gm.pack_mixture(gm.weights(mixture), gm.positions(mixture), gm.covariances(mixture).inverse().transpose(-2, -1))

                            xes = torch.rand([n_batch, n_layers, n_xes, n_dims]) * position_radius * 2 - position_radius
                            reference = gm.old_evaluate_inversed(mixture, xes)
                            self._test_forward(mixture, xes, reference, cpp_inversed_eval.apply, "cpp")

                            xes = torch.rand([1, 1, n_xes, n_dims]) * position_radius * 2 - position_radius
                            reference = gm.old_evaluate_inversed(mixture, xes)
                            self._test_forward(mixture, xes, reference, cpp_inversed_eval.apply, "cpp")

    def _test_backward(self, mixture, xes, reference_mixture_grad, reference_xes_grad, test_fun, test_name):
        mixture.requires_grad = True
        xes.requires_grad = True
        mixture.grad = None
        xes.grad = None
        forward_result = test_fun(mixture, xes)
        forward_result.sum().backward()
        rmse_mixture_grad = ((mixture.grad - reference_mixture_grad)**2).mean().sqrt().item()
        rmse_xes_grad = ((xes.grad - reference_xes_grad) ** 2).mean().sqrt().item()
        self.assertAlmostEqual(rmse_mixture_grad, 0, places=test_precision_places, msg=f"RMSE mixtures grad {test_name}_cpu")
        self.assertAlmostEqual(rmse_xes_grad, 0, places=test_precision_places, msg=f"RMSE xes grad {test_name}_cpu")

        mixture.grad = None
        xes.grad = None
        forward_result = test_fun(mixture.cuda(), xes.cuda())
        forward_result.sum().backward()
        rmse_mixture_grad = ((mixture.grad.cpu() - reference_mixture_grad)**2).mean().sqrt().item()
        rmse_xes_grad = ((xes.grad.cpu() - reference_xes_grad) ** 2).mean().sqrt().item()
        self.assertAlmostEqual(rmse_mixture_grad, 0, places=test_precision_places, msg=f"RMSE mixtures grad {test_name}_cpu")
        self.assertAlmostEqual(rmse_xes_grad, 0, places=test_precision_places, msg=f"RMSE xes grad {test_name}_cpu")

    def test_backward(self):
        for n_batch in (1, 5, 10):
            for n_layers in (1, 4, 7):
                for n_components in (1, 6, 17, 50):
                    for n_xes in (1, 8, 23, 50):
                        for n_dims in (2, 3):
                            mixture = gm.generate_random_mixtures(n_batch, n_layers, n_components, n_dims,
                                                                  pos_radius=position_radius, cov_radius=covariance_radius)
                            mixture = gm.pack_mixture(gm.weights(mixture), gm.positions(mixture), gm.covariances(mixture).inverse().transpose(-2, -1))

                            xes = torch.rand([n_batch, n_layers, n_xes, n_dims]) * position_radius * 2 - position_radius
                            mixture.requires_grad = True
                            xes.requires_grad = True
                            forward = gm.old_evaluate_inversed(mixture, xes)
                            forward.sum().backward()
                            mixture_reference_grad = mixture.grad.clone()
                            xes_reference_grad = xes.grad.clone()
                            self._test_backward(mixture, xes, mixture_reference_grad, xes_reference_grad, cpp_inversed_eval.apply, "cpp")

                            xes = torch.rand([1, 1, n_xes, n_dims]) * position_radius * 2 - position_radius
                            mixture.grad = None
                            xes.grad = None
                            mixture.requires_grad = True
                            xes.requires_grad = True
                            forward = gm.old_evaluate_inversed(mixture, xes)
                            forward.sum().backward()
                            mixture_reference_grad = mixture.grad.clone()
                            xes_reference_grad = xes.grad.clone()
                            self._test_backward(mixture, xes, mixture_reference_grad, xes_reference_grad, cpp_inversed_eval.apply, "cpp")

    def test_gradcheck(self):
        print("test_gradcheck")
        eps = 1e-6
        # gradcheck takes a tuple of tensors as input, check if your gradient
        # evaluated with these tensors are close enough to numerical
        # approximations and returns True if they all verify this condition.
        # this test is quite unstable; it fails for the python ref implementation.
        # it does not work for the auto expansion when xes.n_batch < mixture.n_batch or the same with n_layers
        # (see doc: different indices to the same memory location)
        for n_batch in (3, ):
            for n_layers in (5, ):
                for n_components in (7,):
                    for n_xes in (11, ):
                        for n_dims in (2, 3):
                            print(f"n_batch={n_batch}, n_layers={n_layers}, n_components={n_components}, n_xes={n_xes}, n_dims={n_dims}")
                            mixture = gm.generate_random_mixtures(n_batch, n_layers, n_components, n_dims,
                                                                  pos_radius=position_radius, cov_radius=covariance_radius*4).to(torch.float64)
                            covariances = gm.covariances(mixture) + torch.eye(n_dims, dtype=torch.float32) * 1
                            mixture = gm.pack_mixture(gm.weights(mixture), gm.positions(mixture), covariances.inverse().transpose(-2, -1))
                            xes = torch.rand([n_batch, n_layers, n_xes, n_dims]).to(torch.float64) * position_radius * 2 - position_radius

                            mixture.requires_grad = True
                            xes.requires_grad = True
                            mixture.grad = None
                            xes.grad = None
                            test = torch.autograd.gradcheck(cpp_inversed_eval.apply, (mixture, xes), eps=eps, atol=1e-3, nondet_tol=1e-6)
                            self.assertTrue(test)

                            mixture = mixture.detach().cuda()
                            xes = xes.detach().cuda()
                            mixture.requires_grad = True
                            xes.requires_grad = True
                            mixture.grad = None
                            xes.grad = None
                            test = torch.autograd.gradcheck(cpp_inversed_eval.apply, (mixture, xes), eps=eps, atol=1e-3, nondet_tol=1e-6)
                            self.assertTrue(test)


if __name__ == '__main__':
    unittest.main()
