import unittest
import gm
import torch
import numpy as np
import numpy.linalg as npla
import numpy.random as nprnd

class TestGM(unittest.TestCase):
    
    def test_xAx_3d(self):
        A = np.array(([[3., 1., 8.],
                       [1., 2., 3.],
                       [8., 3., 4.]]))
        B = np.array(([[5., 4., 2.],
                       [4., 7., 1.],
                       [2., 1., 6.]]))

        M = np.array([A[np.triu_indices(3)],
                      B[np.triu_indices(3)]]).transpose()
        M: torch.Tensor = torch.tensor(M)
        
        xes = nprnd.rand(3, 5)
        xesAxes = gm._xAx_withTriangleA(M[:][0])
        xesBxes = gm._xAx_withTriangleA(M[:][1])
        
        
        self.assertEqual(xesAxes.size().item(), 5)
        self.assertEqual(xesBxes.size().item(), 5)
        
        for i in range(5):
            
        
        self.assertAlmostEqual(npla.det(A), dets[0].item())
        self.assertAlmostEqual(npla.det(B), dets[1].item())
    
    def test_det(self):
        A = np.array(([[3., 1., 8.],
                       [1., 2., 3.],
                       [8., 3., 4.]]))
        B = np.array(([[5., 4., 2.],
                       [4., 7., 1.],
                       [2., 1., 6.]]))

        M = np.array([A[np.triu_indices(3)],
                      B[np.triu_indices(3)]]).transpose()
        M: torch.Tensor = torch.tensor(M)
                
        dets = gm._determinants(M)
        self.assertEqual(dets.size().item(), 2)
        self.assertAlmostEqual(npla.det(A), dets[0].item())
        self.assertAlmostEqual(npla.det(B), dets[1].item())
    
    def test_polynomMulRepeat(self):
        A: torch.Tensor = torch.tensor([[1, 2, 3, 4],
                                        [1, 1, 1, 1],
                                        [4, 2, 3, 1],
                                        [4, 2, 3, 1]], dtype=torch.float32)    # testing with col# = 3 is not propper.

        B: torch.Tensor = torch.tensor([[1, 2],
                                        [3, 6],
                                        [2, 1],
                                        [1, 2]], dtype=torch.float32)

        (Ap, Bp) = gm._polynomMulRepeat(A, B)

        self.assertEqual(Ap.size()[0], 4)
        self.assertEqual(Bp.size()[0], 4)
        self.assertEqual(Ap.size()[1], 8)
        self.assertEqual(Bp.size()[1], 8)

        AtimesB = Ap * Bp
        R = torch.sum(AtimesB, 1)
        
        self.assertAlmostEqual(R[0].item(), 30.)
        self.assertAlmostEqual(R[1].item(), 4*3 + 4*6)
        self.assertAlmostEqual(R[2].item(), 30.)
        self.assertAlmostEqual(R[3].item(), 30.)
        sorted = AtimesB.sort().values
        self.assertAlmostEqual(torch.sum(torch.abs(sorted[0]-sorted[2]), 0).item(), 0)
        self.assertAlmostEqual(torch.sum(torch.abs(sorted[3]-sorted[2]), 0).item(), 0)
        


if __name__ == '__main__':
    unittest.main()

