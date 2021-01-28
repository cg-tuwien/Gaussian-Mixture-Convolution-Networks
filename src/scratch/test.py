import torch
import numpy as np

#torch.
# A = np.array([[1.]]).transpose()
# M = torch.tensor(A)


#M = np.array([[1.]]).transpose()
#M = torch.tensor(M)
# M. gives all, any, argmax.. completions if the next lines are commented

#M = np.array([[1.]])
#M = torch.tensor(M)
# M. gives partition, all, any, argmax..

#M = torch.tensor(np.array([[1.]]))
# M. gives nothing if the previous lines are commented

#M = torch.Tensor([[1.]])
# M. gives nothing if the previous lines are commented


def sq(v):
    return v*v

pi = 3.141592653589793238

def comp_eig_vals(A: torch.Tensor):
    p1 = sq(A[0][1]) + sq(A[0][2]) + sq(A[1][2])
    if p1 <= 0.000001:
        #// A is diagonal.
        eig1 = A[0][0]
        eig2 = A[1][1]
        eig3 = A[2][2]
    else:
        q = torch.trace(A) / 3
        p2 = sq(A[0][0] - q) + sq(A[1][1] - q) + sq(A[2][2] - q) + 2 * p1
        p = torch.sqrt(p2 / 6)
        B = (1 / p) * (A - q * torch.eye(3))
        r = torch.det(B) / 2

        #// In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        #// but computation error can leave it slightly outside this range.
        if r <= -1:
            phi = pi / 3
        elif r >= 1:
            phi = 0
        else:
            phi = torch.acos(r) / 3

        #// the eigenvalues satisfy eig3 <= eig2 <= eig1
        eig1 = q + 2 * p * torch.cos(phi)
        eig3 = q + 2 * p * torch.cos(phi + (2 * pi / 3))
        eig2 = 3 * q - eig1 - eig3         #// since trace(A) = eig1 + eig2 + eig3;
    return torch.tensor([eig1, eig2, eig3])


def norm_and_scale_longest_column(A: torch.Tensor):
    lengths = A.norm(dim=1)
    length, index_max = torch.max(lengths, 0)
    return A[index_max][:] / length

def comp_eig_vecs(A: torch.Tensor, eigvals: torch.Tensor):
    eigvecs0 = (A - eigvals[1] * torch.eye(3)) @ (A - eigvals[2] * torch.eye(3))
    eigvecs1 = (A - eigvals[0] * torch.eye(3)) @ (A - eigvals[2] * torch.eye(3))
    eigvecs2 = (A - eigvals[0] * torch.eye(3)) @ (A - eigvals[1] * torch.eye(3))

    eigvec0 = norm_and_scale_longest_column(eigvecs0).view(-1, 1)
    eigvec1 = norm_and_scale_longest_column(eigvecs1).view(-1, 1)
    eigvec2 = norm_and_scale_longest_column(eigvecs2).view(-1, 1)

    return torch.cat([eigvec0, eigvec1, eigvec2], dim=1)

A = torch.tensor([[1.4, 0.3, 0.8], [0.3, 2.3, 0.7], [0.8, 0.7, 1.9]])
print(torch.eig(A, True))
print(comp_eig_vals(A))
print(comp_eig_vecs(A, comp_eig_vals(A)))