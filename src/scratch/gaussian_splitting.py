import torch

import gmc.mixture as gm
import gmc.render as gr
import matplotlib.pyplot as plt

m = torch.tensor([[0.01, -5, -5, 0.05, 0, 0, 0.05],
                  [0.7, 0, -2, 1, 0.7, 0.7, 0.6],
                  [1, 5, -2, 1, 0, 0, 0.5],
                  [0.2, 0, 2, 1, 0.7, 0.7, 0.5],
                  [1, 5, 2, 1, -0.4, -0.4, 0.5],
                  [1.6, 0, 7, 1, 0, 0, 1],
                  [0.8, 5, 7, 1, -0.9, -0.9, 1],
                  [0.01, 10, 10, 0.05, 0, 0, 0.05]]).view(1, 1, -1, 7)
gr.imshow(m, width=500, height=500)


# eigenvalues, eigenvectors = torch.linalg.eigh(gm.covariances(m))
#
# ranking = eigenvalues[:, :, :, :]
# selection = ranking > 0.499
# n = m[selection[..., 0]].view(1, 1, -1, 7)
# eig_vals = eigenvalues[selection[..., 0]].view(1, 1, -1, 2)
# eig_vecs = eigenvectors[selection[..., 0]].view(1, 1, -1, 2, 2)
#
# w = gm.weights(n) / 2
# p1 = gm.positions(n) + (torch.sqrt(eig_vals[..., -2].unsqueeze(-1)) * 0.6 * eig_vecs[..., -2, :])
# p2 = gm.positions(n) - (torch.sqrt(eig_vals[..., -2].unsqueeze(-1)) * 0.6 * eig_vecs[..., -2, :])
#
# c = (eig_vecs @ torch.diag_embed(eig_vals * torch.tensor([0.3, 1.0]).view(1, 1, 1, 2)) @ eig_vecs.transpose(-1, -2))
#
# m[selection[..., 0]] = gm.pack_mixture(w, p1, c)
# m = torch.cat((m, gm.pack_mixture(w, p2, c)), dim=2)
#
# gr.imshow(m, width=500, height=500)


displacement = 0.5
resize = 0.25


eigenvalues, eigenvectors = torch.linalg.eigh(gm.covariances(m))

ranking = eigenvalues[:, :, :, :]
selection = ranking > 0.499
number = selection[..., 1].sum() + selection[..., 0].sum() * 2

n = m[selection[..., 1]].view(1, 1, -1, 7)
eig_vals = eigenvalues[selection[..., 1]].view(1, 1, -1, 2)
eig_vecs = eigenvectors[selection[..., 1]].view(1, 1, -1, 2, 2)

w = gm.weights(n) / 2
p1 = gm.positions(n) + (torch.sqrt(eig_vals[..., -1].unsqueeze(-1)) * displacement * eig_vecs[..., -1, :])
p2 = gm.positions(n) - (torch.sqrt(eig_vals[..., -1].unsqueeze(-1)) * displacement * eig_vecs[..., -1, :])

c = (eig_vecs @ torch.diag_embed(eig_vals * torch.tensor([1.0, resize]).view(1, 1, 1, 2)) @ eig_vecs.transpose(-1, -2))

m[selection[..., 1]] = gm.pack_mixture(w, p1, c)
m = torch.cat((m, gm.pack_mixture(w, p2, c)), dim=2)

gr.imshow(m, width=500, height=500)





eigenvalues, eigenvectors = torch.linalg.eigh(gm.covariances(m))

ranking = eigenvalues[:, :, :, :]
selection = ranking > 0.499
number = selection[..., 1].sum() + selection[..., 0].sum() * 2

n = m[selection[..., 1]].view(1, 1, -1, 7)
eig_vals = eigenvalues[selection[..., 1]].view(1, 1, -1, 2)
eig_vecs = eigenvectors[selection[..., 1]].view(1, 1, -1, 2, 2)

w = gm.weights(n) / 2
p1 = gm.positions(n) + (torch.sqrt(eig_vals[..., -1].unsqueeze(-1)) * displacement * eig_vecs[..., -1, :])
p2 = gm.positions(n) - (torch.sqrt(eig_vals[..., -1].unsqueeze(-1)) * displacement * eig_vecs[..., -1, :])

c = (eig_vecs @ torch.diag_embed(eig_vals * torch.tensor([1.0, resize]).view(1, 1, 1, 2)) @ eig_vecs.transpose(-1, -2))

m[selection[..., 1]] = gm.pack_mixture(w, p1, c)
m = torch.cat((m, gm.pack_mixture(w, p2, c)), dim=2)

gr.imshow(m, width=500, height=500)


plt.show(block=True)


