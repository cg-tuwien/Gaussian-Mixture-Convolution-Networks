import torch
import matplotlib.pyplot as plt
import gmc.mixture as mixture
epsilon = 1e-7

def main():
    #path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/covlog2"
    path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/negdet-logcovposdet5"
    #max = 61 #2
    #max = 57 #3
    #max = 70 #5
    #max = 77 #negdet4
    max = 60 #negdet5

    alldets_1 = []
    alldets_2 = []

    last = torch.load(f"{path}/covd-{max}.gm")
    cov, icov, det = reconstruct_matrix(last)
    idx = torch.nonzero(cov.det() <= 0)
    idx = torch.tensor([[0, 0, 24790]])
    for i in range(max + 1):
        data = torch.load(f"{path}/covd-{i}.gm")
        cov, icov, det = reconstruct_matrix(data)
        thecov = cov[idx[0,0],idx[0,1],idx[0,2],:,:]
        thedet = thecov.det()
        mydet = det[idx[0,0],idx[0,1],idx[0,2]]
        #print("data", data[idx[0,0],idx[0,1],idx[0,2],:])
        #print("cov", thecov)
        #print("det", thedet)
        #print("mydet", mydet)
        #print("--")
        alldets_1.append(thedet.item())
        alldets_2.append(mydet.item())
    plt.plot(alldets_1, 'r--', alldets_2, 'g--')    #rot = fehlerhafte, gruen = meine
    plt.yscale("log")
    plt.show()

def main_ply():
    path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/negdet-logcovposdet5"
    max = 60

    alldets = []

    lastgm = mixture.read_gm_from_ply(f"{path}/pcgmm-0-" + str(max).zfill(5) + ".ply", ismodel=False)
    lastcovs = mixture.covariances(lastgm)
    #idx = torch.nonzero(lastcovs.det() <= 0)
    idx = torch.tensor([[0, 0, torch.argmin(lastcovs.det())]])
    for i in range(max + 1):
        data = mixture.read_gm_from_ply(f"{path}/pcgmm-0-" + str(i).zfill(5) + ".ply", ismodel=False)
        covs = mixture.covariances(data)
        thecov = covs[idx[0,0],idx[0,1],idx[0,2],:,:]
        thedet = thecov.det()
        #print("data", data[idx[0,0],idx[0,1],idx[0,2],:])
        #print("cov", thecov)
        print("det", thedet)
        #print("mydet", mydet)
        #print("--")
        alldets.append(thedet.item())
    plt.plot(alldets, 'r--')    #rot = fehlerhafte, gruen = meine
    plt.yscale("log")
    plt.show()


def reconstruct_matrix(data):
    cov_shape = data.shape
    cov_factor_mat_rec = torch.zeros((cov_shape[0], cov_shape[1], cov_shape[2], 3, 3)).to(data.device)
    cov_factor_mat_rec[:, :, :, 0, 0] = torch.abs(data[:, :, :, 0]) + epsilon
    cov_factor_mat_rec[:, :, :, 1, 1] = torch.abs(data[:, :, :, 1]) + epsilon
    cov_factor_mat_rec[:, :, :, 2, 2] = torch.abs(data[:, :, :, 2]) + epsilon
    cov_factor_mat_rec[:, :, :, 1, 0] = data[:, :, :, 3]
    cov_factor_mat_rec[:, :, :, 2, 0] = data[:, :, :, 4]
    cov_factor_mat_rec[:, :, :, 2, 1] = data[:, :, :, 5]
    covariances = cov_factor_mat_rec @ cov_factor_mat_rec.transpose(-2, -1)
    cov_factor_mat_rec_inv = cov_factor_mat_rec.inverse()
    inversed_covariances = cov_factor_mat_rec_inv.transpose(-2, -1) @ cov_factor_mat_rec_inv
    # numerically better way of calculating the determinants
    determinants = torch.pow(cov_factor_mat_rec[:, :, :, 0, 0], 2) * \
                   torch.pow(cov_factor_mat_rec[:, :, :, 1, 1], 2) * \
                   torch.pow(cov_factor_mat_rec[:, :, :, 2, 2], 2)
    return covariances, inversed_covariances, determinants

main()