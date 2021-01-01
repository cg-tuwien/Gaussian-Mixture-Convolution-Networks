import os
import gc
import time

import torch
from typing import List

from gmc import mixture
from prototype_pcfitting import GMMGenerator, PCDatasetIterator, Scaler, GMLogger, ErrorFunction, data_loading


def execute_fitting(training_name: str, model_path: str, genpc_path: str, gengmm_path: str, log_path: str,
                    n_points: int, n_gaussians: int, batch_size: int, generators: List[GMMGenerator],
                    generator_identifiers: List[str], log_positions: int, log_loss_console: int,
                    log_loss_tb: int, log_rendering_tb: int, log_gm: int, log_seperate_directories: bool):
    # ---- GMM FITTING ----

    # Create Dataset Iterator and Scaler
    dataset = PCDatasetIterator(model_path, n_points, batch_size, genpc_path)
    scaler = Scaler()

    # Iterate over Dataset
    i = 1
    while dataset.has_next():
        print("-----------------------------------------------------------------------------------------")

        # Next Batch
        batch, names = dataset.next_batch()
        print(f"Dataset Batch {i}: {batch.shape}, Remaining Batches: {dataset.remaining_batches_count()}")

        # Scale down
        # scaler.set_pointcloud_batch(batch)
        scaler.set_pointcloud_batch_for_identity(batch)
        batch_scaled = scaler.scale_down_pc(batch)

        for j in range(len(generators)):
            gc.collect()
            torch.cuda.empty_cache()
            print(generator_identifiers[j], "on", names)

            gen_id = training_name + "/" + generator_identifiers[j]

            # Create Logger
            logger = GMLogger(names=names, log_prefix=gen_id, log_path=log_path,
                              log_positions=log_positions, gm_n_components=n_gaussians,
                              log_loss_console=log_loss_console, log_loss_tb=log_loss_tb,
                              log_rendering_tb=log_rendering_tb, log_gm=log_gm, pointclouds=batch, scaler=scaler,
                              log_seperate_directories=log_seperate_directories)
            generators[j].set_logging(logger)

            # Generate GMM
            gmbatch, gmmbatch = generators[j].generate(batch_scaled)
            gmbatch = gmbatch.float()
            gmmbatch = gmmbatch.float()

            # Save resulting GMs
            data_loading.save_gms(scaler.scale_up_gm(gmbatch), scaler.scale_up_gmm(gmmbatch),
                                  os.path.join(gengmm_path, gen_id), names)

            sumweights = mixture.weights(gmmbatch).sum(dim=2).squeeze(1)
            ones = sumweights.gt(0.99) & sumweights.lt(1.01)
            assert ones.all(), "Generator created an invalid mixture (sum of weights not 1)!"

            # Terminate Logging
            logger.finalize()

        i += 1

    print("Done")


def execute_fitting_on_single_pcbatch(training_name: str, pcbatch: torch.Tensor, gengmm_path: str,
                                                      log_path: str, n_gaussians: int, generators: List[GMMGenerator],
                                                      generator_identifiers: List[str], log_positions: int,
                                                      log_loss_console: int,
                                                      log_loss_tb: int, log_rendering_tb: int, log_gm: int,
                                                      initialgmbatch: torch.Tensor = None):
    t = int(time.time())
    names = ["" + str(i + 1) for i in range(pcbatch.shape[0])]

    # scaler = None
    scaler = Scaler()
    # scaler.set_pointcloud_batch(pcbatch.cuda())
    scaler.set_pointcloud_batch_for_identity(pcbatch.cuda())

    scaled_pc = scaler.scale_down_pc(pcbatch.cuda())
    # scaled_pc = pcbatch

    for j in range(len(generators)):
        gen_id = training_name + "/" + str(t) + "-" + generator_identifiers[j]

        # Create Logger
        logger = GMLogger(names=names, log_prefix=gen_id, log_path=log_path,
                          log_positions=log_positions, gm_n_components=n_gaussians,
                          log_loss_console=log_loss_console, log_loss_tb=log_loss_tb,
                          log_rendering_tb=log_rendering_tb, log_gm=log_gm, pointclouds=pcbatch, scaler=scaler)
        generators[j].set_logging(logger)

        # Generate GMM
        gmbatch, gmmbatch = generators[j].generate(scaled_pc, initialgmbatch)
        gmbatch = gmbatch.float()
        gmmbatch = gmmbatch.float()

        gmbatch = scaler.scale_up_gm(gmbatch)
        gmmbatch = scaler.scale_up_gmm(gmmbatch)

        # Save resulting GMs
        data_loading.save_gms(gmbatch, gmmbatch,
                              os.path.join(gengmm_path, gen_id), names)

        sumweights = mixture.weights(gmmbatch).sum(dim=2).squeeze(1)
        ones = sumweights.gt(0.99) & sumweights.lt(1.01)
        assert ones.all(), "Generator created an invalid mixture (sum of weights not 1)!"

        # Terminate Logging
        logger.finalize()


def execute_evaluation(training_name: str, model_path: str, genpc_path: str, gengmm_path: str, n_points: int,
                       eval_points: int, generator_identifiers: List[str], error_functions: List[ErrorFunction],
                       error_function_identifiers: List[str]):
    # Create Dataset Iterator and Scaler
    dataset = PCDatasetIterator(model_path, n_points, 1, genpc_path)  # Batch Size must be one!
    scaler = Scaler()

    # Iterate over GMs
    while dataset.has_next():
        print("-----------------------------------------------------------------------------------------")

        # Next Batch
        pc, names = dataset.next_batch()

        # Scale down
        scaler.set_pointcloud_batch(pc)
        pc_scaled = scaler.scale_down_pc(pc)

        # Sample, if necessary
        if eval_points != n_points:
            pc_scaled = data_loading.sample(pc_scaled, eval_points)

        name = names[0]
        # Iterate over generators
        for gid in generator_identifiers:
            # Get GM Path
            gm_path = os.path.join(gengmm_path, training_name, gid, name + ".gma.ply")
            if not os.path.exists(gm_path):
                print(name + " / " + gid + ": No GM found")
            else:
                gm = mixture.read_gm_from_ply(gm_path, False).cuda()
                gm = scaler.scale_down_gm(gm)
                # Evaluate using each error function
                for j in range(len(error_functions)):
                    loss = error_functions[j].calculate_score_packed(pc_scaled, gm).item()
                    print(name, " / ", gid, ". ", error_function_identifiers[j], ": ", loss)
                print("Invalid Gaussians: ", (mixture.weights(gm).eq(0)).sum().item())
                print("Sum of Weights: ", (mixture.weights(mixture.convert_amplitudes_to_priors(gm)).sum()))

    print("Done")


def quick_evaluation(pc_path: str, gm_path: str, is_model: bool, error_function: ErrorFunction):
    # Load pc
    pc = data_loading.load_pc_from_off(pc_path).cuda()

    # Scale down
    scaler = Scaler()
    #scaler.set_pointcloud_batch(pc)
    scaler.set_pointcloud_batch_for_identity(pc)
    pc_scaled = scaler.scale_down_pc(pc)

    # Load gm
    gm = mixture.read_gm_from_ply(gm_path, is_model).cuda()
    if is_model:
        gm = mixture.convert_priors_to_amplitudes(gm)
    print("Invalid Gaussians: ", (mixture.weights(gm).eq(0)).sum().item())
    gm = scaler.scale_down_gm(gm)

    # Evaluate using each error function
    loss = error_function.calculate_score_packed(pc_scaled, gm).item()
    print("Loss: ", loss)


def quick_refine(training_name: str, pc_path: str, gm_in_path: str, gm_in_ismodel: bool, out_path: str,
                 generator: GMMGenerator, log_positions: int, log_loss_console: int, log_loss_tb: int,
                 log_rendering_tb: int, log_gm: int):
    # Load data
    pc = data_loading.load_pc_from_off(pc_path).cuda()
    gm = mixture.read_gm_from_ply(gm_in_path, gm_in_ismodel).cuda()

    # Scale down
    scaler = Scaler()
    scaler.set_pointcloud_batch(pc)
    pc_scaled = scaler.scale_down_pc(pc)
    gm_scaled = scaler.scale_down_gm(gm)

    # Create Logger
    logger = GMLogger(names=["log"], log_prefix=training_name, log_path=out_path, log_positions=log_positions,
                      log_loss_console=log_loss_console, log_loss_tb=log_loss_tb, log_rendering_tb=log_rendering_tb,
                      log_gm=log_gm, pointclouds=pc, scaler=scaler)
    generator.set_logging(logger)

    # Generate GMM
    gmbatch, gmmbatch = generator.generate(pc_scaled, gm_scaled)

    # Save resulting GMs
    data_loading.save_gms(scaler.scale_up_gm(gmbatch), scaler.scale_up_gmm(gmmbatch),
                          os.path.join(out_path, training_name), ["quick_refine"])

    # Terminate Logging
    logger.finalize()
