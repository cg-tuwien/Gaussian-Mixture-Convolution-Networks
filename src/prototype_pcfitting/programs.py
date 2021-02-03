import os
import gc
import time

import torch
from typing import List, Tuple, Optional

from gmc import mixture, mat_tools
import gmc.io as gmio
from prototype_pcfitting import GMMGenerator, Scaler, PCDatasetIterator, GMLogger, ErrorFunction, data_loading
from prototype_pcfitting.generators.em_tools import EMTools
import prototype_pcfitting.pc_dataset_iterator


def execute_fitting2(training_name: str, dataset: prototype_pcfitting.pc_dataset_iterator.DatasetIterator,
                     generators: List[GMMGenerator], generator_identifiers: List[str],
                     gengmm_path: str,formats: List[str] = None, log_path: str = None,
                     scaling_active: bool = False, scaling_interval: Tuple[float, float] = (-50.0, 50.0),
                     log_positions: int = 0, log_loss_console: int = 0,
                     log_loss_tb: int = 0, log_rendering_tb: int = 0, log_gm: int = 0,
                     log_n_gaussians: int = 0, continuing: bool = True, verbosity: int = 2):
    # ---- GMM FITTING ----
    if formats is None:
        formats = [".gma.ply"]

    # Create Dataset Iterator and Scaler
    scaler = Scaler(active=scaling_active, interval=scaling_interval)

    # Iterate over Dataset
    i = 1
    while dataset.has_next():
        if verbosity > 1:
            print("-----------------------------------------------------------------------------------------")

        # Next Batch
        batch, names = dataset.next_batch()
        if verbosity > 0:
            print(f"Dataset Batch {i}: {batch.shape}, Remaining Batches: {dataset.remaining_batches_count()}")

        # Scale down
        scaler.set_pointcloud_batch(batch)
        batch_scaled = scaler.scale_pc(batch)

        for j in range(len(generators)):
            gc.collect()
            torch.cuda.empty_cache()
            if verbosity > 1:
                print(generator_identifiers[j], "on", names)

            if training_name is not None:
                gen_id = training_name + "/" + generator_identifiers[j]
            else:
                gen_id = generator_identifiers[j]

            if continuing and os.path.exists(os.path.join(gengmm_path, gen_id, f"{names[-1]}.gma.ply")):
                if verbosity > 0:
                    print("Skipped - already exists!")
                continue

            # Create Logger
            logger = GMLogger(names=names, log_prefix=gen_id, log_path=log_path,
                              log_positions=log_positions, gm_n_components=log_n_gaussians,
                              log_loss_console=log_loss_console, log_loss_tb=log_loss_tb,
                              log_rendering_tb=log_rendering_tb, log_gm=log_gm, pointclouds=batch, scaler=scaler, verbosity=verbosity)
            generators[j].set_logging(logger)

            # Generate GMM
            gmbatch, gmmbatch = generators[j].generate(batch_scaled)
            gmbatch = gmbatch.float()
            gmmbatch = gmmbatch.float()

            # Save resulting GMs
            data_loading.save_gms(scaler.unscale_gm(gmbatch), scaler.unscale_gmm(gmmbatch),
                                  os.path.join(gengmm_path, gen_id), names, formats)

            sumweights = mixture.weights(gmmbatch).sum(dim=2).squeeze(1)
            ones = sumweights.gt(0.99) & sumweights.lt(1.01)
            if not ones.all():
                if verbosity > 0:
                    print(f"Generator created an invalid mixture (sum of weights not 1) for one of {names} ({gen_id})!")

            # Terminate Logging
            logger.finalize()

        i += 1

    print("Done")


def execute_fitting(training_name: str, n_points: int, batch_size: int, generators: List[GMMGenerator],
                    generator_identifiers: List[str], model_path: Optional[str], genpc_path: str, gengmm_path: str,
                    formats: List[str] = None, log_path: str = None, scaling_active: bool = False,
                    scaling_interval: Tuple[float, float] = (-50.0, 50.0),
                    log_positions: int = 0, log_loss_console: int = 0,
                    log_loss_tb: int = 0, log_rendering_tb: int = 0, log_gm: int = 0,
                    log_n_gaussians: int = 0, continuing: bool = True):
    dataset = prototype_pcfitting.pc_dataset_iterator.PCDatasetIterator(batch_size, n_points, genpc_path, model_path)
    execute_fitting2(training_name, dataset, generators, generator_identifiers, gengmm_path, formats, log_path,scaling_active, scaling_interval, log_positions, log_loss_console, log_loss_tb, log_rendering_tb, log_gm,
                     log_n_gaussians, continuing)


def execute_fitting_on_single_pcbatch(training_name: str, pcbatch: torch.Tensor, gengmm_path: str,
                                      log_path: str, n_gaussians: int, generators: List[GMMGenerator],
                                      generator_identifiers: List[str], log_positions: int = 0,
                                      log_loss_console: int = 0, log_loss_tb: int = 0, log_rendering_tb: int = 0,
                                      log_gm: int = 0, initialgmbatch: torch.Tensor = None,
                                      scaling_active: bool = False,
                                      scaling_interval: Tuple[float, float] = (-50.0, 50.0)):
    t = int(time.time())
    names = ["" + str(i + 1) for i in range(pcbatch.shape[0])]

    # scaler = None
    scaler = Scaler(active=scaling_active, interval=scaling_interval)
    scaler.set_pointcloud_batch(pcbatch.cuda())

    scaled_pc = scaler.scale_pc(pcbatch.cuda())
    # scaled_pc = pcbatch

    for j in range(len(generators)):
        gen_id = training_name + "/" + str(t) + "-" + generator_identifiers[j]

        print(gen_id)

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

        gmbatch = scaler.unscale_gm(gmbatch)
        gmmbatch = scaler.unscale_gmm(gmmbatch)

        # Save resulting GMs
        data_loading.save_gms(gmbatch, gmmbatch,
                              os.path.join(gengmm_path, gen_id), names)

        sumweights = mixture.weights(gmmbatch).sum(dim=2).squeeze(1)
        ones = sumweights.gt(0.99) & sumweights.lt(1.01)
        if not ones.all():
            print("Generator created an incomplete mixture (sum of weights not 1)!")

        # Terminate Logging
        logger.finalize()


def execute_evaluation(training_name: str, model_path: str, genpc_path: str, gengmm_path: str, n_points: int,
                       eval_points: int, generator_identifiers: List[str], error_functions: List[ErrorFunction],
                       error_function_identifiers: List[str], scaling_active: bool = False,
                       scaling_interval: Tuple[float, float] = (-50.0, 50.0)):
    # Create Dataset Iterator and Scaler
    dataset = PCDatasetIterator(1, n_points, genpc_path, model_path)  # Batch Size must be one!
    scaler = Scaler(active=scaling_active, interval=scaling_interval)

    # Iterate over GMs
    while dataset.has_next():
        print("-----------------------------------------------------------------------------------------")

        # Next Batch
        pc, names = dataset.next_batch()

        # Scale down
        scaler.set_pointcloud_batch(pc)
        pc_scaled = scaler.scale_pc(pc)

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
                gm = gmio.read_gm_from_ply(gm_path, False).cuda()
                gm = scaler.scale_gm(gm)
                # Evaluate using each error function
                print(name, " / ", gid)
                for j in range(len(error_functions)):
                    loss = error_functions[j].calculate_score_packed(pc_scaled, gm).item()
                    print("  ", error_function_identifiers[j], ": ", loss)
                covariances = mixture.covariances(gm)
                invcovs = mat_tools.inverse(covariances).contiguous()
                irelcovs = ~EMTools.find_valid_matrices(covariances, invcovs, True)
                print("Broken Covariances: ", (irelcovs.sum().item()))
                print("   Invalid Gaussians: ", (mixture.weights(gm).eq(0)).sum().item())
                print("   Valid Gaussians: ", (~mixture.weights(gm).eq(0)).sum().item())
                print("   Sum of Weights: ", (mixture.weights(mixture.convert_amplitudes_to_priors(gm)).sum()).item())

    print("Done")


def quick_evaluation(pc_path: str, gm_path: str, is_model: bool, error_function: ErrorFunction,
                     scaling_active: bool = False,
                     scaling_interval: Tuple[float, float] = (-50.0, 50.0)):
    # Load pc
    pc = data_loading.load_pc_from_off(pc_path).cuda()

    # Scale down
    scaler = Scaler(active=scaling_active, interval=scaling_interval)
    scaler.set_pointcloud_batch(pc)
    pc_scaled = scaler.scale_pc(pc)

    # Load gm
    gm = gmio.read_gm_from_ply(gm_path, is_model).cuda()
    if is_model:
        gm = mixture.convert_priors_to_amplitudes(gm)
    covariances = mixture.covariances(gm)
    invcovs = mat_tools.inverse(covariances).contiguous()
    irelcovs = ~EMTools.find_valid_matrices(covariances, invcovs, True)
    print("Broken Covariances: ", (irelcovs.sum().item()))
    print("Invalid Gaussians: ", (mixture.weights(gm).eq(0)).sum().item())
    gm = scaler.scale_gm(gm)

    # Evaluate using each error function
    loss = error_function.calculate_score_packed(pc_scaled, gm).item()
    print("Loss: ", loss)


def quick_refine(training_name: str, pc_path: str, gm_in_path: str, gm_in_ismodel: bool, out_path: str,
                 generator: GMMGenerator, log_positions: int = 0, log_loss_console: int = 0, log_loss_tb: int = 0,
                 log_rendering_tb: int = 0, log_gm: int = 0, scaling_active: bool = False,
                 scaling_interval: Tuple[float, float] = (-50.0, 50.0)):
    # Load data
    pc = data_loading.load_pc_from_off(pc_path).cuda()
    gm = gmio.read_gm_from_ply(gm_in_path, gm_in_ismodel).cuda()

    # Scale down
    scaler = Scaler(active=scaling_active, interval=scaling_interval)
    scaler.set_pointcloud_batch(pc)
    pc_scaled = scaler.scale_pc(pc)
    gm_scaled = scaler.scale_gm(gm)

    # Create Logger
    logger = GMLogger(names=["log"], log_prefix=training_name, log_path=out_path, log_positions=log_positions,
                      log_loss_console=log_loss_console, log_loss_tb=log_loss_tb, log_rendering_tb=log_rendering_tb,
                      log_gm=log_gm, pointclouds=pc, scaler=scaler)
    generator.set_logging(logger)

    # Generate GMM
    gmbatch, gmmbatch = generator.generate(pc_scaled, gm_scaled)

    # Save resulting GMs
    data_loading.save_gms(scaler.unscale_gm(gmbatch), scaler.unscale_gmm(gmmbatch),
                          os.path.join(out_path, training_name), ["quick_refine"])

    # Terminate Logging
    logger.finalize()
