import os
import gc
import torch
from typing import List

from gmc import mixture
from prototype_pcfitting import GMMGenerator, PCDatasetIterator, Scaler, GMLogger, ErrorFunction, data_loading


def execute_fitting(training_name: str, model_path: str, genpc_path: str, gengmm_path: str, log_path: str,
                    n_points: int, n_gaussians: int, batch_size: int, generators: List[GMMGenerator],
                    generator_identifiers: List[str], log_positions: int, log_loss_console: int,
                    log_loss_tb: int, log_rendering_tb: int, log_gm: int):
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
        scaler.set_pointcloud_batch(batch)
        batch_scaled = scaler.scale_down_pc(batch)

        for j in range(len(generators)):
            gc.collect()
            torch.cuda.empty_cache()

            gen_id = training_name + "/" + generator_identifiers[j]

            # Create Logger
            logger = GMLogger(names=names, log_prefix=gen_id, log_path=log_path,
                              log_positions=log_positions, gm_n_components=n_gaussians,
                              log_loss_console=log_loss_console, log_loss_tb=log_loss_tb,
                              log_rendering_tb=log_rendering_tb, log_gm=log_gm, pointclouds=batch, scaler=scaler)
            generators[j].set_logging(logger)

            # Generate GMM
            gmbatch, gmmbatch = generators[j].generate(batch_scaled)

            # Save resulting GMs
            data_loading.save_gms(scaler.scale_up_gm(gmbatch), scaler.scale_up_gmm(gmmbatch),
                                  os.path.join(gengmm_path, gen_id), names)

            # Terminate Logging
            logger.finalize()

        i += 1

    print("Done")


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

    print("Done")


def quick_evaluation(pc_path: str, gm_path: str, is_model: bool, error_function: ErrorFunction):
    # Load pc
    pc = data_loading.load_pc_from_off(pc_path).cuda()

    # Scale down
    scaler = Scaler()
    scaler.set_pointcloud_batch(pc)
    pc_scaled = scaler.scale_down_pc(pc)

    # Load gm
    gm = mixture.read_gm_from_ply(gm_path, is_model).cuda()
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
