import numpy as np
import os

from fit_cv_to_qrsd import get_rv_pacing_location
from fit_cv_to_qrsd import create_ekbatch_cmd_line
from fit_cv_to_qrsd import fit_cv_to_total_act
from fit_cv_to_qrsd import create_re_stim_file
from fit_cv_to_qrsd import create_re_phie_cmd_line
from fit_cv_to_qrsd import compute_ecg_and_qrsd
from fit_cv_to_qrsd import fit_cv_to_qrs


def main():
    # example of parameterization pipeline

    # define executables
    ekbatch_bin = "ekbatch"
    carp_bin = "carp.pt"
    meshtool_bin = "meshtool"

    # define mesh tags
    tags = {'he': 2, 'bz': 6, 'rv': 9, 'he_fec': 202, 'bz_fec': 206, 'sc_fec': 205, 'rv_fec': 209}
    biv = True  # mesh type

    # pig number
    pig = 27

    # define directories
    base_dir = "/data2/KCL/Martin/"
    ex_dir = os.path.join(base_dir, "fit_cv_example/")
    uvc_dir = os.path.join(ex_dir, "UVC/")
    sim_dir = os.path.join(ex_dir, "simulations/")
    os.makedirs(sim_dir, exist_ok=True)
    sim_id = "pig{:d}_biv".format(pig)

    # define mesh files
    mesh_name = os.path.join(ex_dir, "pig{:d}-biv-2mm-smooth-no-bath".format(pig))
    intra_mesh_name = mesh_name + "_i"
    base_vtx_file = os.path.join(ex_dir, "mapped_surfaces/pig{:d}-biv-2mm-smooth.base.surf.vtx".format(pig))
    rv_septum_vtx = os.path.join(ex_dir, "mapped_surfaces/pig{:d}-biv-2mm-smooth.rvsept.surf.vtx".format(pig))
    electrode_file = mesh_name + "-electrodes"

    # define experimental QRSd file or value
    qrsd_exp_name = os.path.join(ex_dir, "pig{:d}-qrsd-vcg-mean-beat-norm.dat".format(pig))
    qrsd_exp = np.loadtxt(qrsd_exp_name, skiprows=2, dtype=float)  # usually the third line on the file

    # define cores
    n_cores = 12
    time_limit = 24  # for HPC only

    # run initial parameterization using total activation time
    cv_step = 0.01
    tact_cv_min = 0.36
    tact_cv_max = 0.96
    tact_cv_list = np.arange(tact_cv_min, tact_cv_max + cv_step, cv_step)

    stim_node = get_rv_pacing_location(mesh_name, uvc_dir, rv_septum_vtx)

    # run locally
    if not os.path.isfile(sim_dir + sim_id + "_{:1.2f}.dat".format(tact_cv_list[len(tact_cv_list)-1])):
        ekbatch_cmd = create_ekbatch_cmd_line(ekbatch_bin, mesh_name, sim_dir, sim_id, stim_node, tact_cv_list, biv, tags)
        print(ekbatch_cmd)
        os.system(ekbatch_cmd)

    # or run on TOM2 using array job
    # job_name = "job"
    # script_name = "array_ekbatch_job.sh"
    # write_ekbatch_slurm(ekbatch_bin, mesh_name, sim_dir, sim_id, stim_node, tact_cv_list, biv, tags,
    #                     job_name, script_name, n_cores, time_limit)

    best_total_act_cv = fit_cv_to_total_act(sim_dir, sim_id, tact_cv_list, qrsd_exp, base_vtx_file)
    print(best_total_act_cv)

    # # refine parameters using simulated QRSd
    # qrsd_interval = 0.3
    # qrsd_cv_min = max([best_total_act_cv - qrsd_interval, tact_cv_min])
    # qrsd_cv_max = min([best_total_act_cv + qrsd_interval, tact_cv_max])
    # qrsd_cv_list = np.arange(qrsd_cv_min, qrsd_cv_max + cv_step, cv_step)
    #
    # # run locally
    # for cv in qrsd_cv_list:
    #     cv_sim_id = "{}-{:1.2f}".format(sim_id, cv)
    #     stim_file = create_re_stim_file(meshtool_bin, intra_mesh_name, sim_dir, cv_sim_id)
    #     re_cmd = create_re_phie_cmd_line(carp_bin, n_cores, intra_mesh_name, sim_dir, cv_sim_id, electrode_file, stim_file, tags)
    #     os.system(re_cmd)
    #
    # # or run all on TOM2 using array job
    # # script_name = "array_job_re.sh"
    # # write_re_phie_slurm(carp_bin, n_cores, mesh_name, sim_dir, sim_id, electrode_file, stim_file, tags, tact_cv_list,
    # #                     job_name, script_name, time_limit)
    #
    # vcg_thresh = 0.3
    # qrsd_sim, flag = compute_ecg_and_qrsd(sim_dir, sim_id, qrsd_cv_list, vcg_thresh)
    # best_qrsd_cv = fit_cv_to_qrs(sim_dir, sim_id, qrsd_cv_list, qrsd_exp, qrsd_sim)
    #
    # print(best_qrsd_cv)


if __name__ == "__main__":
    main()
