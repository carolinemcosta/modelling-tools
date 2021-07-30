import os
import numpy as np

import ecg_tools as et
import slurm_tools as st
import vcg_analysis as vg


def write_ekbatch_init_file(file_name, cv_lon, stim_node, tags, biv):
    """ Build and write init file for ekbatch simulation

    Args:
        file_name (string): script file name
        cv_lon (float): longitudinal CV
        stim_node (int): number of stimulus node
        tags (dict): tags definition for each mesh region
        biv (bool): consider bi-ventricular mesh if true

    Returns:
        writes out init file
    """

    init_file = file_name + ".init"
    if not os.path.isfile(init_file):
        # define CVs
        cv_trans = cv_lon * 0.45
        cv_bz = cv_trans * 0.5
        cv_lon_fec = cv_lon * 6.
        cv_bz_fec = cv_bz * 6.

        # define CV regions using tags and CV
        cv_reg_he = "{:d} {:f} {:f} {:f}".format(tags['he'], cv_lon, cv_trans, cv_trans)
        cv_reg_bz = "{:d} {:f} {:f} {:f}".format(tags['bz'], cv_bz, cv_bz, cv_bz)
        cv_reg_rv = "{:d} {:f} {:f} {:f}".format(tags['rv'], cv_lon, cv_trans, cv_trans)
        cv_reg_he_fec = "{:d} {:f} {:f} {:f}".format(tags['he_fec'], cv_lon_fec, cv_trans, cv_trans)
        cv_reg_bz_fec = "{:d} {:f} {:f} {:f}".format(tags['bz_fec'], cv_bz_fec, cv_bz_fec, cv_bz_fec)
        cv_reg_sc_fec = "{:d} {:f} {:f} {:f}".format(tags['sc_fec'], cv_bz_fec, cv_bz_fec, cv_bz_fec)
        cv_reg_rv_fec = "{:d} {:f} {:f} {:f}".format(tags['rv_fec'], cv_lon_fec, cv_trans, cv_trans)

        # define header
        header = "vf:1.000000 vs:1.000000 vn:1.000000 vPS:4.000000\nretro_delay:3.000000 antero_delay:10.000000"  # keep this fixed
        stim_def = "{:d} 0.00000".format(stim_node)  # TODO allow more stim nodes

        # create script string
        if biv:
            stim_region = "1 7"
            script = "\n".join([header, stim_region, stim_def,
                                cv_reg_he, cv_reg_bz, cv_reg_rv,
                                cv_reg_he_fec, cv_reg_bz_fec, cv_reg_sc_fec, cv_reg_rv_fec])
        else:
            stim_region = "1 5"
            script = "\n".join([header, stim_region, stim_def,
                                cv_reg_he, cv_reg_bz,
                                cv_reg_he_fec, cv_reg_bz_fec, cv_reg_sc_fec])

        # write script
        with open(init_file, 'w+') as f:
            f.write(script)


def create_ekbatch_cmd_line(ekbatch_exec, mesh_name, sim_dir, sim_id, stim_node, cv_list, biv, tags):
    """
    Run ekbatch locally simulation with all CVs in cv_vector
    Args:
        ekbatch_exec (str): name of ekbatch executable
        mesh_name (str): name of the mesh file
        sim_dir (str): name of simulation directory
        sim_id (str): name of simulation ID (folder)
        stim_node (int): node number of RV pacing location on BIV mesh septum
        cv_list (list): list with all CVs to simulate
        biv (bool): consider bi-ventricular mesh if true
        tags (dict): tags definition for each mesh region

    Returns:
        Writes command line on terminal and run simulation
    """

    # generate init file for each cv_lon
    files = list()
    for cv_lon in cv_list:
        file_name = "{}/{}_{:1.2f}".format(sim_dir, sim_id, cv_lon)
        write_ekbatch_init_file(file_name, cv_lon, stim_node, tags, biv)
        files.append(file_name)

    # build command line
    all_files = ",".join(files)
    if biv:
        cmd = "{} {} {} {:d},{:d},{:d},{:d},{:d},{:d},{:d}".format(ekbatch_exec, mesh_name, all_files,
                                                                   tags['he'], tags['bz'], tags['rv'],
                                                                   tags['he_fec'], tags['bz_fec'], tags['sc_fec'],
                                                                   tags['rv_fec'])
    else:
        cmd = "{} {} {} {:d},{:d},{:d},{:d},{:d}".format(ekbatch_exec, mesh_name, all_files,
                                                         tags['he'], tags['bz'],
                                                         tags['he_fec'], tags['bz_fec'], tags['sc_fec'])

    return cmd


def write_ekbatch_slurm(ekbatch_exec, mesh_name, sim_dir, sim_id, stim_node, cv_list, biv, tags,
                        job_name, script_name, n_cores, time_limit):
    # build header
    header = st.generate_header(n_cores, job_name, time_limit)

    # source modules from bashrc
    bash = st.source_bashrc()

    # build command line
    cmd = create_ekbatch_cmd_line(ekbatch_exec, mesh_name, sim_dir, sim_id, stim_node, cv_list, biv, tags)

    # write script
    script = "\n\n".join([header, bash, cmd])
    with open(script_name, 'w') as f:
        f.write(script)


def post_process_act(file_name):
    """
    Replace inf values in activation file with -1
    Args:
        file_name (str): name of activation file

    Returns:
        writes out new file with post processed act

    """

    act_file = file_name + ".dat"
    act_file_pp = file_name + "-pp.dat"

    if not os.path.isfile(act_file_pp):
        act = np.loadtxt(act_file, dtype=float)
        # replace inf values
        act = np.where(act > 1000., -1., act)
        np.savetxt(act_file_pp, act, delimiter='\n')


def compute_total_base_act(act_file, base_vtx_file):
    """
    Compute the total activation time at the base of the ventricle(s)
    Args:
        act_file (str): name of activation file
        base_vtx_file (str): name of file with base vertices (ext = .vtx)

    Returns:
        total_act (float): total activation time at the base

    """

    act_file_pp = act_file + "-pp.dat"
    total_act_file = act_file + "-pp.total_act"
    if not os.path.isfile(total_act_file):
        base_nodes = np.loadtxt(base_vtx_file, skiprows=2, dtype=int)
        act = np.loadtxt(act_file_pp)
        # get myo total activation
        total_act = np.amax(act[base_nodes])
        # save total act
        with open(total_act_file, 'w+') as f:
            f.write("{:f}".format(total_act))
    else:
        total_act = np.loadtxt(total_act_file, dtype=float)

    return total_act


def get_rv_pacing_location(mesh_name, uvc_dir, rv_septum_vtx):
    """
    Compute RV pacing location on BIV mesh using UVC
    Args:
        mesh_name (string): name of mesh file
        uvc_dir (string): name of directory with UVC files
        rv_septum_vtx (str): name of .vtx file with RV septum node(s)

    Returns:
        rv_pac_loc_node (ind): node number of RV pacing location on BIV mesh septum
    """

    pac_loc_vtx_file = uvc_dir + "rv_sept_pac_loc.vtx"
    pac_loc_pts_file = uvc_dir + "rv_sept_pac_loc.pts_t"
    if not os.path.isfile(pac_loc_pts_file):
        print("Computing BiV pacing location...\n")

        # load data
        mesh_pts = np.loadtxt(mesh_name + ".pts", skiprows=1, dtype=float)
        septum_nodes = np.loadtxt(rv_septum_vtx, skiprows=2, dtype=int)
        z_coord = np.loadtxt(uvc_dir + "COORDS_Z.dat", dtype=float)
        phi_coord = np.loadtxt(uvc_dir + "COORDS_PHI_ROT.dat", dtype=float)

        # get RV apex as RHO = 1, PHI = 0, and min(Z)
        idx = np.arange(z_coord.size)
        mid_septum = idx[np.logical_and(phi_coord > -0.01, phi_coord < 0.01)]
        mid_rv_septum = np.intersect1d(septum_nodes, mid_septum, assume_unique=True)
        rv_pac_loc_node = mid_rv_septum[z_coord[mid_rv_septum].argmin()]

        # write out pacing location files
        with open(pac_loc_vtx_file, "w") as f:
            f.write("1\nextra\n{:d}".format(rv_pac_loc_node))
        with open(pac_loc_pts_file, "w") as f:
            f.write("1\n1\n{:f} {:f} {:f}".format(mesh_pts[rv_pac_loc_node, 0], mesh_pts[rv_pac_loc_node, 1],
                                                  mesh_pts[rv_pac_loc_node, 2]))
    else:
        rv_pac_loc_node = np.loadtxt(pac_loc_vtx_file, skiprows=2, dtype=int)

    return rv_pac_loc_node


def get_rv_pacing_location_lv(mesh_name, uvc_dir):
    """
    Compute RV pacing location on LV mesh using UVC
    Args:
        mesh_name (string): name of mesh file
        uvc_dir (string): name of directory with UVC files

    Returns:
        rv_pac_loc_node (ind): node number of RV pacing location on LV mesh septum
    """

    pac_loc_vtx_file = uvc_dir + "rv_sept_pac_loc_lv.vtx"
    pac_loc_pts_file = uvc_dir + "rv_sept_pac_loc_lv.pts_t"
    if not os.path.isfile(pac_loc_pts_file):
        print("Computing LV pacing location...\n")

        # load mesh points
        mesh_pts = np.loadtxt(mesh_name + ".pts", skiprows=1, dtype=float)

        # load rv sept pacing location from BiV mesh
        biv_pac_loc_pts_file = uvc_dir + "rv_sept_pac_loc.pts_t"
        biv_pac_loc_node = np.loadtxt(biv_pac_loc_pts_file, skiprows=2, dtype=float)

        # find closest point in LV mesh
        rv_pac_loc_node = np.sum((mesh_pts - biv_pac_loc_node) ** 2, axis=1).argmin()

        # write out files
        with open(pac_loc_vtx_file, "w") as f:
            f.write("1\nextra\n{:d}".format(rv_pac_loc_node))
        with open(pac_loc_pts_file, "w") as f:
            f.write("1\n1\n{:f} {:f} {:f}".format(mesh_pts[rv_pac_loc_node, 0], mesh_pts[rv_pac_loc_node, 1],
                                                  mesh_pts[rv_pac_loc_node, 2]))
    else:
        rv_pac_loc_node = np.loadtxt(pac_loc_vtx_file, skiprows=2, dtype=int)

    return rv_pac_loc_node


def fit_cv_to_total_act(sim_dir, sim_id, cv_list, qrs_d, base_vtx_file):
    """
    Fit the CV from ekbatch simulations based using the simulated total activation time and experimental QRS duration
    Args:
        sim_dir (str): name of simulation directory
        sim_id (str): name of simulation ID (folder)
        cv_list (list): list with all CVs to simulate
        qrs_d (float): experimental QRS duration
        base_vtx_file (str): name of .vtx file with base nodes

    Returns:
        best_cv_lon (float): best fit longitudinal CV

    """

    fit_file_name = sim_dir + sim_id + "-total-act-fit.dat"
    if not os.path.isfile(fit_file_name):
        print("Computing best CV using total activation time")

        # initialize arrays
        total_act_list = np.zeros(len(cv_list))
        total_act_qrs_diff = np.zeros(len(cv_list))
        indices = np.arange(len(cv_list))

        # compute difference between the total activation for each CV and the experimental QRS duration
        for i in indices:
            # remove inf values
            file_name = sim_dir + sim_id + "-{:1.2f}".format(cv_list[i])
            post_process_act(file_name)

            # get base total act
            total_act = compute_total_base_act(file_name, base_vtx_file)

            # compare with QRSd
            total_act_qrs_diff[i] = abs(qrs_d - total_act)
            total_act_list[i] = total_act

        # get min difference - will return only the first occurrence
        min_diff_idx = np.argmin(total_act_qrs_diff)

        # best CVs and total act
        best_cv_lon = cv_list[min_diff_idx]
        best_cv_trans = 0.45 * best_cv_lon
        best_cv_bz = 0.5 * best_cv_trans
        best_total_act = total_act_list[min_diff_idx]

        # write total act fit to file
        with open(fit_file_name, 'w+') as f:
            f.write("{:f} {:f} {:f} {:f} {:f}\n".format(qrs_d, best_total_act, best_cv_lon, best_cv_trans, best_cv_bz))
    else:
        best_cv_lon = np.loadtxt(fit_file_name, usecols=2, dtype=float)

    return best_cv_lon


def fit_cv_to_qrs(sim_dir, sim_id, cv_list, data_qrs_d, sim_qrs_d):
    """

    Args:
        sim_dir (str): name of simulation directory
        sim_id (str): name of simulation ID (folder)
        cv_list (list): list with all CVs to simulate
        data_qrs_d (float): experimental QRS duration
        sim_qrs_d (list): simulated QRS duration

    Returns:
        best_cv_lon (float): best fit longitudinal CV
    """

    fit_file_name = sim_dir + sim_id + "-qrs-fit.dat"
    if not os.path.isfile(fit_file_name):
        print("Computing best CV using QRSd...")

        # get min difference - will return only the first occurrence
        qrs_diff = np.abs(sim_qrs_d - data_qrs_d)
        min_diff_idx = np.argmin(qrs_diff)

        # best CVs and QRSd
        best_cv_lon = cv_list[min_diff_idx]
        best_cv_trans = 0.45 * best_cv_lon
        best_cv_bz = 0.5 * best_cv_trans
        best_qrs_d = qrs_diff[min_diff_idx]

        # write total act fit to file
        with open(fit_file_name, 'w+') as f:
            f.write("{:f} {:f} {:f} {:f} {:f}\n".format(data_qrs_d, best_qrs_d, best_cv_lon, best_cv_trans, best_cv_bz))
    else:
        best_cv_lon = np.loadtxt(fit_file_name, usecols=2, dtype=float)

    return best_cv_lon


def compute_ecg_and_qrsd(sim_dir, sim_id, cv_list, vcg_thresh):
    """
    Compute the 12-lead ECG from simulated phie_recovery file and compute QRSd using VCG spatial velocity method
    Args:
        sim_dir (str): name of simulation directory
        sim_id (str): name of simulation ID (folder)
        cv_list (list): list with all CVs to simulate
        vcg_thresh (float): spatial velocity threshold for VCG-based QRSd calculation

    Returns:
        qrd_d (array): array with QRSd for each simulation. Shape=len(cv_list)
        flag (bool): array with boolean flag to check if any simulations are missing. Shape=len(cv_list)
    """

    print("Computing ECG and QRS duration...")

    # initialize arrays
    n_cv = len(cv_list)
    qrs_d = np.zeros((n_cv,), dtype=float)
    flag = np.ones((n_cv,), dtype=bool)

    # compute ECG and QRSd for each simulation
    for i in range(n_cv):
        cv_sim_id = "{}-{:1.2f}".format(sim_id, cv_list[i])
        phie_file = sim_dir + cv_sim_id + "/phie_recovery.igb"
        if os.path.isfile(phie_file):
            ecg_file = sim_dir + cv_sim_id + '/12-lead-ecg.csv'
            if not os.path.isfile(ecg_file):
                # compute ECG from electrode data
                elec_data, t_steps = et.get_electrodes_from_phie_rec_file(phie_file)
                ecg = et.convert_electrodes_to_ecg(elec_data)

                # save ECG
                ecg_file = sim_dir + cv_sim_id + "/12-lead-ecg.csv"
                et.write_ecg_to_data_file(ecg_file, ecg, t_steps)
            else:
                # TODO: check that the saved ECGs have the time steps too
                ecg, t_steps = et.read_ecg_from_data_file(ecg_file)

            # convert normalized ECG to VCG
            norm_ecg = et.normalize_ecg(ecg)
            vcg = vg.convert_ecg_to_vcg(norm_ecg)

            # TODO test that this works with this dt. Unit?
            # Compute QRSd using VCG method
            dt = t_steps[1] - t_steps[0]
            t_end = len(t_steps) * dt
            qrs_start, qrs_end, qrs_d[i] = vg.get_qrs_start_end(vcg, dt=dt, velocity_offset=2, low_p=40,
                                                                order=2,
                                                                threshold_frac_start=vcg_thresh,
                                                                threshold_frac_end=vcg_thresh, filter_sv=True,
                                                                t_end=t_end, matlab_match=False)

            # save QRSd to file
            qrsd_file = sim_dir + cv_sim_id + '/qrsd.dat'
            if not os.path.isfile(qrsd_file):
                np.savetxt(qrsd_file, [qrs_start, qrs_end, qrs_d[i]], delimiter=' ', fmt='%1.6f')

        else:  # if phie is missing return flag with false
            flag[i] = False

    return qrs_d, flag


def create_re_phie_cmd_line(carp_exec, n_cores, intra_mesh_name, sim_dir, sim_id, electrode_file, stim_file, tags):
    """
    Create command lines for Reaction Eikonal and Phie recovery simulations
    Args:
        carp_exec (str): (path +) name of carp (or carpentry) executable
        n_cores (int): number of cpu cores to run simulations
        intra_mesh_name (str): name of intracellular mesh
        sim_dir (str): name of simulation directory
        sim_id (str): name of simulation ID (folder)
        electrode_file (str): name of file with ECG electrode coordinates
        stim_file (str): name of stimulus file (extension .vtx)
        tags (dict): dictionary defining mesh tags

    Returns:
        re_cmd (str): reaction eikonal command line
        rec_cmd (str): phie recovery command line (used together with re_cmd)

    """
    # build command line
    imp = " ".join(["-num_imp_regions 1",
                    "-imp_region[0].num_IDs 7",
                    "-imp_region[0].ID[0] {}".format(tags['he']),
                    "-imp_region[0].ID[1] {}".format(tags['bz']),
                    "-imp_region[0].ID[2] {}".format(tags['rv']),
                    "-imp_region[0].ID[3] {}".format(tags['he_fec']),
                    "-imp_region[0].ID[4] {}".format(tags['bz_fec']),
                    "-imp_region[0].ID[5] {}".format(tags['sc_fec']),
                    "-imp_region[0].ID[6] {}".format(tags['rv_fec']),
                    "-imp_region[0].im TT2"
                    ])

    num = " ".join(["-diffusionOn 0",
                    "-mass_lumping 0",
                    "-tend 500.0",
                    "-dt 100.0",
                    "-spacedt 5"
                    ])

    stim = " ".join(["-num_stim 2",
                     "-stimulus[0].stimtype 8",
                     "-stimulus[0].data_file {}".format(stim_file),
                     "-stimulus[1].stimtype 8",
                     "-stimulus[1].duration 2.0"
                     ])

    re_cmd = "mpirun -np {:d} {} -meshname {} -simID {}/{} {} {} {}".format(n_cores, carp_exec, intra_mesh_name,
                                                                            sim_dir, sim_id,
                                                                            imp, num, stim)

    rec_cmd = " ".join(["-experiment 4",
                        "-dump_ecg_leads 1",
                        "-post_processing_opts 1",
                        "-phie_rec_ptf {}".format(electrode_file)
                        ])

    return re_cmd, rec_cmd


def create_re_stim_file(meshtool_exec, intra_mesh_name, sim_dir, sim_id):
    """
    Create stimulus file for reaction eikonal simulation
    Args:
        meshtool_exec (str): (path +) name of meshtool executable
        intra_mesh_name (str): name of intracellular mesh
        sim_dir (str): name of simulation directory
        sim_id (str): name of simulation ID (folder)

    """

    stim_file = sim_dir + sim_id + "-pp_i.dat"
    if not os.path.isfile(stim_file):
        # extract post processed activation file onto intracellular grid using meshtool
        act_file_pp = sim_dir + sim_id + "-pp.dat"
        os.system("{} extract data -submsh={} -msh_data={} -submsh_data={}".format(meshtool_exec, intra_mesh_name,
                                                                                   act_file_pp, stim_file))


def write_re_phie_slurm(carp_exec, n_cores, intra_mesh_name, sim_dir, sim_id, electrode_file, stim_file, tags, cv_list,
                        job_name, script_name, time_limit):
    # build header
    header = st.generate_header(n_cores, job_name, time_limit)

    # add array job command
    total_jobs = len(cv_list)
    if total_jobs > 1:
        header = "\n".join([header, st.add_job_array_options(n_cores, 0, total_jobs - 1)])

    # source modules from bashrc
    bash = st.source_bashrc()

    # create cv array
    cv_str = ""
    for cv_lon in cv_list:
        cv_str += "\"{:1.2f}\" ".format(cv_lon)
    cv_arr = "declare -a cv_list=({})".format(cv_str)

    # build simulation command line
    cv_sim_id = "{}-${{cv_list[$SLURM_ARRAY_TASK_ID]}}".format(sim_id)  # array job
    cv_stim_file = "{}-${{cv_list[$SLURM_ARRAY_TASK_ID]}}".format(stim_file)
    re_cmd, rec_cmd = create_re_phie_cmd_line(carp_exec, n_cores, intra_mesh_name, sim_dir, cv_sim_id, electrode_file,
                                              cv_stim_file, tags)

    # write script
    script = "\n\n".join([header, bash, cv_arr, re_cmd, re_cmd + rec_cmd])
    with open(script_name, 'w') as f:
        f.write(script)


def main():
    # TODO create function to check simulation files and directories
    # define tags
    tags = {'he': 2, 'bz': 6, 'rv': 9, 'he_fec': 202, 'bz_fec': 206, 'sc_fec': 205, 'rv_fec': 209}
    file_name = "test"
    cv_lon = 0.6
    stim_node = 0
    biv = True
    ekbatch_exec = "ekbatch"
    mesh_name = "pig"
    sim_dir = "."  # TODO this will need to be checked
    sim_id = "test_cmd"
    cv_list = [0.6, 0.7]

    write_ekbatch_init_file(file_name, cv_lon, stim_node, tags, biv)

    job_name = "job"
    script_name = "test_job.sh"
    n_cores = 256
    time_limit = 24
    write_ekbatch_slurm(ekbatch_exec, mesh_name, sim_dir, sim_id, stim_node, cv_list, biv, tags,
                        job_name, script_name, n_cores, time_limit)

    carp_exec = "carp.pt"
    electrode_file = "test_electrodes"
    stim_file = "rv_test"
    script_name = "test_job_re.sh"
    write_re_phie_slurm(carp_exec, n_cores, mesh_name, sim_dir, sim_id, electrode_file, stim_file, tags, cv_list,
                        job_name, script_name, time_limit)


if __name__ == "__main__":
    main()
