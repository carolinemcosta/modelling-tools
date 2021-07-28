import numpy as np
import sys
import os

# insert packages/modules from PYTHONPATH
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

import vcg_analysis as vg
import ECG_tools as et


def create_ekbatch_init_file(file_name, cv_lon, stim_node, tags, biv):
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
        stim_def = "{:d} 0.00000".format(stim_node)

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


def run_ekbatch(ekbatch_exec, mesh_name, sim_dir, sim_id, uvc_dir, rv_septum_vtx, cv_list, biv):
    """
    Run ekbatch locally simulation with all CVs in cv_vector
    Args:
        ekbatch_exec (str): name of ekbatch executable
        mesh_name (str): name of the mesh file
        sim_dir (str): name of simulation directory
        sim_id (str): name of simulation ID (folder)
        uvc_dir (str): name of directory with UVC files
        rv_septum_vtx (str): name of .vtx file with RV septum node(s)
        cv_list (list): list with all CVs to simulate
        biv (bool): consider bi-ventricular mesh if true

    Returns:
        Writes command line on terminal and run simulation
    """
    # read stim node
    if biv:
        stim_node = get_rv_pacing_location(mesh_name, uvc_dir, rv_septum_vtx)
    else:
        stim_node = get_rv_pacing_location_lv(mesh_name, uvc_dir)

    # define tags
    tags = {'he': 2, 'bz': 6, 'rv': 9, 'he_fec': 202, 'bz_fec': 206, 'sc_fec': 205, 'rv_fec': 209}

    # generate init file for each cv_lon
    files = list()
    for cv_lon in cv_list:
        file_name = sim_dir + sim_id + "-{:1.2f}".format(cv_lon)
        create_ekbatch_init_file(file_name, cv_lon, stim_node, tags, biv)
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
    # run sim
    print(cmd)
    os.system(cmd)


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
        best_total_act = total_act_list[min_diff_idx]

        # best CVs
        best_cv_lon = cv_list[min_diff_idx]
        best_cv_trans = 0.45 * best_cv_lon
        best_cv_bz = 0.5 * best_cv_trans

        # write total act fit to file
        with open(fit_file_name, 'w+') as f:
            f.write("{:f} {:f} {:f} {:f} {:f}\n".format(qrs_d, best_total_act, best_cv_lon, best_cv_trans, best_cv_bz))
    else:
        best_cv_lon = np.loadtxt(fit_file_name, usecols=2, dtype=float)

    return best_cv_lon

def main():
    print("test")


if __name__ == "__main__":
    main()
