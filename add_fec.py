import numpy as np
import os
from shutil import copyfile


def create_init_file(init_file, endo_nodes, tags):
    # build init file
    myo_cv = 1.0

    # define CV regions using tags and CV
    cv_reg = ""
    for key in tags.keys():
        cv_reg += "{:d} {:f} {:f} {:f}\n".format(tags[key], myo_cv, myo_cv, myo_cv)

    # define header
    header = "vf:1.000000 vs:1.000000 vn:1.000000 vPS:4.000000\nretro_delay:3.000000 antero_delay:10.000000"  # keep this fixed

    # define stim
    n_stim = len(endo_nodes)
    stim_region = "{:d} {:d}".format(n_stim, len(tags))
    stim_def = ""
    for s in range(n_stim):
        stim_def += "{:d} 0.00000\n".format(endo_nodes[s])

    # build script
    script = "\n".join([header, stim_region, stim_def, cv_reg])

    # write script
    with open(init_file, 'w') as f:
        f.write(script)


def run_ekbatch(ekbatch_bin, mesh_name, tags, sim_dir, sim_id, endo_nodes):
    # create init file
    init_file = os.path.join(sim_dir, sim_id)
    ek_sol_dat = os.path.join(sim_dir, "{}.dat".format(sim_id))
    if not os.path.isfile(ek_sol_dat):
        create_init_file(init_file + ".init", endo_nodes, tags)

        # run sim
        keys = list(tags.keys())
        tags_def = "{:d}".format(tags[keys[0]])
        for key in keys[1::]:
            tags_def += ",{:d}".format(tags[key])

        cmd = "{} {} {} {}".format(ekbatch_bin, mesh_name, init_file, tags_def)
        print(cmd)
        os.system(cmd)


def label_fec(meshtool_bin, ekbatch_bin, mesh_name, sim_dir, sim_id, tags, endo_vtx_list, fec_thick):
    # load endo vertices
    endo_vtx = np.array([], dtype=int)
    for vtx_file in endo_vtx_list:
        vtx = np.loadtxt(vtx_file, skiprows=2, dtype=int)
        endo_vtx = np.concatenate((endo_vtx, vtx), axis=0)

    # run ekbatch
    run_ekbatch(ekbatch_bin, mesh_name, tags, sim_dir, sim_id, endo_vtx)

    # map EK onto element centres
    ek_sol_dat = os.path.join(sim_dir, "{}.dat".format(sim_id))
    ek_sol_elem_dat = os.path.join(sim_dir, "{}_elem.dat".format(sim_id))
    if not os.path.isfile(ek_sol_elem_dat):
        cmd = "{} interpolate node2elem -omsh={} -idat={} -odat={}".format(meshtool_bin, mesh_name, ek_sol_dat,
                                                                           ek_sol_elem_dat)
        os.system(cmd)

    # get ek solution on elements
    ek_sol = np.loadtxt(ek_sol_elem_dat, dtype=float)

    # get mesh elements and tags tags
    n_elems = np.loadtxt(mesh_name + ".elem", max_rows=1, dtype=int)
    elems = np.loadtxt(mesh_name + ".elem", skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=int)

    # label tags as fec (5th column)
    elems[ek_sol <= fec_thick, 4] += 200

    # write out new mesh
    fec_mesh_name = mesh_name + "-fec"
    copyfile(mesh_name + ".pts", fec_mesh_name + ".pts")
    copyfile(mesh_name + ".lon", fec_mesh_name + ".lon")
    with open(fec_mesh_name + ".elem", "w") as f:
        f.write("{:d}\n".format(n_elems))
        for e in range(n_elems):
            f.write(
                "Tt {:d} {:d} {:d} {:d} {:d}\n".format(elems[e, 0], elems[e, 1], elems[e, 2], elems[e, 3], elems[e, 4]))
