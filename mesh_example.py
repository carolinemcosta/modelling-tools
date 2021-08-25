import os
from shutil import copyfile

from add_fec import label_fec
from generate_mesh_from_seg import write_cgal_config_file, extract_surface
from generate_mesh_from_seg import generate_cgal_multi_label_mesh
from generate_mesh_from_seg import smooth_myo_surface
from generate_mesh_from_seg import extract_sub_mesh
from generate_mesh_from_seg import convert_nrrd_to_inr


def interpolate_uvc(meshtool_bin, in_mesh_name, in_uvc_dir, out_mesh_name, out_uvc_dir):
    os.makedirs(out_uvc_dir, exist_ok=True)
    coord_names = ["COORDS_PHI_ROT.dat", "COORDS_RHO.dat", "COORDS_Z.dat", "COORDS_V.dat"]
    for coord_name in coord_names:
        if not os.path.isfile(os.path.join(out_uvc_dir, coord_name)):
            cmd = "{} interpolate nodedata -omsh={} -imsh={} -idat={}/{} -odat={}/{}".format(meshtool_bin,
                                                                                             out_mesh_name, in_mesh_name,
                                                                                             in_uvc_dir, coord_name,
                                                                                             out_uvc_dir, coord_name)
            os.system(cmd)


def extract_and_map_surfaces(meshtool_bin, smooth_mesh, myo_mesh, ex_dir, mapped_dir, surf_dir):
    if not (os.path.isdir(surf_dir) and os.path.isdir(mapped_dir)):
        # create directories
        os.makedirs(mapped_dir, exist_ok=False)
        os.makedirs(surf_dir, exist_ok=False)

        # extract surfaces
        base_surf_name = smooth_mesh + ".base"
        tag_op = "1:2,5,6,9"
        extract_surface(meshtool_bin, smooth_mesh, base_surf_name, tag_op)

        lv_endo_surf_name = smooth_mesh + ".lvendo"
        tag_op = "3:2,5,6"
        extract_surface(meshtool_bin, smooth_mesh, lv_endo_surf_name, tag_op)

        rv_endo_surf_name = smooth_mesh + ".rvendo"
        tag_op = "8:2,5,6,9"
        extract_surface(meshtool_bin, smooth_mesh, rv_endo_surf_name, tag_op)

        rv_septum_surf_name = smooth_mesh + ".rvsept"
        tag_op = "8:2,5,6"
        extract_surface(meshtool_bin, smooth_mesh, rv_septum_surf_name, tag_op)

        epi_surf_name = smooth_mesh + ".epi"
        tag_op = "7:2,5,6,9"
        extract_surface(meshtool_bin, smooth_mesh, epi_surf_name, tag_op)

        # map surfaces onto myo mesh
        cmd = "{} map -submsh={} -files={}.surf,{}.surf,{}.surf,{}.surf,{}.surf -outdir={}".format(
            meshtool_bin, myo_mesh, base_surf_name, lv_endo_surf_name, rv_endo_surf_name, epi_surf_name,
            rv_septum_surf_name, mapped_dir)
        os.system(cmd)

        # map vertices onto myo mesh
        cmd = "{} map -submsh={} -files={}.surf.vtx,{}.surf.vtx,{}.surf.vtx,{}.surf.vtx,{}.surf.vtx -outdir={}".format(
            meshtool_bin, myo_mesh, base_surf_name, lv_endo_surf_name, rv_endo_surf_name, epi_surf_name,
            rv_septum_surf_name, mapped_dir)
        os.system(cmd)

        # move surfaces files
        cmd = "mv {}/*.surf* {}".format(ex_dir, surf_dir)
        os.system(cmd)

        cmd = "mv {}/*.neubc {}".format(ex_dir, surf_dir)
        os.system(cmd)


def main():
    # binaries
    cgal_bin = "cgalmeshmultilabel"
    meshtool_bin = "meshtool"
    segconvert_bin = "segconvert"
    ekbatch_bin = "ekbatch"

    # directories
    base_dir = "/data2/KCL/Martin/"
    seg_dir = os.path.join(base_dir, "segs/high-res-smooth/")
    ex_dir = os.path.join(base_dir, "fit_cv_example/")
    uvc_dir = os.path.join(ex_dir, "UVC")
    ep_dir = "/home/cmc16/Dropbox/PigEPdata/ecg-data"
    fec_sim_dir = os.path.join(ex_dir, "fec/")
    mapped_dir = os.path.join(ex_dir, "mapped_surfaces/")
    surf_dir = os.path.join(ex_dir, "surfaces/")


    # resolutions
    mesh_res = 2.
    bath_res = 8.

    # pig number
    pig = 27

    # file names
    seg_name = os.path.join(seg_dir, "pig{:d}-biv-segModel-smooth-0.05-ff-1.0-base-bath".format(pig))
    mesh_name = os.path.join(ex_dir, "pig{:d}-biv-{:d}mm".format(pig, int(mesh_res)))
    config_name = mesh_name + "-config.yaml"

    # mesh/image tags
    tags = {'base': 1, 'he': 2, 'lv_pool': 3, 'sc': 5, 'bz': 6, 'bath': 7, 'rv_pool': 8, 'rv': 9,
            'he_fec': 202, 'bz_fec': 206, 'sc_fec': 205, 'rv_fec': 209}
    myo_tags = [tags['he'], tags['sc'], tags['bz'], tags['rv']]
    intra_tags = [tags['he'],  tags['bz'], tags['rv']]
    bath_tag = tags['bath']
    lv_pool_tag = tags['lv_pool']
    rv_pool_tag = tags['rv_pool']

    # fec
    fec_thick = 3

    # convert NRRD to INR format
    convert_nrrd_to_inr(segconvert_bin, seg_name + ".nrrd")

    # create configuration file for CGAL
    write_cgal_config_file(config_name, seg_name + ".inr", mesh_name, mesh_res, bath_res, bath_tag, rv_pool_tag, lv_pool_tag)

    # generate CGAL mesh
    generate_cgal_multi_label_mesh(cgal_bin, mesh_name, config_name)

    # smooth outer and inner surfaces - optional
    smooth_mesh = mesh_name + "-smooth"
    smooth_myo_surface(meshtool_bin, mesh_name, smooth_mesh, myo_tags)

    # remove bath
    myo_mesh = smooth_mesh + "-no-bath"
    extract_sub_mesh(meshtool_bin, smooth_mesh, myo_mesh, myo_tags)

    # extract and map surfaces
    extract_and_map_surfaces(meshtool_bin, smooth_mesh, myo_mesh, ex_dir, mapped_dir, surf_dir)

    # add FEC
    os.makedirs(fec_sim_dir, exist_ok=True)
    lv_endo_surf_name = "pig{:d}-biv-{:d}mm-smooth.lvendo.surf.vtx".format(pig, int(mesh_res))
    rv_endo_surf_name = "pig{:d}-biv-{:d}mm-smooth.rvendo.surf.vtx".format(pig, int(mesh_res))
    endo_vtx_list = ["{}/{}".format(mapped_dir, lv_endo_surf_name), "{}/{}".format(mapped_dir, rv_endo_surf_name)]
    sim_id = "endo_fec_sim"
    label_fec(meshtool_bin, ekbatch_bin, myo_mesh, fec_sim_dir, sim_id, tags, endo_vtx_list, fec_thick)
    fec_mesh_name = myo_mesh + "-fec"

    # interpolate UVC onto example mesh (not needed if already available)
    no_scar_mesh = "pig{:d}-biv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath-no-scar".format(pig)
    in_mesh_name = os.path.join(base_dir, "biv-meshes/{}".format(no_scar_mesh))
    in_uvc_dir = os.path.join(base_dir, "biv-meshes/uvc/{}.uvc/UVC".format(no_scar_mesh))
    interpolate_uvc(meshtool_bin, in_mesh_name, in_uvc_dir, fec_mesh_name, uvc_dir)

    # remove scar
    intra_mesh = fec_mesh_name + "_i"
    extract_sub_mesh(meshtool_bin, fec_mesh_name, intra_mesh, intra_tags)

    # copy electrodes file
    in_elec_file = os.path.join(base_dir,
                                "biv-meshes/pig{:d}-biv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath-fibres-electrodes.pts".format(pig))
    out_elec_file = os.path.join(ex_dir, "{}-electrodes.pts".format(fec_mesh_name))
    if not os.path.isfile(out_elec_file):
        copyfile(in_elec_file, out_elec_file)

    # copy QRSd file
    in_qrs_file = os.path.join(ep_dir, "pig{:d}-qrsd-vcg-mean-beat-norm.dat".format(pig))
    out_qrs_file = os.path.join(ex_dir, "pig{:d}-qrsd-vcg-mean-beat-norm.dat".format(pig))
    if not os.path.isfile(out_qrs_file):
        copyfile(in_qrs_file, out_qrs_file)


if __name__ == "__main__":
    main()
