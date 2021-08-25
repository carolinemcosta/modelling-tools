import os
from generate_mesh_from_seg import write_cgal_config_file
from generate_mesh_from_seg import generate_cgal_multi_label_mesh
from generate_mesh_from_seg import smooth_myo_surface
from generate_mesh_from_seg import extract_sub_mesh
from generate_mesh_from_seg import convert_nrrd_to_inr


def main():
    # binaries
    cgal_bin = "cgalmeshmultilabel"
    meshtool_bin = "meshtool"
    segconvert_bin = "segconvert"

    # directories
    base_dir = "/data2/KCL/Martin/"
    seg_dir = base_dir + "segs/high-res-smooth/"
    ex_dir = base_dir + "fit_cv_example/"

    # resolutions
    mesh_res = 2.
    bath_res = 8.

    # pig number
    pig = 27

    # file names
    seg_name = os.path.join(seg_dir, "pig{:d}-biv-segModel-smooth-0.05-ff-1.0-base-bath".format(pig))
    mesh_name = os.path.join(ex_dir, "pig{:d}-biv-{:d}mm".format(pig, int(mesh_res)))
    config_name = mesh_name + "-config.yaml"

    print(config_name)

    # mesh/image tags
    myo_tags = [2, 5, 6, 9]
    intra_tags = [2, 6, 9]
    bath_tag = 7
    lv_pool_tag = 3
    rv_pool_tag = 8

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

    # remove scar
    intra_mesh = myo_mesh + "_i"
    extract_sub_mesh(meshtool_bin, myo_mesh, intra_mesh, intra_tags)


if __name__ == "__main__":
    main()
