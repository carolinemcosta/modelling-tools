import os


def write_cgal_config_file(script_name, image_name, mesh_name, myo_res, bath_res, bath_tag,
                           rv_pool_tag=-1, lv_pool_tag=-1, facet_angle=30, facet_distance=0.5,
                           edge_size=0.5, cell_radius_edge_ratio=1.5, n_threads=8):
    """ Creates CGAL configuration file for cgalmeshmultilabel, which is part of the cgalmesh packaged from the MUG

        Parameters:
          script_name (string): path+name of configuration file (extension: .yaml)
          image_name (string): path+name of input image
          mesh_name (string): path+name of output mesh
          myo_res (float): target resolution of myocardium
          bath_res (float): target resolution of surrounding bath
          bath_tag (int): image tag identifying bath
          rv_pool_tag (int): image tag identifying RV blood pool (if present, otherwise same as bath_tag)
          lv_pool_tag (int): image tag identifying LV blood pool (if present, otherwise same as bath_tag)
          facet_angle (int): CGAL's facet_angle. Default = 30
          facet_distance (float): CGAL's facet_distance. Default = 0.5
          edge_size (float): CGAL's edge_size. Default = 0.5
          cell_radius_edge_ratio (float): CGAL's cell_radius_edge_ratio. Default = 1.5
          n_threads (int): number of threads to use while generating mesh. Default = 8

        Returns:
          Writes out a yaml file with configuration parameters for cgalmeshmultilabel
    """

    if not os.path.isfile(script_name):
        # set RV and LV blood pool tags depending on arguments passed: if not passed, set as bath_tag
        if rv_pool_tag < 0:
            rv_pool_tag = bath_tag
        if lv_pool_tag < 0:
            lv_pool_tag = bath_tag

        # build dictionary with config parameters
        config = {"input_img": image_name,
                  "output_carp": mesh_name,
                  "output_vtk": "{}.vtk".format(mesh_name),
                  "facet_angle": facet_angle,
                  "facet_size": myo_res,
                  "facet_distance": facet_distance,
                  "cell_radius_edge_ratio": cell_radius_edge_ratio,
                  "cell_size": myo_res,
                  "edge_size": edge_size,
                  "bath_size": bath_res,
                  "bath_tag": bath_tag,
                  "rvp_tag": rv_pool_tag,
                  "lvp_tag": lv_pool_tag,
                  "tbb_threads": n_threads
                  }

        # write dictionary to text file
        with open(script_name, 'w') as f:
            for key, value in config.items():
                f.write('%s: %s\n' % (key, value))
            # print(config, file=f)


def convert_nrrd_to_inr(segconvert_bin, image_name):
    """ Converts NRRD image to INR format required by CGAL
      Parameters:
        segconvert_bin (string): path+name of segconvert binary
        image_name (string): path+name of input image in NRRD format. File ends in .nrrd
      Returns:
        Writes out new image file in INR format with .inr extension
    """

    inr_image = image_name.split(".nrrd")[0] + ".inr"

    if os.path.isfile(image_name) and not os.path.isfile(inr_image):
        print("Converting NRRD to INR format...")
        cmd = "{} {} {}".format(segconvert_bin, image_name, inr_image)
        os.system(cmd)


def generate_cgal_multi_label_mesh(cgal_bin, mesh_name, config_script):
    """ Generates CGAL mesh from a multi label segmentation.
        Uses cgalmeshmultilabel (cgalmesh package) and a configuration file (.yaml)
      Parameters:s
        cgal_bin (string): path+name of segconvert binary
        mesh_name (string): path+basename of output mesh
        config_script: name of .yaml configuration script for CGAL (create_CGAL_config_file)
      Returns:
        Writes out a CGAL mesh in both carp_txt and VTK formats
    """

    if not os.path.isfile(mesh_name + ".elem"):
        print("Generating CGAL mesh...")
        cmd = "{} {}".format(cgal_bin, config_script)
        print(cmd)
        os.system(cmd)


def smooth_myo_surface(meshtool_bin, mesh_name, smooth_mesh, myo_tags):
    """ Smooths the myocardium surface of a mesh using meshtool
      Parameters:
        meshtool_bin (string): path+name of meshtool binary
        mesh_name (string): path+basename of input mesh (expects carp_txt format)
        smooth_mesh (string): path+basename of output smooth mesh (writes out in carp_txt format)
        myo_tags (list): list of myocardium (including scar) mesh tags. Ex [1, 2, 3]
      Returns:
        Writes out new mesh files for smooth mesh in carp_txt format
    """

    if os.path.isfile(mesh_name + ".elem") and not os.path.isfile(smooth_mesh + ".elem"):
        print("Smoothing myocardium surface...")

        # extract surface surrounding the myocardium
        cmd = "{} extract surface -msh={} -surf={}-outer -op={} -ofmt=carp_txt".format(meshtool_bin,
                                                                                       mesh_name,
                                                                                       mesh_name,
                                                                                       ",".join(
                                                                                           str(t) for t in myo_tags))
        print(cmd)
        os.system(cmd)

        # smooth the surface surrounding the myocardium
        cmd = "{} smooth surface -msh={} -surf={}-outer -outmsh={} -ofmt=carp_txt -thr=0.1 -smth=0.15 -iter=50".format(
            meshtool_bin,
            mesh_name,
            mesh_name,
            smooth_mesh)
        os.system(cmd)


def extract_sub_mesh(meshtool_bin, mesh_name, output_mesh, tags_list):
    if os.path.isfile(mesh_name + ".elem") and not os.path.isfile(output_mesh + ".elem"):
        print("Extracting sub mesh...")

        # extract surface surrounding the myocardium
        cmd = "{} extract mesh -msh={} -tags={} -submsh={} -ofmt=carp_txt".format(
            meshtool_bin,
            mesh_name,
            ",".join(str(t) for t in tags_list),
            output_mesh)

        os.system(cmd)
