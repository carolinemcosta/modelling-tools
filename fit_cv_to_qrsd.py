import numpy as np
import os

from ecg_analysis import vcg_analysis
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
    init_file = file_name + ".init"
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
        stim_node = getRVpacingLocation(mesh_name, uvc_dir, rv_septum_vtx)
    else:
        stim_node = getRVpacingLocationLV(mesh_name, uvc_dir)

    # define tags
    tags = dict(he=2, bz=6, rv=9, he_fec=202, bz_fec=206, sc_fec=205, rv_fec=209)

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
        tact (float): total activation time at the base

    """

    act_file_pp = act_file + "-pp.dat"
    tact_file = act_file + "-pp.TACTbase"
    if not os.path.isfile(tact_file):
        base_nodes = np.loadtxt(base_vtx_file, skiprows=2, dtype=int)
        act = np.loadtxt(act_file_pp)
        # get myo TACT
        tact = np.amax(act[base_nodes])
        # save TACT
        with open(tact_file, 'w+') as f:
            f.write("{:f}".format(tact))
    else:
        tact = np.loadtxt(tact_file, dtype=float)

    return tact


# TODO: Functions below
def getBestCV(pigname, meshname, simdir, simID, cvlvec, qrsd, basename, endoname):
    diffendo = np.zeros(cvlvec.size)
    diffbase = np.zeros(cvlvec.size)
    inds = np.arange(cvlvec.size)
    tact = np.zeros(cvlvec.size)

    for i in inds:
        cvl = cvlvec[i]
        filename = simdir + simID + "-%1.2f" % cvl

        if not os.path.isfile(filename + "-pp.dat"):
            # remove inf values
            post_process_act(filename)

        # get base total ACT
        tactfile = filename + "-pp.TACTbase"
        if not os.path.isfile(tactfile):
            tactbase = compute_total_base_act(filename, basename)
        else:
            tactbase = np.loadtxt(tactfile, dtype=float)

        # compare with QRSd
        diffbase[i] = abs(qrsd - tactbase)
        tact[i] = tactbase

    # get min differences
    mindiffbase = np.amin(diffbase)
    mindiffindbase = inds[diffbase == mindiffbase]

    # treat multiple solution
    if mindiffindbase.size > 1:
        print("multiple solutions for TACT base, taking first one")
        minind = mindiffindbase[0]
    else:
        print("unique solution found TACT base")
        minind = mindiffindbase

    finalcvlbase = cvlvec[minind]
    # transverse velocity
    finalcvtbase = 0.45 * finalcvlbase

    # write fit files
    cvfilebase = simdir + simID + "-baseTACT" + "-fit.dat"
    fb = open(cvfilebase, 'w+')
    fb.write("%f %f %f %f\n" % (qrsd, tact[minind], finalcvlbase, finalcvtbase))
    fb.close()

    return finalcvlbase




def getRVpacingLocation(meshname, uvcdir, rvseptvtx):
    plptsname = uvcdir + "rvseptpl.pts_t"
    if not os.path.isfile(plptsname):
        print("computing pacing location...\n")

        # load rv sept nodes
        snodes = np.loadtxt(rvseptvtx, skiprows=2, dtype=int)

        # load UVC coordinates
        zcoord = np.loadtxt(uvcdir + "COORDS_Z.dat", dtype=float)
        # rhocoord = np.loadtxt(uvcdir+"COORDS_RHO.dat",dtype=float)
        phicoord = np.loadtxt(uvcdir + "COORDS_PHI_ROT.dat", dtype=float)

        # get RV apex as RHO = 1, PHI = 0, and min(Z)
        idx = np.arange(zcoord.size)
        # epi = idx[rhocoord==1.0]

        midseptp = idx[phicoord > -0.01]
        midseptm = idx[phicoord < 0.01]
        midsept = np.intersect1d(midseptp, midseptm, assume_unique=True)

        midrvsept = np.intersect1d(snodes, midsept, assume_unique=True)

        rvpl = midrvsept[zcoord[midrvsept].argmin()]

        plvtxname = uvcdir + "rvseptpl.vtx"
        f = open(plvtxname, "w")
        f.write("1\nextra\n%d" % rvpl)
        f.close()

        pts = np.loadtxt(meshname + ".pts", skiprows=1, dtype=float)
        plpts = pts[rvpl, :]
        f = open(plptsname, "w")
        f.write("1\n1\n%f %f %f" % (plpts[0], plpts[1], plpts[2]))
        f.close()
    else:
        rvpl = np.loadtxt(uvcdir + "rvseptpl.vtx", skiprows=2, dtype=int)

    return rvpl


def getRVpacingLocationLV(meshname, uvcdir):
    print("computing pacing location...\n")

    # load mesh points
    pts = np.loadtxt(meshname + ".pts", skiprows=1, dtype=float)

    # load rv sept pacing location from BiV mesh
    plfile = uvcdir + "rvseptpl.pts_t"
    bivpl = np.loadtxt(plfile, skiprows=2, dtype=float)

    # find closest point in LV mesh
    rvpl = np.sum((pts - bivpl) ** 2, axis=1).argmin()

    print(rvpl)

    return rvpl


def runPhieRec(carpexec, parfile, imeshname, simdir, cvsimID, elecfile):
    actfile = simdir + cvsimID + ".dat"
    actfilepp = simdir + cvsimID + "-pp.dat"
    if not os.path.isfile(actfilepp):
        post_process_act(actfile)  # remove inf values

    stimfile = simdir + cvsimID + "-pp_i.dat"
    if not os.path.isfile(stimfile):
        # extract .dat on intracellular grid
        cmd = "meshtool extract data -submsh=%s -msh_data=%s -submsh_data=%s" % (imeshname, actfilepp, stimfile)
        os.system(cmd)

    num = "-diffusionOn 0 -mass_lumping 0 -tend 500.0 -dt 100.0 -spacedt 5 "
    stim = "-num_stim 2 -stimulus[0].stimtype 8 -stimulus[0].data_file %s -stimulus[1].stimtype 8 -stimulus[1].duration 2.0 " % stimfile
    rec = "-experiment 4 -dump_ecg_leads 1 -post_processing_opts 1 -phie_rec_ptf %s" % elecfile

    if not os.path.isfile(simdir + cvsimID + "/phie_recovery.igb"):
        print("recoving Phie on local machine...")
        cmd = "mpirun -np 12 %s +F %s -meshname %s -simID %s %s %s %s" % (
        carpexec, parfile, imeshname, simdir + cvsimID, num, stim, rec)
        os.system(cmd)


def computeECGandQRSd(carpexec, parfile, meshname, imeshname, cvlrange, simdir, simID, elecfile, biv, thresh):
    qrsd = np.zeros((cvlrange.size,), dtype=float)
    flag = np.ones((cvlrange.size,), dtype=bool)

    for i in range(cvlrange.size):

        cvsimID = "%s-%1.2f" % (simID, cvlrange[i])

        phiefile = simdir + cvsimID + "/phie_recovery.igb"

        if os.path.isfile(phiefile):
            ecgfile = simdir + cvsimID + '/12-lead-ecg.csv'

            if not os.path.isfile(ecgfile):
                # get data from electrodes
                print("Converting Phie to ECG...")
                edata, t_steps = et.get_electrodes_from_phie_rec_file(phiefile)
                tstep = t_steps[1] - t_steps[0]

                # get ECG from electrodes
                ecg = et.convert_electrodes_to_ecg(edata)

                # save ECG
                ecgfile = simdir + cvsimID + "/12-lead-ecg.csv"
                et.write_ecg(ecgfile, ecg)

            else:
                ecg, nsteps = et.read_ecg(ecgfile)
                tstep = 2

            # normalize ECG
            norm_ecg = et.normalize_ecg(ecg)

            # convert normlized ECG to VCG
            print("Computing QRS duration...")
            vcg = vcg_analysis.convert_ecg_to_vcg(norm_ecg)
            start, end, duration = vcg_analysis.get_qrs_start_end(vcg, dt=1, velocity_offset=2, low_p=40, order=2,
                                                                  threshold_frac_start=thresh,
                                                                  threshold_frac_end=thresh, filter_sv=True, t_end=250,
                                                                  matlab_match=False)

            qrsd[i] = duration[0] * tstep
            qstart = start[0] * tstep
            qend = end[0] * tstep

            # save QRSd to file
            qrsdfile = simdir + cvsimID + '/qrsd.dat'
            if not os.path.isfile(qrsdfile):
                np.savetxt(qrsdfile, (qstart, qend, qrsd[i]), delimiter=' ', fmt='%1.6f')

        else:  # if phie is missing return flag with false
            flag[i] = False

    return qrsd, flag


def phiRecTom2(pig, carpexec, meshname, imeshname, cvlrange, simdir, simID, elecfile, biv, machine, simprefix, factor):
    # set directories
    tommeshdir = "/scratch/cmc16/meshes/"
    tomsimdir = "/scratch/cmc16/simulations"
    if factor > 4.:
        tomsimdir += "-fec6x"

        # set sim name
    if biv:
        model = "biv"
    else:
        model = "lv"
    simname = "%s%s%s" % (pig, model, simprefix)

    # generate script
    scriptname = "%s/pig%s-%s%s-ref.sh" % (simdir, pig, model, simprefix)
    if not os.path.isfile(scriptname):
        print("Generating scripts for Phie recovery on Tom2...")
        # create vm.igb file from eikonal activation sequence
        mask = np.ones(len(cvlrange), dtype=bool)
        for i in range(cvlrange.size):
            cvsimID = "%s-%1.2f" % (simID, cvlrange[i])
            actfile = simdir + cvsimID
            actfilepp = simdir + cvsimID + "-pp.dat"
            actfileppi = simdir + cvsimID + "-pp_i.dat"
            ptsname = imeshname + ".pts"

            if not os.path.isfile(actfilepp):
                # remove inf values
                post_process_act(actfile)

            # extract .dat on intracellular grid
            if not os.path.isfile(actfileppi):
                cmd = "meshtool extract data -submsh=%s -msh_data=%s -submsh_data=%s" % (
                imeshname, actfilepp, actfileppi)
                os.system(cmd)

                # create slurm scrips for tom2
        tommesh = tommeshdir + imeshname.split('-meshes/')[1]
        tomelecfile = tommeshdir + elecfile.split('-meshes/')[1]
        tomparfile = tomsimdir + "/reaction-eikonal.par"

        # build header
        tjobs = cvlrange.size - 1
        nbatch = 4

        header = "#!/bin/bash\n#SBATCH -p compute\n#SBATCH -J %s\n#SBATCH -t 0-24:00:00\n#SBATCH --nodes=2\n#SBATCH --ntasks-per-node=64\n" % simname
        if tjobs > 0:
            header += "#SBATCH --array=[0-%d]%%%d\n\n" % (tjobs, nbatch)  # array job
        header += "source ${HOME}/.bashrc\nexport OMP_NUM_THREADS=1\n\n"

        # create cv array
        header += "declare -a cvrange=("
        for cvl in cvlrange:
            header += "\"%1.2f\" " % cvl

        header += ")\n\n"

        # create command line
        nsimID = "%s/%s-${cvrange[$SLURM_ARRAY_TASK_ID]}" % (tomsimdir, simID)
        stimfile = nsimID + "-pp_i.dat"
        num = "-diffusionOn 0 -mass_lumping 0 -tend 500.0 -dt 100.0 -spacedt 2 -pstrat_i 1 -pstrat 2"
        stim = "-num_stim 2 -stimulus[0].stimtype 8 -stimulus[0].data_file %s -stimulus[1].stimtype 8 -stimulus[1].duration 2.0 " % stimfile
        rec = "-experiment 4 -dump_ecg_leads 1 -post_processing_opts 1 -phie_rec_ptf %s" % tomelecfile

        recmd = "mpirun -np 128 %s +F %s -meshname %s -simID %s %s %s\n\n" % (
        carpexec, tomparfile, tommesh, nsimID, num, stim)
        phiecmd = "mpirun -np 128 %s +F %s -meshname %s -simID %s %s %s %s\n\n" % (
        carpexec, tomparfile, tommesh, nsimID, num, stim, rec)

        rmcmd = "rm %s/vm.igb\n" % nsimID

        # write script
        script = header + recmd + phiecmd + rmcmd
        f = open(scriptname, 'w')
        f.write(script)
        f.close()

        # copy script to tom2
        cmd = "scp %s cmc16@tom2-login.hpc.isd.kcl.ac.uk:%s" % (scriptname, tomsimdir)
        os.system(cmd)


def getBestCVfromQRSd(qrsd, simqrsd, cvlrange, simdir, simID):
    print("Computing best CV from QRSd...")

    qrsddiff = abs(qrsd - simqrsd)
    mindiff = np.amin(qrsddiff)
    inds = np.arange(cvlrange.size)
    mindiffind = inds[qrsddiff == mindiff]

    # treat multiple solution
    if mindiffind.size > 1:
        print(cvlrange[mindiffind])
        minind = mindiffind[0]
    else:
        minind = mindiffind

    finalcvl = cvlrange[minind]

    print(finalcvl)

    finalcvt = finalcvl * 0.45
    finalcvfile = simdir + simID + "-baseTACT" + "-final-fit-qrsd.dat"
    f = open(finalcvfile, 'w+')
    f.write("%f %f %f %f\n" % (qrsd, simqrsd[minind], finalcvl, finalcvt))
    f.close()

    return finalcvl


def runSimFitCV(carpexec, parfile, lmdir, edir, pigs, simprefix, mshprefix, ekbatchexec, simdir, cvlvec, biv, simtype,
                factor, machine, thresh):
    for pig in pigs:
        pigname = "pig%d" % pig
        print("\n%s...\n" % pigname)

        omeshname = "%s/%s-biv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing" % (lmdir, pigname)
        bivmeshname = "%s/%s-biv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath-fibres%s-fec-0.5-scar" % (
        lmdir, pigname, mshprefix)
        lvmeshname = "%s/%s-lv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath-fibres%s-fec-0.5-scar" % (
        lmdir, pigname, mshprefix)
        lvnfmeshname = "%s/%s-lv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath-fibres%s" % (
        lmdir, pigname, mshprefix)  # lv no fec

        uvcdir = "%s/uvc/%s-biv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath-no-scar.uvc/UVC/" % (
        lmdir, pigname)
        rvseptvtx = "%s/uvc/%s-biv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath-no-scar.rvsept.surf.vtx" % (
        lmdir, pigname)

        elecfile = "%s/%s-biv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath-fibres-electrodes" % (
        lmdir, pigname)

        if biv == 1:
            basename = "%s/mapped/%s-biv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath.base.surf.vtx" % (
            lmdir, pigname)
            endoname = "%s/mapped/%s-biv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath.lvendo.surf.vtx" % (
            lmdir, pigname)
            simID = "%s-biv%s" % (pigname, simprefix)
            meshname = bivmeshname

        else:
            # extract LV from BiV mesh for this because of FEC on RV endo.
            outbasename = "%s/%s-lv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath.base" % (lmdir, pigname)
            basename = "%s/mapped/lv/%s-lv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath.base.surf.vtx" % (
            lmdir, pigname)
            endoname = "%s/mapped/lv/%s-lv-smooth-0.05-ff-1.0-base-bath-0.25-surfsmoothing-no-bath.endo.surf.vtx" % (
            lmdir, pigname)
            simID = "%s-lv%s" % (pigname, simprefix)
            meshname = lvmeshname

            if not os.path.isfile(meshname + ".elem"):
                print("extracing LV from original mesh")
                os.system(
                    "meshtool extract mesh -msh=%s -tags=2,5,6 -submsh=%s -ofmt=carp_txt" % (omeshname, lvnfmeshname))
            if not os.path.isfile(outbasename + ".surf"):
                print("extract base from original mesh with bath")
                os.system("meshtool extract surface -msh=%s -surf=%s -op=1:2 -ofmt=carp_txt" % (omeshname, outbasename))
            if not os.path.isfile(basename):
                print(" map base onto LV mesh withouth FEC")
                os.system("meshtool map -submsh=%s -files=%s,%s -outdir=%s/mapped/lv" % (
                lvnfmeshname, outbasename + ".surf", outbasename + ".surf.vtx", lmdir))
            if not os.path.isfile(lvmeshname + ".elem"):
                print("extract LV from biv mesh with FEC to be used in simulations")
                os.system("meshtool extract mesh -msh=%s -tags=2,202,5,205,6,206 -submsh=%s -ofmt=carp_txt" % (
                bivmeshname, lvmeshname))

        # intracellular grid
        imeshname = meshname + "_i"

        ### INITIAL PARAMETERIZATION USING EIKONAL SIMULATION TOTAL ACTIVATION TIME
        data = np.loadtxt(edir + pigname + "-qrsd-vcg-mean-beat-norm.dat", dtype=float)  # NEW QRSd FILE!!!
        qrsd = data[2]

        tactname = simdir + simID + "-0.96-pp.TACTbase"
        fitname = simdir + simID + "-baseTACT" + "-fit.dat"
        if os.path.isfile(fitname):
            tactcvl = np.loadtxt(fitname, usecols=2, dtype=float)
        else:
            # run sims
            mask = np.ones(len(cvlvec), dtype=bool)
            for idcv in range(len(cvlvec)):
                actname = simdir + simID + "-%1.2f.dat" % cvlvec[idcv]
                if os.path.isfile(actname):
                    mask[idcv] = False

            miscvlvec = cvlvec[mask]
            if miscvlvec.size > 0:
                print("running sims...\n")
                run_ekbatch(pigname, ekbatchexec, meshname, simdir, simID, uvcdir, rvseptvtx, miscvlvec, biv, simtype,
                            factor)

            # compute best CV based on total activation time at base
            print("computing best CV...\n")
            tactcvl = getBestCV(pigname, meshname, simdir, simID, cvlvec, qrsd, basename, endoname)

        ### PARAMETERIZATION REFINIMENT USING QRS DURATION OF ESTIMATED ECG

        # CV range for ECG sim
        # qrsdinc = 0.2 # qrsd search interval
        qrsdinc = 0.3  # qrsd search interval
        mincvl = max([tactcvl - qrsdinc, 0.36])
        maxcvl = min([tactcvl + qrsdinc, 0.96])

        # Widen search if final QRSd far from experimental value
        # widened = False
        # nqrsdinc = qrsdinc + 0.1
        # fitname = simdir + simID + "-baseTACT" + "-final-fit-qrsd.dat"
        # if os.path.isfile(fitname):
        # finalqrsd, finalcvl = np.loadtxt(fitname,usecols=(1,2),dtype=float)
        # if finalqrsd-qrsd < -2.0: # QRS too fast, widen search to slower CVs
        # widened = True
        # nmincvl = max([tactcvl-nqrsdinc,0.36])
        # lowint = np.arange(nmincvl,mincvl,0.01)
        # upint = []
        # mincvl = nmincvl # replace min
        # elif finalqrsd-qrsd > 2.0: # QRS too slow, widen search to faster CVs
        # widened = True
        # nmaxcvl = min([tactcvl+nqrsdinc,0.96])
        # upint = np.arange(maxcvl,nmaxcvl,0.01)
        # lowint = []
        # maxcvl = nmaxcvl # replace max

        # fit simulated QRSd to data QRSd
        if machine == 'local':
            cvlrange = np.arange(mincvl, maxcvl, 0.01)
            # compute QRSd
            simqrsd, flag = computeECGandQRSd(carpexec, parfile, meshname, imeshname, cvlrange, simdir, simID, elecfile,
                                              biv, thresh)
            # get best CV based on QRS duration
            finalcvl = getBestCVfromQRSd(qrsd, simqrsd, cvlrange, simdir, simID)
        else:
            # generate scripts
            # if widened:
            # cvlrange = np.concatenate((lowint,upint))
            # if cvlrange.size > 0:
            # phiRecTom2(pig,carpexec,meshname,imeshname,cvlrange,simdir,simID,elecfile,biv,machine,simprefix,factor)

            cvlrange = np.arange(mincvl, maxcvl, 0.01)
            phiRecTom2(pig, carpexec, meshname, imeshname, cvlrange, simdir, simID, elecfile, biv, machine, simprefix,
                       factor)


def main():
    # executables
    ekbatchexec = "ekbatch"
    carpexec = "carp.pt"

    # directories
    mdir = "/data2/KCL/Martin/"
    edir = mdir + "ECGdata/"
    lmdir = mdir + "biv-meshes/"

    # sim variables
    parfile = mdir + "reaction-eikonal.par"

    machine = 'local'
    # machine = 'tom2'
    # machine = 'midlands'

    # CV ranges
    # CV_l : 0.36 - 0.96
    # CV_t : 0.16 - 0.43 (0.45 x CV_l)
    # CV_bz: 0.08 - 0.22 (0.50 x CV_t)

    mincvl = 0.36
    maxcvl = 0.96
    cvstep = 0.01
    cvlvec = np.arange(mincvl, maxcvl + cvstep, cvstep)

    pigs = [20, 21, 23, 24, 25, 27]

    factor = 6.0
    bivsimdir = mdir + "fit-ecg-biv-fec6x-sims/"
    lvsimdir = mdir + "fit-ecg-lv-fec6x-sims/"

    mshprefix = ""
    thresh = 0.3  # using different thresholds for the 1mm and the 4 and 10mm models due to more fractionated signals in the 4 and 10mm models
    biv = 1
    simtype = 0
    simprefix = "-1mm"
    runSimFitCV(carpexec, parfile, lmdir, edir, pigs, simprefix, mshprefix, ekbatchexec, bivsimdir, cvlvec, biv,
                simtype, factor, machine, thresh)

    # biv = 0
    # simtype = 0
    # simprefix = "-1mm"
    # runSimFitCV(carpexec,parfile,lmdir,edir,pigs,simprefix,mshprefix,ekbatchexec,lvsimdir,cvlvec,biv,simtype,factor,machine,thresh)

    # thresh = 0.15
    # mshprefix = "-slice-4mm"
    # biv = 1
    # simtype = 0
    # simprefix = "-4mm"
    # runSimFitCV(carpexec,parfile,lmdir,edir,pigs,simprefix,mshprefix,ekbatchexec,bivsimdir,cvlvec,biv,simtype,factor,machine,thresh)

    # mshprefix = "-slice-10mm"
    # biv = 1
    # simtype = 0
    # simprefix = "-10mm"
    # runSimFitCV(carpexec,parfile,lmdir,edir,pigs,simprefix,mshprefix,ekbatchexec,bivsimdir,cvlvec,biv,simtype,factor,machine,thresh)


if __name__ == "__main__":
    main()
