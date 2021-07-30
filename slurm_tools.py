import math

def generate_header(n_cores, job_name, time_limit):
    """
    Create header of SLURM script for TOM2
    Args:
        n_cores (int): number of CPU cores
        job_name (str): name of job on SLURM script. Must be less than 8 characters long.
        time_limit (int): simulation time limit in hours. Less than 76 hours is recommended

    Returns:
        header (str): header of SLURM script
    """

    # set number of nodes
    n_tasks = 64
    if n_cores % n_tasks == 0:
        n_nodes = int(n_cores / n_tasks)
    else:
        raise ValueError('# cores must be multiple of {:d} on TOM2'.format(n_tasks))

    # build header
    header = "\n".join(["#!/bin/bash",
                        "#SBATCH -p compute",
                        "#SBATCH -J {}".format(job_name),
                        "#SBATCH -t 0-{:d}:00:00".format(time_limit),
                        "#SBATCH --nodes={:d}".format(n_nodes),
                        "#SBATCH --ntasks-per-node={:d}".format(n_tasks)
                        ])

    return header

def add_job_array_options(n_cores, start_nbr, end_nbr, core_limit=512):
    """
    Add command line to run job arrays on TOM2
    Args:
        n_cores: number of CPU cores
        start_nbr: number of job to start array from
        end_nbr: number of job to end array
        core_limit: number of cores that can be used at once (Recommended 512 on TOM2)

    Returns:
        job_array (str): string with job array command line

    """

    if math.floor(core_limit / n_cores) < 2:
        raise ValueError(
            'core_limit (={:d}) divided by # cores (={:d}) smaller than 2. Cannot set array job'.format(core_limit,
                                                                                                        n_cores))

    batch_size = int(math.floor(core_limit / n_cores))  # number of jobs to run in batch
    job_array = "#SBATCH --array=[{:d}-{:d}]%{:d}\n".format(start_nbr, end_nbr, batch_size)

    return job_array


def source_bashrc():
    """
    Add command line to source modules from .bashrc
    Returns:
        line (str):
    """
    line = "source ${HOME}/.bashrc"

    return line

def add_modules(module_names):
    """
    Add command line to load individual modules within script
    Args:
        module_names (list): string or list of strings with module names to load

    Returns:
        all_modules (str): string with module load commands for each module in the module_names
    """

    lines = list()
    for name in module_names:
        lines.append("load module {}".format(name))

    all_modules = "\n".join(lines)

    return all_modules

