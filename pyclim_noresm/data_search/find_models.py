import argparse as ap
import os
import pandas as pd
import yaml
import glob

import matplotlib.pyplot as plt
import pathlib as pl

import yaml


def find_models(root_dir, activity, experiment, variables, realization="*"):
    """
    Find all models that have the requested variable available.
        Parameters
        ----------
        directory:  str
                        root directory for where the CMIP6 data is stored
        activity:   str
                        under wihch CMIP6 activity to search
        expermient: str
                        name of experiment
        variable:   str
                        which variable to search for

        Return
        ------
            results: dict
                        models and variables that are available. 

    """
    temp_results = {}
    root_dir = pl.Path(root_dir)
    subdir = root_dir.joinpath(activity)
    experiments_path = sorted(subdir.glob(f"*/**/{experiment}"))
    if experiments_path:
        for variable in variables:
            temp_results[variable] = [
                sorted(exp_path.glob(f"{realization}/**/{variable}"))
                for exp_path in experiments_path
            ]
        results = {}
        results[experiment] = {}
        results[experiment][activity]={} 
        for variable in variables:
            hits = temp_results[variable]
            results[experiment][activity][variable] = {}
            if hits:
                results[experiment][activity][variable] = {hit[0].parts[-5]: str(hit[0]) for hit in hits if hit}

    else:
        results = {}
    return results
    


def main():
    parser = ap.ArgumentParser(
        prog="CMIP6 data run through",
        description="""Runs through the Betzy/Nrid CMIP6 and 
                    tells you whether the data is there or not""",
        usage="%(prog)s [options]",
    )

    parser.add_argument("variables", nargs="+")
    parser.add_argument(
        "--root_dir",
        "--rd",
        help="Top directory of where cmip output is stored",
        default=None,
    )

    parser.add_argument(
        "--file",
        "--f",
        help="""path to config file
     (if provided other args are optional)""",
        default=None,
    )
    parser.add_argument("--experiment", "--e", help="which experiment to search")
    parser.add_argument(
        "--out_path", "--o", help="which file to store the information", default=None
    )
    parser.add_argument("--activity", help="CMIP6 activity")
    parser.add_argument("--find_control","--fc", action="store_true", 
            help="If enabled will also search for the piClim-control runs")

    args = parser.parse_args()
    variables = args.variables
    root_dir = args.root_dir
    experiment = args.experiment
    out_path = args.out_path
    activity = args.activity
    find_control = args.find_control
    result = find_models(root_dir, activity, experiment, variables)
    
    if find_control:
        root = pl.Path(root_dir)
        control_list = []
        for path in list(root.iterdir()):
            control = find_models(root_dir,path.name,'piClim-control', variables)
            if control:
                control_list.append(control)
        control = dict(pair for d in control_list for pair in d['piClim-control'].items())
        result['piClim-control'] = control

    if out_path:
        fname = os.path.join(out_path, f"available_{experiment}_{activity}.yaml")
    else:
        fname = f"available_{experiment}_{activity}.yaml"
    with open(fname, "w") as out_file:
        yaml.dump(result, out_file, default_flow_style=False)

if __name__ == "__main__":
    main()