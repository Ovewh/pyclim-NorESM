import yaml
import argparse as ap
import os


def find_missing_variables(data, models):
    """
    Takes a list of CMIP6 models checks if these models have the variables are available
    base on a data dictionary which generated by run_through_CMIP_data and is read in from
    a yaml file.
        Parameters
        ----------
            data:   dict
                        dict containing availalbe models and requested variables in a experiment
            models: list
                        list of model names to be checked if there are available

    """

    if isinstance(models, str):

        models = [models]
    missing_data = {}
    missing_data_piClim = {}
    for experiment in data:
        if experiment not in ["piClim-control", "piControl"]:
            missing_data[experiment] = {}
            for activity in data[experiment]:
                missing_data[experiment][activity] = {}
                for model in models:
                    missing_data[experiment][activity][model] = []
                for varialbe in data[experiment][activity]:

                    avaialbe_models = list(data[experiment][activity][varialbe].keys())
                    # Find out of set models
                    missing_models = list(set(models) - set(avaialbe_models))
                    for missing_model in missing_models:
                        missing_data[experiment][activity][missing_model].append(
                            varialbe
                        )
        elif experiment == 'piClim-control':
            piClim_control = data['piClim-control']
            RFMIP = piClim_control['RFMIP']
            AerChemMIP = piClim_control['AerChemMIP']
            missing_data_piClim['piClim-control']={}
            for model in models:
                missing_data_piClim[experiment][model] = []
            for variable in AerChemMIP:
                aerChem_models = [model for model in AerChemMIP[variable]]
                RFMIP_models = [model for model in RFMIP[variable]]
                not_in_aerChem = list(set(RFMIP_models) - set(aerChem_models))
                avaialbe_models = aerChem_models + not_in_aerChem
                missing_models_piClim = list(set(models)-set(avaialbe_models))

                if missing_models_piClim:
                    for missing_model in missing_models_piClim:
                        missing_data_piClim['piClim-control'][missing_model].append(variable)


        elif experiment == 'piControl':
            raise(NotImplementedError("""piControl simulation interface have not be implement yet"""))
                    

    if missing_data_piClim:
        missing_data = {**missing_data, **missing_data_piClim}

    return missing_data


def create_synda_file(missing_data, outpath="./", **synda_kwargs):
    """
    Create synda file for each model listed in the missing_data_dict and
    variables. The synda files are seperated by model activity and experiment.
    The file can then be use as input to synda for downloading new data.
    By default data from CMIP6 is requested
        Parameters:
        -----------
            missing_data:   dict
                                dict of variables that are missing in the Betzy CMIP6 archive.
            outpath:        str, default = "./"
                                where to store the synda setup files

    """
    def _to_file(variables_ids, experiment, project,  model,**synda_kwargs):
        with open( os.path.join(outpath, f"{project}_{experiment}_{model}.txt"),
                    mode="w",) as outfile:
            outfile.write(f"project={project}\n")
            outfile.write(f"source_id={model}\n")
            outfile.write(f"experiment_id={experiment}\n")
            outfile.write(
                "variable_id={}\n".format(
                    ",".join(variables_ids)
                )
            )
            for key, val in synda_kwargs.items():
                if isinstance(val,list):
                    val = ",".join(val)
                outfile.write(f"{key}={val}\n")


    project = synda_kwargs.get("project", "CMIP6")
    for experiment in missing_data:
        if experiment == "piClim-control":
            for model in missing_data[experiment]:
                _to_file(missing_data[experiment][model], experiment, project, model, **synda_kwargs)
        else:
            for activity in missing_data[experiment]:
                for model in missing_data[experiment][activity]:
                    _to_file(missing_data[experiment][activity][model],experiment, project, model, **synda_kwargs)



def main():
    parser = ap.ArgumentParser(
        description="""Tool finding what data is missing and create 
            synda files for downloading what is missing"""
    )
    parser.add_argument(
        "--file", "--f", help="yaml file that contains info on what is already downloaded"
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="which models to check against existing data and if needed",
    )
    parser.add_argument(
        "--synda_path",
        "--sp",
        default=None,
        help="""Path to where the synda settings files 
                                    for downloading the data sould be stored. If set to false no 
                                    synda files will be created""",
    )
    parser.add_argument(
        "--synda_config",
        "--sc",
        help="""Path to a yaml file with synda addition synda settings""",
        default=None,
    )
    args = parser.parse_args()
    fname = args.file
    models = args.models
    synda_outpath = args.synda_path
    synda_config_file = args.synda_config

    with open(fname, mode="r") as df:
        data = yaml.safe_load(df)
    missing_data = find_missing_variables(data, models)
    with open("missing_data.yaml", "w") as outfile:
        yaml.dump(missing_data, outfile, default_flow_style=False)

    if synda_config_file:
        with open(synda_config_file, mode="r") as conf:
            conf_synda = yaml.safe_load(conf)
    else:
        conf_synda={}


    if synda_outpath=='false':
        pass
    elif synda_outpath:
        create_synda_file(missing_data, outpath=synda_outpath, **conf_synda)
    else:
        create_synda_file(missing_data, **conf_synda)

if __name__ == "__main__":
    main()
