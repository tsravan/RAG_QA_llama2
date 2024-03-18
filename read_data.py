import os
import shutil
import app_utils
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from helper.constants import _Constant

CONST = _Constant()

cfg = app_utils.read_yaml_file(CONST.DATA_LOADER_CONFIG_YML)

# read metadata information
def read_metadata(path_mdata, use_cols=None, nrows=None):
    df_metadata = pd.read_csv(path_mdata, skipinitialspace=True, usecols=use_cols, nrows=nrows)
    df_metadata = df_metadata.loc[~df_metadata.index.duplicated(keep='first')]

    # Keep papers with access to pdf/pmc content
    df_metadata = df_metadata[~(df_metadata['pdf_json_files'].isnull() & df_metadata['pmc_json_files'].isnull())]   
    df_metadata.reset_index(inplace=True) 
    return df_metadata

def get_cord19_data_dir():
    '''
    Checks for Cord19 directory
    if cor19 directory given along with sample data size and sample_data_folder_name then- 
    - specified sample size data will be moved to the specified sample data folder name
    '''
    if cfg['cord19_dir'] != None and cfg['sample_data_folder_name'] != None and cfg['data_sample_size'] != None:
        path_mdata = os.path.join(cfg['cord19_dir'], cfg['meta_data_file_name'])
        if cfg['data_sample_size']:
            print('---------> sample size :', cfg['data_sample_size'])
            df_metadata = read_metadata(path_mdata, use_cols=cfg['use_columns'], nrows=cfg['data_sample_size'])

            sample_data_dir = os.path.join(os.getcwd(), cfg['sample_data_folder_name'])

            if os.path.exists(sample_data_dir):
                shutil.rmtree(sample_data_dir)

            print('---------> creating sample data folder')    
            os.makedirs(sample_data_dir)
            
            for ind in tqdm(df_metadata.index):
                if not pd.isnull(df_metadata.iloc[ind]['pmc_json_files']):
                    path_json = os.path.join(cfg['cord19_dir'], df_metadata.iloc[ind]['pmc_json_files'])
                    _ = shutil.copy2(path_json, sample_data_dir)
                elif not pd.isnull(df_metadata.loc[ind,'pdf_json_files']):
                    path_json = os.path.join(cfg['cord19_dir'], df_metadata.iloc[ind]['pdf_json_files'])
                    _ =shutil.copy2(path_json, sample_data_dir)
        
            return sample_data_dir
        else:
            return cfg['cord19_dir']
        
    elif cfg['cord19_dir'] == None:        
        sample_data_dir = Path(CONST.BASE_PATH, cfg['sample_data_folder_name'])
        print(f"---------> Using data available in {sample_data_dir}")
        return sample_data_dir

