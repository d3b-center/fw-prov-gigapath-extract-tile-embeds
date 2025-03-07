import os
import pandas as pd
import numpy as np
import json
from zipfile import ZipFile
import logging
import shutil
from glob import glob

from fw_core_client import CoreClient
from flywheel_gear_toolkit import GearToolkitContext
import flywheel
from .run_level import get_analysis_run_level_and_hierarchy

from .gigapath.pipeline import load_tile_slide_encoder
from .gigapath.pipeline import run_inference_with_tile_encoder
import h5py

os.environ["HF_TOKEN"] = "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"
assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

log = logging.getLogger(__name__)

fw_context = flywheel.GearContext()
fw = fw_context.client

def run(client: CoreClient, gtk_context: GearToolkitContext):
    """Main entrypoint

    Args:
        client (CoreClient): Client to connect to API
        gtk_context (GearToolkitContext)
    """
    # get the Flywheel hierarchy for the run
    destination_id = gtk_context.destination["id"]
    hierarchy = get_analysis_run_level_and_hierarchy(gtk_context.client, destination_id)
    acq_label = hierarchy['acquisition_label']
    sub_label = hierarchy['subject_label']
    ses_label = hierarchy['session_label']
    project_label = hierarchy['project_label']
    group_name = hierarchy['group']

    # get the output acqusition container
    acq = fw.lookup(f'{group_name}/{project_label}/{sub_label}/{ses_label}/{acq_label}')
    acq = acq.reload()

    # get the input file
    CONFIG_FILE_PATH = '/flywheel/v0/config.json'
    with open(CONFIG_FILE_PATH) as config_file:
        config = json.load(config_file)

    slide_path = config['inputs']['zipped_tiles']['location']['path'] # input path in the gear container
    shutil.move(slide_path, slide_path.replace(' ', '_')) # remove any spaces from file path
    slide_path = slide_path.replace(' ', '_')

    # set output file name based on input file name
    slide_name = os.path.basename(slide_path)
    slide_id = slide_name.split('_gigapath_tiles')[0]
    out_file_name = f'{slide_id}.h5'

    # extract the zipped tiles
    print("Unpacking the input file")
    local_output_dir = 'output/'
    with ZipFile(slide_path, 'r') as zip_ref:
        zip_ref.extractall(local_output_dir)
        os.remove(slide_path) # remove the zip file to save space

    print(f'======= Generating tile embeddings for file: {slide_name} =======')
    # make sure there are valid PNG files to process
    local_tile_dir = f"{local_output_dir}/tiles/*"
    local_tile_files = glob(f'{local_tile_dir}/*')
    tile_paths = [img for img in local_tile_files if img.endswith('.png')]
    if len(tile_paths) == 0:
        raise ValueError(f"No valid tiles found in {local_tile_dir}")
    else:
    # =========== Load the pretrained Gigapath model ===========================
        print("Loading pretrained model")
        tile_encoder, slide_encoder_model = load_tile_slide_encoder(global_pool=True)

        # =========== Run inference to get tile embeddings ===========================
        print("Running inference to generate tile embeddings")
        tile_encoder_outputs = run_inference_with_tile_encoder(tile_paths, tile_encoder)

        # save to H5 unless embeddings contain NaNs
        if np.isnan(np.sum(tile_encoder_outputs['tile_embeds'])):
            raise ValueError(f"Tile embeddings contain NaNs, please check input tile images")
        else:
            with h5py.File(out_file_name, 'w') as hf:
                for key in tile_encoder_outputs.keys():
                    hf.create_dataset(key, data=tile_encoder_outputs[key])

            print(f'Uploading output to acquisition: {acq.label}/{out_file_name}')
            acq.upload_file(f'{out_file_name}')
        
        # clean up
        os.remove(f'{out_file_name}') # remove from instance to save space
        for tile_dir in glob(f'{local_output_dir}/*'):
            shutil.rmtree(tile_dir) # remove the extracted files to save space
        print("Done!")
