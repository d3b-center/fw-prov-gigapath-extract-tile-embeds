# reads in a list of svs files and if mag=40x and no 20x level is available,
# resamples them to 20x magnification

import openslide
from PIL import Image
import pyvips

import os
import pandas as pd

file_df = pd.read_csv('sub_list_2025_02_01_s3_paths.csv')
slide_dir = '/home/ubuntu/mount/d3b-phi-data-prd/imaging/pathology/cbtn/chop/input'

for ind,row in file_df.iterrows():
    slide_path = row['filepath']
    svs_path = f"{slide_dir}/{slide_path}" # full path to the file

    svs_dir = os.path.dirname(svs_path) # just the directory
    slide_name = os.path.basename(slide_path) # just the file name

    # Open the SVS file
    # svs_dir = "/home/ubuntu/mount/d3b-phi-data-prd/imaging/pathology/cbtn/chop/input/aperio/Images2/2021-03-31/"
    # slide_name = '452716.svs'

    slide = openslide.OpenSlide(svs_path)

    slide_name = os.path.splitext(slide_name)[0] # remove file extension

    # Check available magnification levels
    objective_power = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    downsample_factors = slide.level_downsamples  # List of downsample factors

    # print(f"Objective Power: {objective_power}x")
    # print(f"Downsample Factors: {downsample_factors}")

    if objective_power == 40:
        downsample_factor = 2  # 40x to 20x
        # Get the level corresponding to 20x
        try:
            level_20x = downsample_factors.index(downsample_factor)
        except:
            level_20x = []

        if level_20x != []:
            print(f"20x magnification level found at level {level_20x}.")
            # Extract the 20x region
            # img_20x = slide.read_region((0, 0), level_20x, slide.level_dimensions[level_20x])
            # img_20x = img_20x.convert("RGB")  # Convert from RGBA to RGB
            # img_20x.show()  # Display image or save it
            # img_20x.save("downsampled_20x.png")
        else:
            print("20x magnification not found in metadata, resampling image 40x to 20x.")
            img_40x = slide.read_region((0, 0), 0, slide.level_dimensions[0]).convert("RGB")
            img_20x = img_40x.resize((img_40x.width // 2, img_40x.height // 2), Image.LANCZOS)
            # img_20x.save("manually_downsampled_20x.png")

        # Convert to VIPS image and save as pyramidal TIFF
        vips_img = pyvips.Image.new_from_array(img_20x)
        vips_img.write_to_file(f"{svs_dir}/{slide_name}_20x.tiff", pyramid=True, tile=True, compression="jpeg")

    else:
        print("40x magnification not found in metadata.")
