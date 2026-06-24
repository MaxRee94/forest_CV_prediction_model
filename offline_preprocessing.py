import os
from types import SimpleNamespace
from glob import glob
import rasterio as io
import csv



def save_processed_data(rows, output_file):
    with open(output_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


def main(cfg):
    # Get all the .tif files in the data directory
    tif_files = glob(os.path.join(cfg.data_dir, "*.tif"))
    

    # Do processing
    datarows = []
    for i, tif_file in enumerate(tif_files):
        # Read the data from the .tif file
        with io.open(tif_file) as src:
            data = src.read(1)  # Read the first band

        print("tif file:", tif_file)


        # Calculate the stepsize in pixels based on the degrees and the resolution of the raster
        if i == 0:
            stepsize_pixels = int(cfg.stepsize / src.res[0])  # Assuming square
            n_steps = int(data.shape[0] / stepsize_pixels)  # Number of steps in the raster

        for i in range(n_steps):
            for j in range(n_steps):
                # Calculate indices
                x, y = i * stepsize_pixels, j * stepsize_pixels

                # Obtain value from the data array
                value = data[x, y]
                value = 1 if value >= 65 else 0  # Convert to binary based on 65%-threshold (minimum of tree cover histogram, see Zwaan et al. 2024)

                # Obtain longlat
                longlat = src.xy(x, y)
                longlat = [float(val) for val in longlat]  # Convert to float

                # Prepare row
                datarow = [longlat, value]

                # Perform preprocessing steps on the data
                # For example, you can normalize the data or apply any other transformations
                datarows.append(datarow)
        
        break # TEMP, for testing

    # Save the processed data to the csv file
    save_processed_data(datarows, cfg.output_csv)


if __name__ == "__main__":
    cfg = SimpleNamespace()
    cfg.data_dir = r"C:\Users\6241638\OneDrive - Universiteit Utrecht\Documents\Data\GFC treecover data\big_tiles"
    cfg.output_csv = os.path.join(cfg.data_dir, "preprocessed_TC_data_v3.csv")
    cfg.study_area = ["N00E010", "N02E012"] # top-left and bottom-right corners of the study area
    cfg.stepsize = 0.05 # degrees

    # Initialize the output CSV file with headers
    with open(cfg.output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header row
        writer.writerow(["longlat", "TC_value"])

    main(cfg)
