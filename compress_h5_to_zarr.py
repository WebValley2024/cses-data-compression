import os
from glob import glob
from multiprocessing import Pool
from functools import partial

import fire
import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc, blosc
from tqdm import tqdm


def worker(file, outfolder=None, overwrite=False):
    try:
        zarrfile = file.replace(".h5", ".zarr.zip")

        if outfolder is not None:
            os.makedirs(outfolder, exist_ok=True)
            # keep only the filename
            zarrfile = os.path.basename(zarrfile)
            # add the output folder
            zarrfile = os.path.join(outfolder, zarrfile)

        if os.path.exists(zarrfile) and not overwrite:
            return

        compressor = Blosc(cname="zstd", clevel=9, shuffle=Blosc.AUTOSHUFFLE)
        ds = xr.open_dataset(file)

        # get the time dimensione, which can be called either "UTCTime" or "UTC_TIME"
        if "UTCTime" in ds.var():
            time_dim = "UTCTime"
        elif "UTC_TIME" in ds.var():
            time_dim = "UTC_TIME"
        else:
            raise ValueError("Time dimension not found (tried UTCTime and UTC_TIME)")

        time_dim = ds[time_dim].dims[0]

        encodings = {}
        for k in ds.data_vars:
            if ds[k].dims[0] == time_dim:
                # use chunks of 128 along the time dimension
                chunks = list(ds[k].shape)
                chunks[0] = 128 if chunks[0] > 128 else chunks[0]
                encodings[k] = {"compressor": compressor, "chunks": tuple(chunks)}
            else:
                # do not chunk along other dimensions
                encodings[k] = {"compressor": compressor}

        store = ds.to_zarr(zarrfile, mode="w", encoding=encodings)
        store.close()

    except Exception as e:
        print(f"Error processing {file}: {e}")


def main(data_dir: str, outfolder: str, n_workers: int = 8):
    files = glob(os.path.join(data_dir, "**/*.h5"), recursive=True)
    pool = Pool(n_workers)

    for _ in tqdm(
        pool.imap_unordered(partial(worker, outfolder=outfolder), files),
        total=len(files),
    ):
        pass


if __name__ == "__main__":
    fire.Fire(main)
