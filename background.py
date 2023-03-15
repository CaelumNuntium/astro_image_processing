import argparse
import os
import numpy
from astropy.io import fits
from astropy import stats
from photutils import background

box_size = (64, 64)

parser = argparse.ArgumentParser()
parser.add_argument("input", action="store", help="input file name")
parser.add_argument("-out", "--output", action="store", default=".", help="output directory")
args = parser.parse_args()
fits_file = fits.open(args.input)
sigma_clip = stats.SigmaClip(sigma=3.0)
res = []
backgrounds = []
for img in fits_file:
    bkg = background.Background2D(img.data, box_size, filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=background.MedianBackground(), coverage_mask=(img.data == numpy.nan), fill_value=0.0)
    backgrounds.append(bkg.background)
    res.append(img.data - bkg.background)
res_hdu = [*[fits.PrimaryHDU(data=res[0], header=fits_file[0].header)], *[fits.ImageHDU(data=res[i], header=fits_file[i].header) for i in range(1, len(res))]]
bkg_hdu = [*[fits.PrimaryHDU(data=backgrounds[0])], *[fits.ImageHDU(data=backgrounds[i]) for i in range(1, len(backgrounds))]]
print("Result name:")
filename = input()
if not filename.endswith(".fits") and not filename.endswith(".fts"):
    filename = filename + ".fits"
bkg_name = args.output + "/background_" + filename
filename = args.output + "/" + filename
if not os.path.exists(args.output):
    os.makedirs(args.output)
fits.HDUList(res_hdu).writeto(filename, overwrite=True)
print(f"Result saved to {filename}")
fits.HDUList(bkg_hdu).writeto(bkg_name, overwrite=True)
print(f"Background images saved to {bkg_name}")
