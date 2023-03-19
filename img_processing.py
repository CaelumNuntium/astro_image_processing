print("Loading modules...")

import pathlib
import argparse
import os
import numpy
import warnings
import cv2
from astropy.io import fits
from astropy import visualization
from photutils import detection
from photutils import aperture
from astropy import stats
from matplotlib import pyplot as plt

print("OK")
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", action="store", default=".", help="Input directory")
args = parser.parse_args()

attributes = ["DATE-OBS", "TELESCOP", "OBJECT", "IMAGETYP", "EXPTIME", "RA", "DEC"]
attrs_to_save = ["DATE-OBS", "TELESCOP", "INSTRUME", "OBJECT", "PROG-ID", "OBSERVAT", "DETECTOR"]
# attrs_to_calculate = ["Z"]

fits_names = [str(_) for _ in pathlib.Path(args.dir).glob("**/*.fits") if _ not in pathlib.Path(args.dir).glob("**/aimp/**/*.fits")] + [str(_) for _ in pathlib.Path(args.dir).glob("**/*.fts") if _ not in pathlib.Path(args.dir).glob("**/aimp/**/*.fts")]
fits_names = [_ for _ in fits_names]
print(f"{len(fits_names)} FITS files found in {args.dir}")
fits_files = []
i = 0
bias_nums = []
dark_nums = []
flat_nums = []
for f in fits_names:
    img = fits.open(f)
    fits_files.append(img)
    i = i + 1
    print(f"{i}) {f}:")
    for j in range(len(img)):
        hdu = img[j]
        if j == 0:
            print("  Primary HDU:")
        else:
            print(f"  HDU{j}:")
        for attr in attributes:
            if attr in hdu.header:
                print(f"\t    {attr} = {hdu.header[attr]}")
    if "IMAGETYP" in img[0].header and img[0].header["IMAGETYP"] == "bias":
        bias_nums.append(i)
    if "OBJECT" in img[0].header and img[0].header["OBJECT"] == "sunsky":
        flat_nums.append(i)
print("\n\nWarning: ONLY Primary HDUs will be processed in the next steps!")

print(f"\nFound {len(bias_nums)} BIAS frames: ", end="")
for _ in bias_nums:
    print(f"{_}", end=" ")
print("\nDo you want to select other frames? (n)")
if input() in ["y", "yes"]:
    print("\n\nSelect BIAS frames:")
    bias_nums = [int(_) for _ in input().split()]
print("\nDo you want to select DARK frames? (n)")
if input() in ["y", "yes"]:
    print("\nSelect DARK frames:")
    dark_nums = [int(_) for _ in input().split()]
print(f"\n\nFound {len(flat_nums)} FLAT frames: ", end="")
for _ in flat_nums:
    print(f"{_}", end=" ")
print("\nDo you want to select other frames? (n)")
if input() in ["y", "yes"]:
    print("\nSelect FLAT frames:")
    flat_nums = [int(_) for _ in input().split()]
print("\nSelect frames with object:")
obj_nums = [int(_) for _ in input().split()]
obj_files = [fits_files[i - 1] for i in obj_nums]
if len(obj_nums) == 0:
    print("\nNo frames selected!")
    exit(1)
else:
    print("\nSelected files:")
    for _ in obj_nums:
        print(fits_names[_ - 1])

shape = obj_files[0][0].data.shape
for i in range(len(obj_files)):
    if obj_files[i][0].data.shape != shape:
        print(f"Error: Data shape conflict in {fits_names[obj_nums[0]]} and {fits_names[obj_nums[i]]}")
        exit(2)
for i in bias_nums:
    if fits_files[i][0].data.shape != shape:
        print(f"Warning: Shape of BIAS frame in {fits_names[i]} is not equal to the object frame shape! This frame will be ignored.")
        bias_nums.remove(i)
bias_files = [fits_files[i - 1] for i in bias_nums]
for i in dark_nums:
    if fits_files[i][0].data.shape != shape:
        print(f"Warning: Shape of DARK frame in {fits_names[i]} is not equal to the object frame shape! This frame will be ignored.")
        dark_nums.remove(i)
dark_files = [fits_files[i - 1] for i in dark_nums]
for i in flat_nums:
    if fits_files[i][0].data.shape != shape:
        print(f"Warning: Shape of FLAT frame in {fits_names[i]} is not equal to the object frame shape! This frame will be ignored.")
        flat_nums.remove(i)
flat_files = [fits_files[i - 1] for i in flat_nums]

exptime = max([img[0].header["EXPTIME"] for img in obj_files])
datatype = numpy.dtype('f4')

if not os.path.exists(args.dir + "/aimp/tmp"):
    os.makedirs(args.dir + "/aimp/tmp")

if len(bias_nums) > 0:
    bias_data = numpy.ndarray(shape=shape, dtype=datatype)
    numpy.median([img[0].data for img in bias_files], axis=0, out=bias_data)
else:
    bias_data = numpy.zeros(shape=shape, dtype=datatype)
bias_header = fits.Header()
bias_header["IMAGETYP"] = "bias"
bias_header["EXPTIME"] = 0.0
bias_hdu = fits.PrimaryHDU(data=bias_data, header=bias_header)
fits.HDUList([bias_hdu]).writeto(args.dir + "/aimp/tmp/mean_bias.fits", overwrite=True)

if len(dark_nums) > 0:
    dark_data = numpy.ndarray(shape=shape, dtype=datatype)
    numpy.median([(img[0].data - bias_data) * (exptime / img[0].header["EXPTIME"]) for img in dark_files], axis=0, out=dark_data)
else:
    dark_data = numpy.zeros(shape=shape, dtype=datatype)
dark_header = fits.Header()
dark_header["EXPTIME"] = exptime
dark_hdu = fits.PrimaryHDU(data=dark_data, header=dark_header)
fits.HDUList([dark_hdu]).writeto(args.dir + "/aimp/tmp/mean_dark.fits", overwrite=True)

if len(flat_nums) > 0:
    flat_data = numpy.ndarray(shape=shape, dtype=datatype)
    numpy.median([(img[0].data - bias_data) * (exptime / img[0].header["EXPTIME"]) - dark_data for img in flat_files], axis=0, out=flat_data)
    valmax = numpy.max(flat_data)
    flat_data = flat_data / valmax
else:
    flat_data = numpy.ones(shape=shape, dtype=datatype)
flat_header = fits.Header()
flat_header["IMAGETYP"] = "obj"
flat_header["OBJECT"] = "sunsky"
flat_header["EXPTIME"] = exptime
flat_hdu = fits.PrimaryHDU(data=flat_data, header=flat_header)
fits.HDUList([flat_hdu]).writeto(args.dir + "/aimp/tmp/mean_flat.fits", overwrite=True)


obj_data = numpy.ndarray(shape=shape, dtype=datatype)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if numpy.sum(numpy.where(flat_data < 0.05, 1, 0)) > 0:
        print("Warning: FLAT contains very small elements (<0.05). Corresponding pixels of the result will be NaN")
    obj_images = [numpy.where(flat_data < 0.05, numpy.NaN, ((img[0].data - bias_data) * (exptime / img[0].header["EXPTIME"]) - dark_data) / flat_data) for img in obj_files]
obj_header = fits.Header()
obj_header["IMAGETYP"] = "obj"
obj_header["EXPTIME"] = sum([img[0].header["EXPTIME"] for img in obj_files])
scale = obj_files[0][0].header["IMSCALE"]
for img in obj_files:
    if img[0].header["IMSCALE"] != scale:
        print("Error: Image scale must be the same!")
        exit(3)
obj_header["IMSCALE"] = scale
for attr in attrs_to_save:
    attr0 = obj_files[0][0].header[attr]
    attr_stat = True
    for img in obj_files:
        if img[0].header[attr] != attr0:
            print(f"Warning: Non equal values of {attr} attribute")
            attr_stat = False
    if attr_stat:
        obj_header[attr] = attr0

all_sources = []
if len(obj_images) > 1:
    for i in range(len(obj_images)):
        img = obj_images[i]
        mean, median, std = stats.sigma_clipped_stats(img, sigma=3.0)
        detector = detection.DAOStarFinder(fwhm=7.0, threshold=7.0 * std)
        sources = detector(img - median)
        sources.add_index("id")
        all_sources.append(sources)
        positions = numpy.transpose((sources["xcentroid"], sources["ycentroid"]))
        apertures = aperture.CircularAperture(positions, r=8.0)
        norm = visualization.mpl_normalize.ImageNormalize(stretch=visualization.LogStretch())
        plt.subplot(1, len(obj_images), i + 1)
        plt.imshow(img, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
        apertures.plot(color='blue', lw=1.5, alpha=0.5)
        for star in sources:
            plt.annotate(f"{int(star['id'])}", xy=(float(star["xcentroid"]) + 6, float(star["ycentroid"]) + 6))
    print("\nMultiple object images selected. For correct alignment, please, select reference stars.")
    print("Select and write down the same stars on the images below (minimum 3 stars required)")
    print("Press Enter to show the images:")
    input()
    plt.show()
    print("Please, select the same stars on the images; write down their numbers in the correct order")
    ref_points_ids = []
    for i in range(len(obj_images)):
        print(f"Reference stars ids on image {i}:")
        ref_points_ids.append([int(s) for s in input().split()])
    n_points = len(ref_points_ids[0])
    if n_points < 3:
        print("Error: Minimum 3 reference stars required!")
    for pid in ref_points_ids:
        if len(pid) != n_points:
            print("Error: The number of reference stars must be the same on all images!")
            exit(4)
    dst_points_ids = ref_points_ids[0]
    dst_points = [(float((all_sources[0].loc["id", sid])["xcentroid"]), float((all_sources[0].loc["id", sid])["ycentroid"])) for sid in dst_points_ids]
    print("Selected stars:")
    print("  Image 0:")
    for j in range(len(dst_points_ids)):
        print(f"    # {dst_points_ids[j]}: {dst_points[j]}")
    for i in range(1, len(ref_points_ids)):
        scr_points_ids = ref_points_ids[i]
        scr_points = [(float((all_sources[i].loc["id", sid])["xcentroid"]), float((all_sources[i].loc["id", sid])["ycentroid"])) for sid in scr_points_ids]
        print(f"  Image {i}:")
        for j in range(len(scr_points_ids)):
            print(f"    # {scr_points_ids[j]}: {scr_points[j]}")
        tfm = numpy.float32([[1, 0, 0], [0, 1, 0]])
        A = []
        b = []
        for sp, dp in zip(scr_points, dst_points):
            A.append([sp[0], 0, sp[1], 0, 1, 0])
            A.append([0, sp[0], 0, sp[1], 0, 1])
            b.append(dp[0])
            b.append(dp[1])
        result, residuals, rank, s = numpy.linalg.lstsq(numpy.array(A), numpy.array(b), rcond=None)
        a0, a1, a2, a3, a4, a5 = result
        tfm = numpy.float32([[a0, a2, a4], [a1, a3, a5]])
        print("Found affine transform:")
        print(f"determinant = {numpy.linalg.det(tfm[:2, :2])}")
        print(f"x_shift = {tfm[0, 2]}")
        print(f"y_shift = {tfm[1, 2]}")
        tmp = numpy.ndarray(shape=(obj_images[i].shape[0], obj_images[i].shape[1], 3))
        tmp[:, :, 0] = obj_images[i]
        tmp[:, :, 1] = numpy.zeros(obj_images[i].shape)
        tmp[:, :, 2] = numpy.zeros(obj_images[i].shape)
        tmp = cv2.warpAffine(tmp, tfm, (obj_images[i].shape[1], obj_images[i].shape[0]), borderValue=numpy.nan)
        obj_images[i] = tmp[:, :, 0]

numpy.sum(obj_images, axis=0, out=obj_data)

obj_hdu = fits.PrimaryHDU(data=obj_data, header=obj_header)

print("Result file:")
filename = input()
if not (filename.endswith(".fts") or filename.endswith(".fits")):
    filename = filename + ".fits"
fits.HDUList([obj_hdu]).writeto(args.dir + "/aimp/" + filename, overwrite=True)
print(f"Result saved to: {args.dir + '/aimp/' + filename}")
print(f"BIAS, DARK and FLAT frames saved to directory: {args.dir + '/aimp/tmp'}")
