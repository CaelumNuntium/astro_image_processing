import pathlib
import numpy
import warnings
from astropy.io import fits

attributes = ["DATE-OBS", "TELESCOP", "INSTRUME", "OBJECT", "IMAGETYP", "START", "EXPTIME", "RA", "DEC"]

fits_names = [str(_) for _ in pathlib.Path(".").glob("**/*.fits")] + [str(_) for _ in pathlib.Path(".").glob("**/*.fts")]
fits_names = [_ for _ in fits_names if not (_.startswith("mean_") or _ == "result.fits")]
print(f"{len(fits_names)} FITS files found")
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
        if "IMAGETYP" in hdu.header and hdu.header["IMAGETYP"] == "bias":
            bias_nums.append(i)
        if "OBJECT" in hdu.header and hdu.header["OBJECT"] == "sunsky":
            flat_nums.append(i)

print(f"\n\nFound {len(bias_nums)} BIAS frames: ", end="")
for _ in bias_nums:
    print(f"{_}", end=" ")
print("\nDo you want to select other frames? (n)")
if input() in ["y", "yes"]:
    print("\n\nSelect BIAS frames:")
    bias_nums = [int(_) for _ in input().split()]
bias_files = [fits_files[i - 1] for i in bias_nums]
print("\nDo you want to select DARK frames? (n)")
if input() in ["y", "yes"]:
    print("\nSelect DARK frames:")
    dark_nums = [int(_) for _ in input().split()]
dark_files = [fits_files[i - 1] for i in dark_nums]
print(f"\n\nFound {len(flat_nums)} FLAT frames: ", end="")
for _ in flat_nums:
    print(f"{_}", end=" ")
print("\nDo you want to select other frames? (n)")
if input() in ["y", "yes"]:
    print("\nSelect FLAT frames:")
    flat_nums = [int(_) for _ in input().split()]
flat_files = [fits_files[i - 1] for i in flat_nums]
print("\nSelect frames with object:")
obj_nums = [int(_) for _ in input().split()]
obj_files = [fits_files[i - 1] for i in obj_nums]
if len(obj_nums) == 0:
    print("\nNo frames selected!")
    exit()
else:
    print("\nSelected files:")
    for _ in obj_nums:
        print(fits_names[_ - 1])

exptime = max([img[0].header["EXPTIME"] for img in obj_files])
datatype = numpy.dtype('f4')

if len(bias_nums) > 0:
    bias_data = numpy.ndarray(shape=bias_files[0][0].data.shape, dtype=datatype)
    numpy.median([img[0].data for img in bias_files], axis=0, out=bias_data)
else:
    bias_data = numpy.zeros(shape=obj_files[0][0].data.shape, dtype=datatype)
bias_header = fits.Header()
bias_header["IMAGETYP"] = "bias"
bias_header["EXPTIME"] = 0.0
bias_hdu = fits.PrimaryHDU(data=bias_data, header=bias_header)
fits.HDUList([bias_hdu]).writeto("mean_bias.fits", overwrite=True)

if len(dark_nums) > 0:
    dark_data = numpy.ndarray(shape=dark_files[0][0].data.shape, dtype=datatype)
    numpy.median([(img[0].data - bias_data) * (exptime / img[0].header["EXPTIME"]) for img in dark_files], axis=0, out=dark_data)
else:
    dark_data = numpy.zeros(shape=obj_files[0][0].data.shape, dtype=datatype)
dark_header = fits.Header()
dark_header["EXPTIME"] = exptime
dark_hdu = fits.PrimaryHDU(data=dark_data, header=dark_header)
fits.HDUList([dark_hdu]).writeto("mean_dark.fits", overwrite=True)

if len(flat_nums) > 0:
    flat_data = numpy.ndarray(shape=flat_files[0][0].data.shape, dtype=datatype)
    numpy.median([(img[0].data - bias_data) * (exptime / img[0].header["EXPTIME"]) - dark_data for img in flat_files], axis=0, out=flat_data)
    valmax = numpy.max(flat_data)
    flat_data = flat_data / valmax
else:
    flat_data = numpy.ones(shape=obj_files[0][0].data.shape, dtype=datatype)
flat_header = fits.Header()
flat_header["IMAGETYP"] = "obj"
flat_header["OBJECT"] = "sunsky"
flat_header["EXPTIME"] = exptime
flat_hdu = fits.PrimaryHDU(data=flat_data, header=flat_header)
fits.HDUList([flat_hdu]).writeto("mean_flat.fits", overwrite=True)


obj_data = numpy.ndarray(shape=obj_files[0][0].data.shape, dtype=datatype)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if numpy.sum(numpy.where(flat_data < 0.05, 1, 0)) > 0:
        print("Warning: FLAT contains very small elements (<0.05). Corresponding pixels of the result will be taken equal to 0.")
    numpy.sum([numpy.where(flat_data < 0.05, 0, ((img[0].data - bias_data) * (exptime / img[0].header["EXPTIME"]) - dark_data) / flat_data) for img in obj_files], axis=0, out=obj_data)
obj_header = fits.Header()
obj_header["IMAGETYP"] = "obj"
obj_header["OBJECT"] = obj_files[0][0].header["OBJECT"]
obj_header["EXPTIME"] = exptime
obj_hdu = fits.PrimaryHDU(data=obj_data, header=obj_header)
fits.HDUList([obj_hdu]).writeto("result.fits", overwrite=True)
