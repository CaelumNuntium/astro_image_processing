print("Loading modules...")

import argparse
import math
import numpy
import cv2
from astropy.io import fits
from astropy import visualization
from astropy import stats
from photutils import detection
from photutils import aperture
from matplotlib import pyplot as plt


def align_images(obj_images):
    all_sources = []
    n_rows = math.floor(math.sqrt(len(obj_images) / 2))
    n_cols = math.ceil(len(obj_images) / n_rows)
    for i in range(len(obj_images)):
        im = obj_images[i]
        mean, median, std = stats.sigma_clipped_stats(im, sigma=3.0)
        detector = detection.DAOStarFinder(fwhm=7.0, threshold=7.0 * std)
        sources = detector(im - median)
        sources.add_index("id")
        all_sources.append(sources)
        positions = numpy.transpose((sources["xcentroid"], sources["ycentroid"]))
        apertures = aperture.CircularAperture(positions, r=8.0)
        norm = visualization.mpl_normalize.ImageNormalize(stretch=visualization.LogStretch())
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(im, cmap="Greys", origin="lower", norm=norm, interpolation="nearest")
        apertures.plot(color="blue", lw=1.5, alpha=0.5)
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
    dst_points = [
        (float((all_sources[0].loc["id", sid])["xcentroid"]), float((all_sources[0].loc["id", sid])["ycentroid"])) for
        sid in dst_points_ids]
    print("Selected stars:")
    print("  Image 0:")
    for j in range(len(dst_points_ids)):
        print(f"    # {dst_points_ids[j]}: {dst_points[j]}")
    for i in range(1, len(ref_points_ids)):
        scr_points_ids = ref_points_ids[i]
        scr_points = [
            (float((all_sources[i].loc["id", sid])["xcentroid"]), float((all_sources[i].loc["id", sid])["ycentroid"]))
            for sid in scr_points_ids]
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


p = [1, 1, 1]

print("OK")
parser = argparse.ArgumentParser()
parser.add_argument("R", action="store", help="R filter image")
parser.add_argument("G", action="store", help="G filter image")
parser.add_argument("B", action="store", help="B filter image")
parser.add_argument("-d", "--dir", action="store", default=".", help="directory (default value - current directory)")
args = parser.parse_args()
r_fits = fits.open(args.dir + "/" + args.R)
g_fits = fits.open(args.dir + "/" + args.G)
b_fits = fits.open(args.dir + "/" + args.B)
print("\nWarning: ONLY Primary HDUs will be processed!")
r_exp = float(r_fits[0].header["EXPTIME"])
g_exp = float(g_fits[0].header["EXPTIME"])
b_exp = float(b_fits[0].header["EXPTIME"])
exp = min([r_exp, g_exp, b_exp])
print(f"\nTotal exptime: {3 * exp} s")
images = [r_fits[0].data, g_fits[0].data, b_fits[0].data]
images[0] = images[0] * (exp / r_exp) * p[0]
images[1] = images[1] * (exp / g_exp) * p[1]
images[2] = images[2] * (exp / b_exp) * p[2]
align_images(images)
max_val = []
mean_val = []
for i in range(len(images)):
    images[i] = numpy.sqrt(images[i])
    images[i] = numpy.where(numpy.isnan(images[0] + images[1] + images[2]), 0.0, images[i])
rgb_image_tmp = numpy.array(images)
print("Result filename:")
filename = input()
if not filename.endswith(".bmp"):
    filename = filename + ".bmp"
filename = args.dir + "/" + filename
rgb_image = numpy.ndarray((rgb_image_tmp.shape[1], rgb_image_tmp.shape[2], 3), dtype='i4')
rgb_image[:, :, 0] = rgb_image_tmp[2, :, :]
rgb_image[:, :, 1] = rgb_image_tmp[1, :, :]
rgb_image[:, :, 2] = rgb_image_tmp[0, :, :]
cv2.imwrite(filename, rgb_image)
