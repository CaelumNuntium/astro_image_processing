import math
import numpy as np
import cv2
from astropy.io import fits
from astropy import stats
from astropy import visualization
from photutils import detection
from photutils import aperture
from matplotlib import pyplot as plt


class Line(object):
    def __init__(self, k, b):
        self.k = k
        self.b = b

    def __call__(self, x):
        return self.k * x + self.b


class MyArray(object):
    def __init__(self, arr):
        self.array = arr

    def __getitem__(self, item):
        try:
            return self.array[item]
        except IndexError:
            return 0        


def slicing(image, line, step, datatype='f4'):
    image = MyArray(image)
    shape = image.array.shape
    if 0 <= line(0) <= shape[1]:
        xl = 0
        yl = line(0)
    elif line(0) < 0:
        xl = -line.b / line.k
        yl = 0
    else:
        xl = (shape[1] - line.b) / line.k
        yl = shape[1]
    if 0 <= line(shape[0]) <= shape[1]:
        xr = shape[0]
        yr = line(shape[0])
    elif line(shape[0]) < 0:
        xr = -line.b / line.k
        yr = 0
    else:
        xr = (shape[1] - line.b) / line.k
        yr = shape[1]
    n = math.floor(math.sqrt((xr - xl) ** 2 + (yr - yl) ** 2) / step)
    dx = step * math.sqrt(1 / (line.k ** 2 + 1))
    res = np.ndarray(shape=(2, n), dtype=datatype)
    for i in range(n):
        x = xl + i * dx
        y = line(x)
        ix = math.floor(x)
        iy = math.floor(y)
        fx = x - ix
        fy = y - iy
        res[0, i] = math.sqrt((x - xl) ** 2 + (y - yl) ** 2)
        res[1, i] = (1 - fx) * (1 - fy) * image[ix, iy] + fy * (1 - fx) * image[ix, iy + 1] + fx * (1 - fy) * image[ix + 1, iy] + fx * fy * image[ix + 1, iy + 1]
    return res


def align_images(obj_images, ref_stars=[]):
    all_sources = []
    n_rows = math.floor(math.sqrt(len(obj_images) / 2))
    n_cols = math.ceil(len(obj_images) / n_rows)
    for ii in range(len(obj_images)):
        im = obj_images[ii]
        mean, median, std = stats.sigma_clipped_stats(im, sigma=3.0)
        detector = detection.DAOStarFinder(fwhm=7.0, threshold=7.0 * std)
        sources = detector(im - median)
        sources.add_index("id")
        all_sources.append(sources)
        positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
        if len(ref_stars) == 0:
            apertures = aperture.CircularAperture(positions, r=8.0)
            norm = visualization.mpl_normalize.ImageNormalize(stretch=visualization.LogStretch())
            plt.subplot(n_rows, n_cols, ii + 1)
            plt.imshow(im, cmap="Greys", origin="lower", norm=norm, interpolation="nearest")
            apertures.plot(color="blue", lw=1.5, alpha=0.5)
            for star in sources:
                plt.annotate(f"{int(star['id'])}", xy=(float(star["xcentroid"]) + 6, float(star["ycentroid"]) + 6))
    ref_points_ids = []
    if len(ref_stars) == 0:
        print("\nMultiple object images selected. For correct alignment, please, select reference stars.")
        print("Select and write down the same stars on the images below (minimum 3 stars required)")
        print("Press Enter to show the images:")
        input()
        plt.show()
        print("Please, select the same stars on the images; write down their numbers in the correct order")
        for ii in range(len(obj_images)):
            print(f"Reference stars ids on image {ii}:")
            ref_points_ids.append([int(s) for s in input().split()])
    elif len(ref_stars) != len(obj_images):
        print("Error: Reference stars must be selected on all images!")
        exit(4)
    else:
        ref_points_ids = ref_stars
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
    for ii in range(1, len(ref_points_ids)):
        scr_points_ids = ref_points_ids[ii]
        scr_points = [
            (float((all_sources[ii].loc["id", sid])["xcentroid"]), float((all_sources[ii].loc["id", sid])["ycentroid"]))
            for sid in scr_points_ids]
        print(f"  Image {ii}:")
        for j in range(len(scr_points_ids)):
            print(f"    # {scr_points_ids[j]}: {scr_points[j]}")
        tfm = np.float32([[1, 0, 0], [0, 1, 0]])
        A = []
        b = []
        for sp, dp in zip(scr_points, dst_points):
            A.append([sp[0], 0, sp[1], 0, 1, 0])
            A.append([0, sp[0], 0, sp[1], 0, 1])
            b.append(dp[0])
            b.append(dp[1])
        result, residuals, rank, s = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
        a0, a1, a2, a3, a4, a5 = result
        tfm = np.float32([[a0, a2, a4], [a1, a3, a5]])
        print("Found affine transform:")
        print(f"determinant = {np.linalg.det(tfm[:2, :2])}")
        print(f"x_shift = {tfm[0, 2]}")
        print(f"y_shift = {tfm[1, 2]}")
        tmp = np.ndarray(shape=(obj_images[ii].shape[0], obj_images[ii].shape[1], 3))
        tmp[:, :, 0] = obj_images[ii]
        tmp[:, :, 1] = np.zeros(obj_images[ii].shape)
        tmp[:, :, 2] = np.zeros(obj_images[ii].shape)
        tmp = cv2.warpAffine(tmp, tfm, (obj_images[ii].shape[1], obj_images[ii].shape[0]), borderValue=np.nan)
        obj_images[ii] = tmp[:, :, 0]


def aperture_curve(image, center, r_max, step, datatype='f4'):
    n = math.floor(r_max / step)
    res = np.ndarray(shape=(2, n), dtype=datatype)
    for i in range(n):
        r = i * step
        res[0, i] = r
        if r == 0:
            res[1, i] = 0
        else:
            ap = aperture.CircularAperture(center, r=r)
            res[1, i] = float(aperture.aperture_photometry(image, ap)["aperture_sum"][0])
    return res


def to_magnitude(origin, exp_time, x_scale, y_scale, z, k, std_const):
    mu = -2.5 * np.log10(origin / exp_time) + 2.5 * math.log10(x_scale * y_scale)
    mu = mu - k / math.cos(z / 180 * math.pi)
    mu = mu + std_const
    return mu


fits_names = ["B_clean.fits", "I_clean.fits", "R_clean.fits", "V_clean.fits"]
# fits_names = ["B_row.fts"]
k_sao = [0.34, 0.08, 0.15, 0.21]
const = [29.8, 29.0, 29.3, 29.1]
center = (513, 513)
ref_points = [[46, 195, 145, 146], [59, 180, 134, 136], [59, 180, 137, 140], [53, 195, 147, 149]]
images = []
z_dist = []
exptime = []
x_scales = []
y_scales = []
for f in fits_names:
    hdul = fits.open(f)
    images.append(np.array(hdul[0].data, dtype='f4'))
    z_dist.append(float(hdul[0].header["Z"]))
    exptime.append(float(hdul[0].header["EXPTIME"]))
    scale = hdul[0].header["IMSCALE"].split("x")
    x_scales.append(float(scale[0]))
    y_scales.append(float(scale[1]))
align_images(images, ref_stars=ref_points)
for i in range(len(fits_names)):
    img = images[i]
    for k in range(-80, 90, 10):
        sl = slicing(img, Line(math.tan(k / 180 * math.pi), center[1] - math.tan(k / 180 * math.pi) * center[0]), 1)
        plt.xlabel(f"x, {k} degrees")
        plt.ylabel("Surface brightness, m/asec^2")
        plt.plot(sl[0], to_magnitude(sl[1], exptime[i], x_scales[i], y_scales[i], z_dist[i], k_sao[i], const[i]), color="#00FF00")
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        fig.savefig(f"{fits_names[i]}k{k}.png", dpi=250)
        # plt.show()
        fig.clear()
    crv = aperture_curve(img, center, 300, 5)
    plt.xlabel("r, asec")
    plt.ylabel("flux, m")
    plt.plot(crv[0] / x_scales[i], to_magnitude(crv[1], exptime[i], x_scales[i], y_scales[i], z_dist[i], k_sao[i], const[i]), color="#0000FF")
    m = 40 + np.nanmedian(img)
    print(f"{fits_names[i]}: {m}")
    for _ in range(1, len(crv[1])):
        if crv[1][_] - crv[1][_ - 1] < 2 * math.pi * crv[0][_] * 5 * m:
            print(f"Galaxy magnitude in {fits_names[i]}: {to_magnitude(crv[1][_], exptime[i], x_scales[i], y_scales[i], z_dist[i], k_sao[i], const[i])}")
            plt.plot(crv[0] / x_scales[i], [to_magnitude(crv[1][_], exptime[i], x_scales[i], y_scales[i], z_dist[i], k_sao[i], const[i])] * len(crv[1]), color="#FF0000")
            break
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    fig.savefig(f"{fits_names[i]}_aperture.png", dpi=250)
    fig.clear()
    # plt.show()
v_img = to_magnitude(np.where(images[3] > 1000, images[3], 0), exptime[3], x_scales[3], y_scales[3], z_dist[3], k_sao[3], const[3])
b_img = to_magnitude(np.where(images[0] > 1000, images[0], 0), exptime[0], x_scales[0], y_scales[0], z_dist[0], k_sao[0], const[0])
bv_img = v_img - b_img
# plt.imshow(bv_img)
# plt.show()
hdu_list = [fits.PrimaryHDU(data=bv_img)]
fits.HDUList(hdu_list).writeto("BV.fits", overwrite=True)
