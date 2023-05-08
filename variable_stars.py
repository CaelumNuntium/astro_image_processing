print("Loading modules...")

import os
import math
import argparse
import pathlib
import numpy
from scipy import optimize
from astropy.io import fits
from astropy import stats
from astropy import visualization
from photutils import aperture
from matplotlib import pyplot as plt


def make_2d_gaussian_1d_repr(xy, x0, y0, sigma_x, sigma_y, max_intensity, position_angle):
    f = max_intensity * 2 * sigma_x * sigma_y
    (x, y) = xy
    x0 = float(x0)
    y0 = float(y0)
    a = (numpy.cos(position_angle) ** 2) / (2 * sigma_x ** 2) + (numpy.sin(position_angle) ** 2) / (2 * sigma_y ** 2)
    b = -(numpy.sin(2 * position_angle)) / (4 * sigma_x ** 2) + (numpy.sin(2 * position_angle)) / (4 * sigma_y ** 2)
    c = (numpy.sin(position_angle) ** 2) / (2 * sigma_x ** 2) + (numpy.cos(position_angle) ** 2) / (2 * sigma_y ** 2)
    intensity = max_intensity * numpy.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))
    return intensity.ravel()


def flux(sigma, max_intensity):
    sigma_a = sigma[0]
    sigma_b = sigma[1]
    f = max_intensity * sigma_a * sigma_b * 2 * math.pi
    return f


def make_model(image, model, init_parameters):
    repr_img = numpy.ravel(image)
    x = numpy.linspace(0, image.shape[0], image.shape[0])
    y = numpy.linspace(0, image.shape[1], image.shape[1])
    x, y = numpy.meshgrid(x, y)
    popt, pcov = optimize.curve_fit(model, (x, y), repr_img, init_parameters)
    data_fitted = model((x, y), *popt)
    data_fitted = numpy.reshape(data_fitted, newshape=image.shape)
    return popt, data_fitted


parser = argparse.ArgumentParser()
parser.add_argument("-f", action="store", help="filter")
parser.add_argument("--coords", action="store", default="coords.dat", help="file with the coordinates of the stars")
args = parser.parse_args()
directory = f"../Variable_Stars/{args.f}"
fits_files = [str(f) for f in pathlib.Path(directory).glob("*.fit")] + [str(f) for f in pathlib.Path(directory).glob("*.fits")] + [str(f) for f in pathlib.Path(directory).glob("*.fts")]
print(f"{len(fits_files)} FITS files found in {directory}")
images = {}
times = {}
for f in fits_files:
    img = fits.open(f)[0]
    images[os.path.basename(f)] = img.data
    times[os.path.basename(f)] = float(img.header["MJD_OBS"])
with open(args.coords, "r") as inp:
    coord_strings = inp.readlines()
star_coords = {}
for i in range(len(coord_strings)):
    if coord_strings[i].strip().endswith(":"):
        loc_coords = []
        for ll in range(4):
            if not coord_strings[i + ll + 1].strip().endswith(":") and len(coord_strings[i + ll + 1].strip()) > 0:
                p_str = coord_strings[i + ll + 1].strip().split()
                loc_coords.append((float(p_str[0]), float(p_str[1])))
        if len(loc_coords) == 4 and coord_strings[i].strip()[:-1] in images.keys():
            star_coords[coord_strings[i].strip()[:-1]] = loc_coords
j = 0
print(f"{len(star_coords)} images will be processed")
r = 4
fluxes = []
jds = []
errs = []
filenames = []
for f in star_coords.keys():
    img = images[f]
    if j == 0:
        norm = visualization.mpl_normalize.ImageNormalize(stretch=visualization.LogStretch())
        plt.imshow(img, cmap="Greys", origin="lower", norm=norm, interpolation="nearest")
        apertures = aperture.CircularAperture(star_coords[f], r=4.0)
        apertures.plot(color="red", lw=1.5, alpha=0.5)
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        fig.savefig(f"example_img_{args.f}.png", dpi=250)
        plt.show()
        p = star_coords[f][1]
        fluxes = []
        ann0 = aperture.CircularAnnulus(p, r_in=20, r_out=30)
        bk0 = aperture.ApertureStats(img, ann0, sigma_clip=stats.SigmaClip(sigma=3, maxiters=10)).median
        star_img0 = img[(round(p[1]) - 15):(round(p[1]) + 15), (round(p[0]) - 15):(round(p[0]) + 15)] - bk0
        params0, fitted_gaussian = make_model(star_img0, make_2d_gaussian_1d_repr, [14, 14, 10, 10, numpy.max(star_img0), 0])
        precise_coords0 = (params0[0], params0[1])
        plt.subplot(1, 2, 1)
        plt.imshow(star_img0)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Image")
        plt.subplot(1, 2, 2)
        plt.imshow(fitted_gaussian)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Gaussian model")
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        fig.savefig(f"psf_{args.f}.png", dpi=250)
        plt.show()
        # fig.clear()
        fwhm_list = []
        for st in star_coords[f]:
            ann = aperture.CircularAnnulus(st, r_in=20, r_out=30)
            bk_stats = aperture.ApertureStats(img, ann, sigma_clip=stats.SigmaClip(sigma=3, maxiters=10))
            bk = bk_stats.median
            star_img = img[(round(st[1]) - 15):(round(st[1]) + 15), (round(st[0]) - 15):(round(st[0]) + 15)] - bk
            params, fg = make_model(star_img, make_2d_gaussian_1d_repr, [14, 14, 10, 10, numpy.max(star_img), 0])
            # plt.subplot(1, 2, 1)
            # plt.imshow(star_img, origin="lower")
            # plt.xlabel("x")
            # plt.ylabel("y")
            # plt.title("Image")
            # plt.subplot(1, 2, 2)
            # plt.imshow(fg, origin="lower")
            # plt.xlabel("x")
            # plt.ylabel("y")
            # plt.title("Gaussian model")
            # plt.show()
            precise_coords = (params[0], params[1])
            fwhm_list.append((params[2] + params[3]) / 2 * 2.355)
        fwhm = numpy.mean(fwhm_list)
        print(f"FWHM = {fwhm}")
        r = 3 * fwhm + 1
        print(f"Aperture radius: r = {r}")
        # ap = aperture.CircularAperture(precise_coords0, r=r)
        # m_table = aperture.aperture_photometry(star_img0, ap)
        # ap_flux = float(m_table[0]["aperture_sum"])
        # print(ap_flux)
        # print()
        # full_flux = flux((params0[2], params0[3]), params0[4])
        # print(full_flux)
        # apr = ap_flux / full_flux
        # print(f"Aperture correction: {apr}")
    j += 1
    ier = 0
    loc_fluxes = []
    loc_errors = []
    for j in range(len(star_coords[f])):
        st = star_coords[f][j]
        ann = aperture.CircularAnnulus(st, r_in=r * 2, r_out=r * 3)
        bk_stats = aperture.ApertureStats(img, ann, sigma_clip=stats.SigmaClip(sigma=3, maxiters=10))
        bk = bk_stats.median
        bk_std = bk_stats.std
        star_img = img[(round(st[1]) - 15):(round(st[1]) + 15), (round(st[0]) - 15):(round(st[0]) + 15)] - bk
        try:
            params, fg = make_model(star_img, make_2d_gaussian_1d_repr, [14, 14, 10, 10, numpy.max(star_img), 0])
            precise_coords = (params[0], params[1])
        except RuntimeError:
            print(f"Cannot find precise coordinates of star in {f} image at ({st[0]}, {st[1]})")
            plt.imshow(star_img)
            plt.show()
            print("Do you want to use approximate coordinates? (y)")
            if input().startswith("n"):
                ier = 1
                break
            precise_coords = (15, 15)
        ap = aperture.CircularAperture(precise_coords, r=r)
        m_table = aperture.aperture_photometry(star_img, ap)
        fl = float(m_table[0]["aperture_sum"])
        loc_fluxes.append(fl)
        loc_errors.append(math.sqrt(fl) + bk_std * math.pi * r ** 2)
    if ier == 1:
        continue
    fluxes.append(loc_fluxes)
    jds.append(times[f])
    errs.append(loc_errors)
    filenames.append(f)
ref_mag = []
if args.f == "j":
    ref_mag = [8.71, 10.31]
elif args.f == "h":
    ref_mag = [7.80, 8.01]
elif args.f == "k":
    ref_mag = [7.42, 6.93]
rel_magnitudes1 = [ref_mag[1] - 2.5 * math.log10(fluxes[_][1] / fluxes[_][3]) for _ in range(len(fluxes))]
rel_errors = [math.sqrt((2.5 * errs[_][1] / fluxes[_][1]) ** 2 + (2.5 * errs[_][3] / fluxes[_][3])) + 0.005 for _ in range(len(fluxes))]
with open(f"relative_magnitude_{args.f}.dat", "w") as out:
    for jj in range(len(rel_magnitudes1)):
        out.write(f"{jds[jj]} {rel_magnitudes1[jj]} {rel_errors[jj]} {filenames[jj]}\n")
plt.errorbar(jds, rel_magnitudes1, rel_errors, fmt="o", ecolor="#00FFFF", color="#FF00FF")
plt.plot(list(range(round(min(jds)), round(max(jds)))), [numpy.mean(rel_magnitudes1)] * (round(max(jds)) - round(min(jds))), color="#FF0000")
plt.xlabel("JD")
plt.ylabel("magnitude of the 1st standard measured by the 2nd standard")
fig = plt.gcf()
fig.set_size_inches(12, 6)
fig.savefig(f"relative_magnitude_{args.f}.png", dpi=250)
plt.show()
zero_points = [numpy.mean([ref_mag[0] + 2.5 * math.log10(fluxes[_][1]), ref_mag[1] + 2.5 * math.log10(fluxes[_][3])]) for _ in range(len(fluxes))]
zero_point_errors = [math.sqrt((2.5 * errs[_][1] / fluxes[_][1] / 2) ** 2 + (2.5 * errs[_][3] / fluxes[_][3] / 2) ** 2) + 0.005 for _ in range(len(fluxes))]
var_magnitudes1 = [zero_points[_] - 2.5 * math.log10(fluxes[_][0]) for _ in range(len(fluxes))]
var_errors1 = [math.sqrt((2.5 * errs[_][0] / fluxes[_][0]) ** 2 + zero_point_errors[_] ** 2) for _ in range(len(fluxes))]
var_magnitudes2 = [zero_points[_] - 2.5 * math.log10(fluxes[_][2]) for _ in range(len(fluxes))]
var_errors2 = [math.sqrt((2.5 * errs[_][2] / fluxes[_][2]) ** 2 + zero_point_errors[_] ** 2) for _ in range(len(fluxes))]
with open(f"var_star1_{args.f}.dat", "w") as out:
    out.write(f"# {len(var_magnitudes1) + 1}\n")
    for jj in range(len(var_magnitudes1)):
        out.write(f"{jds[jj]} {var_magnitudes1[jj]} {1 / var_errors1[jj]}\n")
plt.errorbar(jds, var_magnitudes1, var_errors1, fmt="o", ecolor="#00FFFF", color="#F000FF")
plt.xlabel("JD")
plt.ylabel("magnitude")
fig = plt.gcf()
fig.set_size_inches(12, 6)
fig.savefig(f"var_star1_{args.f}.png", dpi=250)
plt.show()
with open(f"var_star2_{args.f}.dat", "w") as out:
    out.write(f"# {len(var_magnitudes2) + 1}\n")
    for jj in range(len(var_magnitudes1)):
        out.write(f"{jds[jj]} {var_magnitudes2[jj]} {1 / var_errors2[jj]}\n")
plt.errorbar(jds, var_magnitudes2, var_errors2, fmt="o", ecolor="#00FFFF", color="#F000FF")
plt.xlabel("JD")
plt.ylabel("magnitude")
fig = plt.gcf()
fig.set_size_inches(12, 6)
fig.savefig(f"var_star2_{args.f}.png", dpi=250)
plt.show()
