print("Loading modules...")

import argparse
import pathlib
import numpy
from astropy.io import fits
from astropy import stats
from astropy import visualization
from photutils import aperture
from photutils import detection
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dir", action="store", default=".", help="input directory")
args = parser.parse_args()
fits_files = [str(f) for f in pathlib.Path(args.dir).glob("*.fit")] + [str(f) for f in pathlib.Path(args.dir).glob("*.fits")] + [str(f) for f in pathlib.Path(args.dir).glob("*.fts")]
example_file = "h2.fit"
example_opened = fits.open(example_file)
example_img = example_opened[0].data
example_opened.close()
example_coords = [(111, 110), (91, 85), (92, 57), (193, 94)]
with open("used.dat", "r") as inp:
    used_fits = [s.strip() for s in inp]
print(used_fits)
fits_files = [f for f in fits_files if f not in used_fits]
images = []
headers = []
fits_opened = []
for f in fits_files:
    fits_opened.append(fits.open(f))
for f in fits_opened:
    img = f[0]
    images.append(img.data)
print("Write down the numbers of the stars from the left image corresponding to the selected stars on the right image.")
print("Press enter to show the images")
input()
with open("coords.dat", "a") as out:
    with open("used.dat", "a") as used:
        for j in range(len(images)):
            img = images[j]
            mean, median, std = stats.sigma_clipped_stats(img, sigma=10.0)
            detector = detection.DAOStarFinder(fwhm=1.0, threshold=7.0 * std)
            sources = detector(img - median)
            sources.add_index("id")
            positions = numpy.transpose((sources["xcentroid"], sources["ycentroid"]))
            apertures = aperture.CircularAperture(positions, r=4.0)
            norm = visualization.mpl_normalize.ImageNormalize(stretch=visualization.LogStretch())
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap="Greys", origin="lower", norm=norm, interpolation="nearest")
            apertures.plot(color="blue", lw=1.5, alpha=0.5)
            for star in sources:
                plt.annotate(f"{int(star['id'])}", xy=(float(star["xcentroid"]) + 3, float(star["ycentroid"]) + 3))
            plt.subplot(1, 2, 2)
            plt.imshow(example_img, cmap="Greys", origin="lower", norm=norm, interpolation="nearest")
            apertures = aperture.CircularAperture(example_coords, r=4.0)
            apertures.plot(color="red", lw=1.5, alpha=0.5)
            for i in range(len(example_coords)):
                plt.annotate(f"{i}", xy=(example_coords[i][0] + 3, example_coords[i][1] + 3))
            plt.show()
            print("Stars:")
            stars = [int(s) for s in input().split()]
            var_coords = [(float((sources.loc["id", sid])["xcentroid"]), float((sources.loc["id", sid])["ycentroid"])) for sid in stars]
            out.write(f"{fits_files[j]}:\n")
            for point in var_coords:
                out.write(f"{point[0]}  {point[1]}\n")
            out.write("\n")
            used.write(f"{fits_files[j]}\n")
            if j % 4 == 3:
                print("Do you want to continue? (y)")
                if input().startswith("n"):
                    break
print("Thank you very much for your contribution!")
