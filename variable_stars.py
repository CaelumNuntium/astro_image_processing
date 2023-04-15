print("Loading modules...")

import argparse
import pathlib
import numpy
from astropy.io import fits
from astropy import stats
from astropy import time
from astropy import visualization
from photutils import aperture
from photutils import detection
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("dir", action="store", default=".", help="input directory")
args = parser.parse_args()
fits_files = [str(f) for f in pathlib.Path(args.dir).glob("*.fit")] + [str(f) for f in pathlib.Path(args.dir).glob("*.fits")] + [str(f) for f in pathlib.Path(args.dir).glob("*.fts")]
images = []
headers = []
for f in fits_files:
    img = fits.open(f)[0]
    images.append(img.data)
    headers.append(img.header)
mean, median, std = stats.sigma_clipped_stats(images[0], sigma=10.0)
detector = detection.DAOStarFinder(fwhm=1.0, threshold=7.0 * std)
sources = detector(images[0] - median)
print(sources)
sources.add_index("id")
positions = numpy.transpose((sources["xcentroid"], sources["ycentroid"]))
apertures = aperture.CircularAperture(positions, r=4.0)
norm = visualization.mpl_normalize.ImageNormalize(stretch=visualization.LogStretch())
plt.imshow(images[0], cmap="Greys", origin="lower", norm=norm, interpolation="nearest")
apertures.plot(color="blue", lw=1.5, alpha=0.5)
for star in sources:
    plt.annotate(f"{int(star['id'])}", xy=(float(star["xcentroid"]) + 3, float(star["ycentroid"]) + 3))
print("Write down the numbers of variable stars and of standards.")
print("Press enter to show the image")
input()
plt.show()
print("Variable stars:")
var_stars = [int(s) for s in input().split()]
print("Standard stars:")
std_stars = [int(s) for s in input().split()]
var_coords = [(float((sources.loc["id", sid])["xcentroid"]), float((sources.loc["id", sid])["ycentroid"])) for sid in var_stars]
std_coords = [(float((sources.loc["id", sid])["xcentroid"]), float((sources.loc["id", sid])["ycentroid"])) for sid in std_stars]

