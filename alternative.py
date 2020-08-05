import click
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree, Voronoi
from codetiming import Timer


def compute_mean_color(img, region):
    """
    This function takes an image array and
    one of those tuples that np.where outputs.
    """
    region_slice = img[region]
    return np.mean(region_slice, axis=0)


@click.command()
@click.argument("img_path", type=str)
@click.option("--out", type=str, default="out.jpg")
@click.option("--n_points", type=int, default=300)
def process(img_path, out, n_points):
    # Loading the image
    with Timer(text="Loading the image: {:.6f} seconds"):
        img = Image.open(img_path)
        img = np.asarray(img)

        random_i = np.random.randint(0, img.shape[1], size=n_points)
        random_j = np.random.randint(0, img.shape[0], size=n_points)
        points = np.vstack((random_i, random_j)).T

    # Creating the kdtree
    with Timer(text="Creating the kdtree: {:.6f} seconds"):
        vor = Voronoi(points)
        kdtree = cKDTree(vor.points)

    # Identifying regions pixel by pixel
    with Timer(text="Computing positions: {:.6f} seconds"):
        region_array = -np.ones(img.shape[:2])
        positions = np.mgrid[0 : img.shape[1], 0 : img.shape[0]].T.reshape(-1, 2)

    with Timer(text="Identifying regions pixel by pixel: {:.6f} seconds"):
        distances, labels = kdtree.query(positions)

    with Timer(text="Labelizing: {:.6f} seconds"):
        y, x = positions.T
        region_array[x, y] = labels

    # Storing colors
    with Timer(text="Storing colors: {:.6f} seconds"):
        new_img = np.zeros(img.shape)
        for region_id in np.unique(labels):
            region = np.where(region_array == region_id)
            new_img[region] = compute_mean_color(img, region)

    # Saving it
    with Timer(text="Saving: {:.6f} seconds"):
        PIL_image = Image.fromarray(new_img.astype("uint8"), "RGB")
        PIL_image.save(out)


if __name__ == "__main__":
    process()
