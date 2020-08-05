import click
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree, Voronoi
from itertools import product

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
    img = Image.open(img_path)
    img = np.asarray(img)

    random_i = np.random.randint(0, img.shape[1], size=n_points)
    random_j = np.random.randint(0, img.shape[0], size=n_points)

    points = np.vstack((random_i, random_j)).T
    
    # Creating the kdtree
    vor = Voronoi(points)
    kdtree = cKDTree(vor.points)

    # Identifying regions pixel by pixel
    region_array = - np.ones(img.shape[:2])
    all_pos = product(range(img.shape[1]), range(img.shape[0]))
    positions = np.array(list(all_pos))
    distances, labels = kdtree.query(positions)

    for pos, label in zip(positions, labels):
        region_array[pos[1], pos[0]]= label

    # Storing colors
    colors = {}
    for region_id in np.unique(region_array):
        colors[region_id] = compute_mean_color(img, np.where(region_array == region_id))
    
    new_img = np.zeros(img.shape)
    for region_id in np.unique(region_array):
        region = np.where(region_array == region_id)
        new_img[region] = colors[region_id]

    new_img = new_img.astype(int)

    # Saving it
    PIL_image = Image.fromarray(new_img.astype('uint8'), 'RGB')
    PIL_image.save(out)

if __name__ == "__main__":
    process()
