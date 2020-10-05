import numpy as np
import tensorflow as tf


# Quantizes points from physical space into cube space
class OccupancyCuboid:
    def __init__(self, delta, x_range=(-40, 40), y_range=(-1, 3), z_range=(0, 70.4)):
        self.delta = delta
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.grid_shape = (np.ceil((x_range[1] - x_range[0]) / delta[0]).astype(np.int32),
                           np.ceil((z_range[1] - z_range[0]) / delta[2]).astype(np.int32),
                           np.ceil((y_range[1] - y_range[0]) / delta[1]).astype(np.int32))

        self.Tout = (tf.bool,)  # For tf.data
        self.output_shape = self.grid_shape

    def encode(self, pts):
        # pts are in physical space; (ix, iy, iz) are in grid space
        ix = ((pts[2] - self.z_range[0]) / self.delta[2]).astype(np.int32)  # PZ
        iy = ((pts[0] - self.x_range[0]) / self.delta[0]).astype(np.int32)  # PX
        iz = ((pts[1] - self.y_range[0]) / self.delta[1]).astype(np.int32)  # PY

        # Remove pts outside of workspace
        mask = (ix >= 0) & (ix < self.grid_shape[1]) & \
               (iy >= 0) & (iy < self.grid_shape[0]) & \
               (iz >= 0) & (iz < self.grid_shape[2])
        ix = ix[mask]
        iy = iy[mask]
        iz = iz[mask]

        occupancy_grid = np.zeros(shape=self.grid_shape, dtype=np.bool)
        occupancy_grid[iy, ix, iz] = 1
        return occupancy_grid


# Sparse Occupancy Cuboid temporarily removed

def main():
    from datasets.kitti.reader import Reader, CARS_ONLY

    reader = Reader(CARS_ONLY)
    encoder = OccupancyCuboid(delta=(0.16, 0.10, 0.16), x_range=(-40, 40), y_range=(-1, 3), z_range=(0, 70.4))

    for t in reader.get_ids('train')[0:10]:
        pts = reader.get_velo_reduced(t)
        cube = encoder.encode(pts)
        print(cube.shape)


if __name__ == '__main__':
    main()
