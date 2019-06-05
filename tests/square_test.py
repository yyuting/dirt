
import numpy as np
import tensorflow as tf
import dirt
import skimage.io

canvas_width, canvas_height = 960, 640
centre_x, centre_y = 320, 480
square_size = 16


def get_non_dirt_pixels():
    xs, ys = tf.meshgrid(tf.range(canvas_width), tf.range(canvas_height))
    xs = tf.cast(xs, tf.float32) + 0.5
    ys = tf.cast(ys, tf.float32) + 0.5
    x_in_range = tf.less_equal(tf.abs(xs - centre_x), square_size / 2)
    y_in_range = tf.less_equal(tf.abs(ys - centre_y), square_size / 2)
    return tf.cast(tf.logical_and(x_in_range, y_in_range), tf.float32)


def get_dirt_pixels():

    # Build square in screen space
    square_vertices = tf.constant([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=tf.float32) * square_size - square_size / 2.
    square_vertices += [centre_x, centre_y]

    # Transform to homogeneous coordinates in clip space
    square_vertices = square_vertices * 2. / [canvas_width, canvas_height] - 1.
    square_vertices = tf.concat([square_vertices, tf.zeros([4, 1]), tf.ones([4, 1])], axis=1)

    return dirt.rasterise(
        vertices=square_vertices,
        faces=[[0, 1, 2], [0, 2, 3]],
        vertex_colors=tf.ones([4, 3]),
        background=tf.zeros([canvas_height, canvas_width, 3]),
        height=canvas_height, width=canvas_width, channels=3
    )


def main():

    session = tf.Session()
    with session.as_default():

        dirt_pixels = get_dirt_pixels().eval()
        #dirt_gt = get_non_dirt_pixels().eval()
        print(np.sum(dirt_pixels))
        print(dirt_pixels)
        print(dirt_pixels.shape)
        if dirt_pixels.shape[2] <= 4:
            skimage.io.imsave('test.png', np.clip(dirt_pixels, 0.0, 1.0))
        elif dirt_pixels.shape[2] > 8:
            # check if checkerboard trace is correct
            assert np.allclose(np.modf(dirt_pixels[:, :, 4:6])[0] - 0.5, dirt_pixels[:, :, 6:8])
            assert np.allclose(dirt_pixels[:, :, 0], dirt_pixels[:, :, 6] * dirt_pixels[:, :, 7] < 0)
            assert np.allclose(dirt_pixels[:, :, 1], dirt_pixels[:, :, 6] * dirt_pixels[:, :, 7] < 0)
            assert np.allclose(dirt_pixels[:, :, 2], dirt_pixels[:, :, 6] * dirt_pixels[:, :, 7] < 0)

        if dirt_pixels.shape[2] == 32:
            assert np.allclose(dirt_pixels[:, :, 4:8], dirt_pixels[:, :, 8:12])
            assert np.allclose(dirt_pixels[:, :, 4:8], dirt_pixels[:, :, 12:16])
            assert np.allclose(dirt_pixels[:, :, 4:8], dirt_pixels[:, :, 16:20])
            assert np.allclose(dirt_pixels[:, :, 4:8], dirt_pixels[:, :, 20:24])
            assert np.allclose(dirt_pixels[:, :, 4:8], dirt_pixels[:, :, 24:28])
            assert np.allclose(dirt_pixels[:, :, 4:8], dirt_pixels[:, :, 28:32])

if __name__ == '__main__':
    main()
