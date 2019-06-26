
import numpy as np
import tensorflow as tf
import dirt
import skimage.io

canvas_width, canvas_height = 960, 640
centre_x, centre_y = 32, 64
square_size = 16


def get_non_dirt_pixels():
    xs, ys = tf.meshgrid(tf.range(canvas_width), tf.range(canvas_height))
    xs = tf.cast(xs, tf.float32) + 0.5
    ys = tf.cast(ys, tf.float32) + 0.5
    x_in_range = tf.less_equal(tf.abs(xs - centre_x), square_size / 2)
    y_in_range = tf.less_equal(tf.abs(ys - centre_y), square_size / 2)
    return tf.cast(tf.logical_and(x_in_range, y_in_range), tf.float32)


def get_dirt_pixels():

    square_vertices = tf.constant([[-1, -1, 0, 1], [-1, 1, 0, 1], [1, 1, 0, 1], [1, -1, 0, 1]], dtype=tf.float32)

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

        dirt_node = get_dirt_pixels()
        for i in range(2):
            dirt_pixels = dirt_node.eval()
            print(dirt_pixels)
            skimage.io.imsave('%d.png' % i, np.clip(dirt_pixels, 0.0, 1.0))
            

if __name__ == '__main__':
    main()

