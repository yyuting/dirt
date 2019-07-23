import os
import numpy as np
import numpy
import sys

import tensorflow as tf
import dirt
import skimage.io

canvas_width, canvas_height = 1024, 768


def get_dirt_pixels_render():

    square_vertices = tf.constant([[-1, -1, 0, 1], [-1, 1, 0, 1], [1, 1, 0, 1], [1, -1, 0, 1]], dtype=tf.float32)

    #background = skimage.io.imread('/n/fs/shaderml/datas_oceanic/test_img/test_middle_ground00000.png')
    #background = tf.constant(skimage.img_as_float(background), dtype=tf.float32)
    background = tf.random_normal([canvas_height, canvas_width, 3], dtype=tf.float32)
    
    camera_pos = tf.placeholder(tf.float32, 8)
    
    dirt_op = getattr(dirt, sys.argv[5])
    
    return dirt_op(
        vertices=square_vertices,
        faces=[[0, 1, 2], [0, 2, 3]],
        vertex_colors=tf.ones([4, 3]),
        background=background,
        camera_pos = camera_pos,
        height=canvas_height, width=canvas_width, channels=3
    ), camera_pos


def main():
    assert len(sys.argv) > 5
    
    cam_pos_file = sys.argv[1]
    sample_freq = int(sys.argv[2])
    outdir = sys.argv[3]
    prefix = sys.argv[4]
    
    par_vals = np.load(cam_pos_file)
    render_node, camera_pos = get_dirt_pixels_render()
    
    render_par_vals = np.empty([par_vals.shape[0], 8])
    
    sess = tf.Session()
    
    feed_dict = np.zeros(8)
    #feed_dict[1] = 100.0
    img = np.empty([canvas_height, canvas_width, 3])
    
    # for clip ocean4_00, roughly the height changes from 100 - 200 in the first 270 frames
    
    nsamples = 100
    
    if True:
        for i in range(0, render_par_vals.shape[0], sample_freq):
            #feed_dict[[0, 2]] = (np.random.random(2) - 0.5) * 500.0
            feed_dict[[0, 2]] = 0.0
            feed_dict[3:6] = par_vals[i, :3]
            #feed_dict[6] = np.random.random() * 180.0
            feed_dict[6] = i / 30
            feed_dict[7] = par_vals[i, 3]
            if i <= 270:
                feed_dict[1] = 100 * i / 270 + 100
            render_par_vals[i, :] = feed_dict[:]
            img[:] = 0.0
            for _ in range(nsamples):
                dirt_pixels = sess.run(render_node, feed_dict={camera_pos: feed_dict})
                img += dirt_pixels
            img /= nsamples
            skimage.io.imsave(os.path.join(outdir, '%s_%05d.png' % (prefix, i)), np.clip(img, 0.0, 1.0))
            print(i)
        np.save(os.path.join(outdir, '%s_camera_pos.npy' % prefix), render_par_vals)
        if sample_freq == 1:
            os.system('ffmpeg -i %s -r 30 -c:v libx264 -preset slow -crf 0 -r 30 %s'%(os.path.join(outdir, prefix + '_%05d.png'), os.path.join(outdir, 'video.mp4')))
   
    if True:

        # generate test frames along a given path
        nframes = 200
        test_render_par = np.empty([nframes, 8])
        for i in range(nframes):
            feed_dict[0] = i
            feed_dict[1] = 100 * i / nframes + 100
            feed_dict[2] = 0.5 * i
            feed_dict[3:6] = np.min(par_vals[:, :3], axis=0) + i / nframes * (np.max(par_vals[:, :3], axis=0) - np.min(par_vals[:, :3], axis=0))
            feed_dict[6] = i / 30
            feed_dict[7] = np.min(par_vals[:, 3], axis=0) + i / nframes * (np.max(par_vals[:, 3], axis=0) - np.min(par_vals[:, 3], axis=0))
            test_render_par[i, :] = feed_dict[:]
            img[:] = 0.0
            for _ in range(nsamples):
                dirt_pixels = sess.run(render_node, feed_dict={camera_pos: feed_dict})
                img += dirt_pixels
            img /= nsamples
            skimage.io.imsave(os.path.join(outdir, 'test_%s_%05d.png' % (prefix, i)), np.clip(img, 0.0, 1.0))
            print(i)
        np.save(os.path.join(outdir, 'test_%s_camera_pos.npy' % prefix), test_render_par)
        os.system('ffmpeg -i %s -r 30 -c:v libx264 -preset slow -crf 0 -r 30 %s'%(os.path.join(outdir, 'test_' + prefix + '_%05d.png'), os.path.join(outdir, 'test_video.mp4')))
    
if __name__ == '__main__':
    main()
        