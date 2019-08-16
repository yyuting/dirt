
import numpy as np
import tensorflow as tf
import dirt
import skimage.io
import skimage
import time
import os

canvas_width, canvas_height = 960, 540
centre_x, centre_y = 32, 64
square_size = 16

def get_hill():

    square_vertices = tf.constant([[-1, -1, 0, 1], [-1, 1, 0, 1], [1, 1, 0, 1], [1, -1, 0, 1]], dtype=tf.float32)

    background_np = np.load('/n/fs/shaderml/OpenSfM/data/hill1_00_full/terrain_lookup.npy')
    background = tf.constant(background_np, dtype=tf.float32)
    skimage.io.imsave('height.png', background_np[:, :, 0])
    
    camera_pos = tf.placeholder(tf.float32, 9)
    
    return dirt.hill(
        vertices=square_vertices,
        faces=[[0, 1, 2], [0, 2, 3]],
        vertex_colors=tf.ones([4, 3]),
        background=background,
        camera_pos = camera_pos,
        height=canvas_height, width=canvas_width, channels=3
    ), camera_pos


def main():
    
    node, camera = get_hill()
    sess = tf.Session()
    arr = sess.run(node, feed_dict={camera: np.zeros(9)})
    print(np.max(arr))
    print(arr)
    skimage.io.imsave('test.png', np.clip(arr, 0, 1))
    return
    
    nsamples = 100
    dir = '/n/fs/shaderml/drone_videos/drone_shader_frames'
    out_dir = '/n/fs/shaderml/drone_videos/drone_shader_frames/glsl_render'
    name = 'v2'
    camera_pos_vals = np.load(os.path.join(dir, 'camera_pos_' + name + '.npy'))
    render_t  = np.load(os.path.join(dir, 'render_t_' + name + '.npy'))
    nframes = camera_pos_vals.shape[0]
    feed_dict_arr = np.empty(9)
    feed_dict_arr[7] = 0.9
    img = np.zeros([640, 960, 3])
        
    session = tf.Session()
    with session.as_default():

        if False:
            dirt_node, camera_pos = get_dirt_pixels()
            dirt_node2, camera_pos2 = get_dirt_pixels_render()
            dirt_node3, camera_pos3 = get_oceanic_no_cloud()
            dirt_node5, camera_pos5, bg = get_oceanic_opt_flow()
        
        dirt_node4, camera_pos4 = get_oceanic_still_cloud()
        
        
        #feed_dict_arr[:6] = camera_pos_vals[0, :]
        feed_dict_arr[:6] = np.array([0.0, 100, 0, 0, 0.3, 0])
        feed_dict_arr[6] = render_t[0]
        
        if False:
            arr1 = session.run(dirt_node, feed_dict={camera_pos: feed_dict_arr})
            arr2 = session.run(dirt_node2, feed_dict={camera_pos2: feed_dict_arr})
            arr3 = session.run(dirt_node3, feed_dict={camera_pos3: feed_dict_arr})
        
        for i in range(10):
            feed_dict_arr[8] = 0.1 * i
            arr4 = session.run(dirt_node4, feed_dict={camera_pos4: feed_dict_arr})
            skimage.io.imsave('test%d.png' % i, np.clip(arr4, 0.0, 1.0))
        return
        
        feed_dict_arr_extended = np.empty(15)
        feed_dict_arr_extended[:8] = feed_dict_arr[:]
        dt = 1 / 30
        dx = 30.0
        dy = 14.0
        dz = 15.0
        dang1 = 0.1
        dang2 = 0.1
        dang3 = 0.1
        feed_dict_arr_extended[8] = dt
        feed_dict_arr_extended[9] = dx
        feed_dict_arr_extended[10] = dy
        feed_dict_arr_extended[11] = dz
        feed_dict_arr_extended[12] = dang1
        feed_dict_arr_extended[13] = dang2
        feed_dict_arr_extended[14] = dang3
        
        feed_dict_arr_extended[:6] += dt * feed_dict_arr_extended[9:]
        feed_dict_arr_extended[6] += dt
        arr5 = session.run(dirt_node5,feed_dict={camera_pos5: feed_dict_arr_extended, bg: arr4})
        
        feed_dict_arr = feed_dict_arr_extended[:8]
        arr6 = session.run(dirt_node4, feed_dict={camera_pos4: feed_dict_arr})
        
        skimage.io.imsave('frame_before.png', np.clip(arr4, 0, 1))
        skimage.io.imsave('frame_after.png', np.clip(arr6, 0, 1))
        skimage.io.imsave('frame_before_wrapped_to_after.png', np.clip(arr5, 0, 1))
        
        return
        skimage.io.imsave('test.png', np.clip(arr2, 0, 1))
        skimage.io.imsave('test2.png', np.clip(arr1[:, :, 0], 0, 1))
        skimage.io.imsave('test3.png', np.clip(arr1[:, :, 1], 0, 1))
        skimage.io.imsave('test4.png', np.clip(arr3, 0, 1))
        skimage.io.imsave('test5.png', np.clip(arr4, 0, 1))
        
        if False:
            for idx in range(nframes):
                feed_dict_arr[:6] = camera_pos_vals[idx, :]
                feed_dict_arr[6] = render_t[idx]

                img[:] = 0.0
                timeA = time.time()
                for i in range(nsamples):
                    dirt_pixels = session.run(dirt_node, feed_dict={camera_pos: feed_dict_arr})
                    img += dirt_pixels
                    #print(dirt_pixels)
                    #skimage.io.imsave('%d.png' % i, np.clip(dirt_pixels, 0.0, 1.0))
                timeB = time.time()
                print(idx, timeB - timeA)
                img /= nsamples
                skimage.io.imsave(os.path.join(out_dir, '%s_%05d.png' % (name, idx)), np.clip(img, 0.0, 1.0))
                #skimage.io.imsave('test.png', np.clip(img, 0.0, 1.0))
            

if __name__ == '__main__':
    main()

