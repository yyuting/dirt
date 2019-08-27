import os
import numpy as np
import numpy
import sys

import tensorflow as tf
import dirt
import skimage.io

canvas_width, canvas_height = 960, 640

def get_dirt_pixels_render(render_op, par_dims=8):

    square_vertices = tf.constant([[-1, -1, 0, 1], [-1, 1, 0, 1], [1, 1, 0, 1], [1, -1, 0, 1]], dtype=tf.float32)

    background = tf.random_normal([canvas_height, canvas_width, 3], dtype=tf.float32)
    dirt_op = getattr(dirt, render_op)

    camera_pos = tf.placeholder(tf.float32, par_dims)
        
    return dirt_op(
        vertices=square_vertices,
        faces=[[0, 1, 2], [0, 2, 3]],
        vertex_colors=tf.ones([4, 3]),
        background=background,
        camera_pos = camera_pos,
        height=canvas_height, width=canvas_width, channels=3
    ), camera_pos, background

def get_par_dims(render_op):
    if render_op == 'oceanic_still_cloud':
        par_dims = 9
    else:
        par_dims = 8
    return par_dims


def main():
    assert len(sys.argv) > 5
    
    outdir = sys.argv[1]
    prefix = sys.argv[2]
    simple_render_op = sys.argv[3]
    complex_render_op = sys.argv[4]
    nvideos = int(sys.argv[5])
    
    nframes_per_video = 30
    framerate = 30
    
    simple_par_dims = get_par_dims(simple_render_op)
    complex_par_dims = get_par_dims(complex_render_op)
        
    global canvas_width, canvas_height
    if complex_render_op.startswith('ocean'):
        canvas_width = 960
        canvas_height = 540
        nsamples = 100
    else:
        canvas_width = 1024
        canvas_height = 768
        nsamples = 1

    render_par_vals = np.empty([nvideos, nframes_per_video, max(simple_par_dims, complex_par_dims)])
    
    ans = get_dirt_pixels_render(simple_render_op, simple_par_dims)
    simple_render_node = ans[0]
    simple_camera_pos = ans[1]

    ans = get_dirt_pixels_render(complex_render_op, complex_par_dims)
    complex_render_node = ans[0]
    complex_camera_pos = ans[1]
    
    sess = tf.Session()
    
    img = np.empty([canvas_height, canvas_width, 3])
            
    feed_dict = {}
    simple_camera_pos_val = np.empty(simple_par_dims)
    complex_camera_pos_val = np.empty(complex_par_dims)
    
    speed_max = np.zeros(6)
    # x speed
    speed_max[0] = 150.0
    # y speed
    speed_max[1] = 50.0
    # z speed
    speed_max[2] = 150.0
    # ang1 speed
    speed_max[3] = 0.1
    # ang2 speed
    speed_max[4] = 0.2
    # ang3 speed
    speed_max[5] = 0.1
    speed_max /= 2
    
    def h00(t):
        return 2 * t ** 3 - 3 * t ** 2 + 1

    def h10(t):
        return t ** 3 - 2 * t ** 2 + t

    def h01(t):
        return -2 * t ** 3 + 3 * t ** 2

    def h11(t):
        return t ** 3 - t ** 2

    for ind_v in range(nvideos):
        
        ini_camera_pos_val = np.random.random(9)
        ini_camera_pos_scale = np.array([1000, 500, 1000, 0.4, 2*np.pi, 0.4, 180, 2.0, 180])
        ini_camera_pos_bias = np.array([0, 100, 0, -0.1, 0, -0.2, 0.0, 0.2, 0.0])
        
        ini_camera_pos_val *= ini_camera_pos_scale
        ini_camera_pos_val += ini_camera_pos_bias
        
        ini_camera_pos_only = ini_camera_pos_val[:6]
        
        speed1 = np.random.random(6) * 2.0 * speed_max - speed_max
        speed2 = np.random.random(6) * 2.0 * speed_max - speed_max
        
        mid_camera_pos_only = ini_camera_pos_only + speed1 * nframes_per_video / 2 / framerate
        final_camera_pos_only = mid_camera_pos_only + speed2 * nframes_per_video / 2 / framerate
        
        interp_p = [ini_camera_pos_only, 
                    mid_camera_pos_only, 
                    final_camera_pos_only]
        
        interp_unscaled_m = [np.random.random(6) * 2.0 * speed_max - speed_max, 
                             final_camera_pos_only - ini_camera_pos_only,
                             np.random.random(6) * 2.0 * speed_max - speed_max]
        #interp_unscaled_m = [np.zeros(6), final_camera_pos_only - ini_camera_pos_only, np.zeros(6)]
        
        
        for ind_f in range(nframes_per_video):
            
            if ind_f < nframes_per_video / 2:
                p0 = interp_p[0]
                p1 = interp_p[1]
                m0 = interp_unscaled_m[0] / (nframes_per_video)
                m1 = interp_unscaled_m[1] / (nframes_per_video)
                t = ind_f
            else:
                p0 = interp_p[1]
                p1 = interp_p[2]
                m0 = interp_unscaled_m[1] / (nframes_per_video)
                m1 = interp_unscaled_m[2] / (nframes_per_video)
                t = ind_f - nframes_per_video / 2
                
            t_scale = nframes_per_video / 2
            t /= t_scale
                
            simple_camera_pos_val[:6] = h00(t) * p0 + \
                                        h10(t) * m0 * t_scale + \
                                        h01(t) * p1 + \
                                        h11(t) * m1 * t_scale
            complex_camera_pos_val[:6] = simple_camera_pos_val[:6]
            
            simple_camera_pos_val[6] = ini_camera_pos_val[6] + ind_f / framerate
            complex_camera_pos_val[6] = simple_camera_pos_val[6]
            
            simple_camera_pos_val[7:] = ini_camera_pos_val[7:simple_par_dims]
            complex_camera_pos_val[7:] = ini_camera_pos_val[7:complex_par_dims]
                        
            render_par_vals[ind_v, ind_f, :simple_par_dims] = simple_camera_pos_val[:]
            render_par_vals[ind_v, ind_f, :complex_par_dims] = complex_camera_pos_val[:]

            feed_dict[simple_camera_pos] = simple_camera_pos_val
            
            img[:] = 0.0
            for _ in range(nsamples):
                dirt_pixels = sess.run(simple_render_node, feed_dict=feed_dict)
                img += dirt_pixels
            img /= nsamples
            
            skimage.io.imsave(os.path.join(outdir, '%s_%05d_simple_%05d.png' % (prefix, ind_v, ind_f)), np.clip(img, 0.0, 1.0))
            
            feed_dict[complex_camera_pos] = complex_camera_pos_val
            
            img[:] = 0.0
            for _ in range(nsamples):
                dirt_pixels = sess.run(complex_render_node, feed_dict=feed_dict)
                img += dirt_pixels
            img /= nsamples
            
            skimage.io.imsave(os.path.join(outdir, '%s_%05d_complex_%05d.png' % (prefix, ind_v, ind_f)), np.clip(img, 0.0, 1.0))
            
            print(ind_v, ind_f)
    
    np.save(os.path.join(outdir, '%s_camera_pos.npy' % prefix), render_par_vals)   

if __name__ == '__main__':
    main()
        