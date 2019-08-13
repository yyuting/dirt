import os
import numpy as np
import numpy
import sys

import tensorflow as tf
import dirt
import skimage.io

canvas_width, canvas_height = 1024, 768
orig_img_dir = '/n/fs/shaderml/drone_videos/drone_frames/ocean3_00'


def get_dirt_pixels_render(is_opt_flow=False, par_dims=8):

    square_vertices = tf.constant([[-1, -1, 0, 1], [-1, 1, 0, 1], [1, 1, 0, 1], [1, -1, 0, 1]], dtype=tf.float32)

    #background = skimage.io.imread('/n/fs/shaderml/datas_oceanic/test_img/test_middle_ground00000.png')
    #background = tf.constant(skimage.img_as_float(background), dtype=tf.float32)
    if is_opt_flow:
        background = tf.placeholder(tf.float32, [canvas_height, canvas_width, 3])
        dirt_op = 'oceanic_opt_flow'
    else:
        background = tf.random_normal([canvas_height, canvas_width, 3], dtype=tf.float32)
        dirt_op = getattr(dirt, sys.argv[5])
    
    camera_pos = tf.placeholder(tf.float32, par_dims)
        
    return dirt_op(
        vertices=square_vertices,
        faces=[[0, 1, 2], [0, 2, 3]],
        vertex_colors=tf.ones([4, 3]),
        background=background,
        camera_pos = camera_pos,
        height=canvas_height, width=canvas_width, channels=3
    ), camera_pos, background


def main():
    assert len(sys.argv) > 5
    
    cam_pos_file = sys.argv[1]
    sample_freq = int(sys.argv[2])
    outdir = sys.argv[3]
    prefix = sys.argv[4]
        
    par_vals = np.load(cam_pos_file)
    
    
    
       
    is_opt_flow = (sys.argv[5] == 'oceanic_opt_flow')
    
    if is_opt_flow:
        par_dims = 15
    else:
        par_dims = 9
        
    
    camera_pos_val = np.zeros(par_dims)
    render_par_vals = np.empty([par_vals.shape[0], par_dims])
    
    ans = get_dirt_pixels_render(is_opt_flow, par_dims)
    render_node = ans[0]
    camera_pos = ans[1]
    bg = ans[2]
    
    sess = tf.Session()
    
    img = np.empty([canvas_height, canvas_width, 3])
    
    # for clip ocean4_00, roughly the height changes from 100 - 200 in the first 270 frames
    
    if is_opt_flow:
        # use 1spp (center of each pixel) to generate opt flow
        nsamples = 1
    else:
        nsamples = 100
        
    if sys.argv[5] == 'oceanic_still_cloud':
        is_still_cloud = True
    else:
        is_still_cloud = False
        
    feed_dict = {}
        
    if True:
        if is_opt_flow:
            ini_round_freq = 1
        else:
            ini_round_freq = sample_freq
        for i in range(0, render_par_vals.shape[0], ini_round_freq):
            #camera_pos_val[[0, 2]] = (np.random.random(2) - 0.5) * 500.0
            camera_pos_val[[0, 2]] = 0.0
            camera_pos_val[0] = -i * 0.1
            camera_pos_val[3:6] = par_vals[i, :3]
            #camera_pos_val[6] = np.random.random() * 180.0
            camera_pos_val[6] = i / 30
            camera_pos_val[7] = par_vals[i, 3]
            if i <= 270:
                camera_pos_val[1] = 60 * i / 270 + 100
                
            render_par_vals[i, :8] = camera_pos_val[:8]
            
        
            
        opt_flow_spacing = 10
        if is_opt_flow:
            restart = opt_flow_spacing
        else:
            restart = 0
            
        orig_img_shape = None
        
        for i in range(restart, render_par_vals.shape[0], sample_freq):
            camera_pos_val[:8] = render_par_vals[i, :8]
            if is_opt_flow:
                dt = render_par_vals[i, 6] - render_par_vals[i-opt_flow_spacing, 6]
                camera_pos_val[8] = dt
                camera_pos_val[9:] = (render_par_vals[i, :6] - render_par_vals[i-opt_flow_spacing, :6]) / dt
                raw_previous_frame = skimage.io.imread(os.path.join(orig_img_dir, '%05d.png' % (i-opt_flow_spacing)))
                previous_frame = skimage.transform.resize(raw_previous_frame, (img.shape[0], img.shape[1]))
                orig_img_shape = raw_previous_frame.shape[:2]
                feed_dict[bg] = previous_frame
            feed_dict[camera_pos] = camera_pos_val
            
            img[:] = 0.0
            for _ in range(nsamples):
                dirt_pixels = sess.run(render_node, feed_dict=feed_dict)
                img += dirt_pixels
            img /= nsamples
            # our interpolation seem to be better than glsl texture()
            if is_opt_flow:
                coord_y = img[:, :, 0]
                coord_x = canvas_height - 1 - img[:, :, 1]
                upper_right = previous_frame[np.clip(np.ceil(coord_x), 0, canvas_height-1).astype('i'), np.clip(np.ceil(coord_y), 0, canvas_width-1).astype('i'), :]
                upper_left = previous_frame[np.clip(np.ceil(coord_x)-1, 0, canvas_height-1).astype('i'), np.clip(np.ceil(coord_y), 0, canvas_width-1).astype('i'), :]
                lower_right = previous_frame[np.clip(np.ceil(coord_x), 0, canvas_height-1).astype('i'), np.clip(np.ceil(coord_y)-1, 0, canvas_width-1).astype('i'), :]
                lower_left = previous_frame[np.clip(np.ceil(coord_x)-1, 0, canvas_height-1).astype('i'), np.clip(np.ceil(coord_y)-1, 0, canvas_width-1).astype('i'), :]
                saved_img = upper_right * ((coord_x - np.ceil(coord_x)-1) * (coord_y - np.ceil(coord_y)-1))[:, :, np.newaxis] + \
                            upper_left * ((np.ceil(coord_x) - coord_x) * (coord_y - np.ceil(coord_y)-1))[:, :, np.newaxis] + \
                            lower_right * ((coord_x - np.ceil(coord_x)-1) * (np.ceil(coord_y) - coord_y))[:, :, np.newaxis] + \
                            lower_left * ((np.ceil(coord_x) - coord_x) * (np.ceil(coord_y) - coord_y))[:, :, np.newaxis]
                saved_img = skimage.transform.resize(saved_img[:, :, :], orig_img_shape)
            else:
                saved_img = img
                
            
            skimage.io.imsave(os.path.join(outdir, '%s_%05d.png' % (prefix, i)), np.clip(saved_img, 0.0, 1.0))
            print(i)
        np.save(os.path.join(outdir, '%s_camera_pos.npy' % prefix), render_par_vals)
        if sample_freq == 1:
            os.system('ffmpeg -i %s -r 30 -c:v libx264 -preset slow -crf 0 -r 30 %s'%(os.path.join(outdir, prefix + '_%05d.png'), os.path.join(outdir, 'video.mp4')))
   
    if True:

        # generate test frames along a given path
        nframes = 200
        test_render_par = np.empty([nframes, 8])
        for i in range(nframes):
            camera_pos_val[0] = i
            camera_pos_val[1] = 100 * i / nframes + 100
            camera_pos_val[2] = 0.5 * i
            camera_pos_val[3:6] = np.min(par_vals[:, :3], axis=0) + i / nframes * (np.max(par_vals[:, :3], axis=0) - np.min(par_vals[:, :3], axis=0))
            camera_pos_val[6] = i / 30
            camera_pos_val[7] = np.min(par_vals[:, 3], axis=0) + i / nframes * (np.max(par_vals[:, 3], axis=0) - np.min(par_vals[:, 3], axis=0))
            test_render_par[i, :] = camera_pos_val[:]
            img[:] = 0.0
            for _ in range(nsamples):
                dirt_pixels = sess.run(render_node, feed_dict={camera_pos: camera_pos_val})
                img += dirt_pixels
            img /= nsamples
            skimage.io.imsave(os.path.join(outdir, 'test_%s_%05d.png' % (prefix, i)), np.clip(img, 0.0, 1.0))
            print(i)
        np.save(os.path.join(outdir, 'test_%s_camera_pos.npy' % prefix), test_render_par)
        os.system('ffmpeg -i %s -r 30 -c:v libx264 -preset slow -crf 0 -r 30 %s'%(os.path.join(outdir, 'test_' + prefix + '_%05d.png'), os.path.join(outdir, 'test_video.mp4')))
    
if __name__ == '__main__':
    main()
        