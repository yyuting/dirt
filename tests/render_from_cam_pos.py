import os
import numpy as np
import numpy
import sys

import tensorflow as tf
import dirt
import skimage.io

from optimize_sequence import cheat_roll, dims


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
        if sys.argv[5].startswith('hill'):
            background_np = np.load(os.path.join(sys.argv[3], 'terrain_lookup.npy'))
            background = tf.constant(background_np, dtype=tf.float32)
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
    assert len(sys.argv) > 7
    
    cam_pos_file = sys.argv[1]
    sample_freq = int(sys.argv[2])
    outdir = sys.argv[3]
    prefix = sys.argv[4]
    par_mode = sys.argv[6]
    is_rand = bool(int(sys.argv[7]))
    
    if len(sys.argv) > 8:
        novel_mode = sys.argv[8]
    else:
        novel_mode = 'easy_still'
        
    par_vals = np.load(cam_pos_file)
    
    
    
       
    is_opt_flow = (sys.argv[5] == 'oceanic_opt_flow')
    
    global canvas_width, canvas_height
    if par_mode.startswith('ocean'):
        canvas_width = 1024
        canvas_height = 768
        par_dims = 9
        nsamples = 100
    elif par_mode.startswith('hill'):
        canvas_width = 960
        canvas_height = 540
        par_dims = 12
        nsamples = 1
    else:
        raise
        
    if is_opt_flow:
        par_dims = 15
        nsamples = 1
        
    
    camera_pos_val = np.zeros(par_dims)
    render_par_vals = np.empty([par_vals.shape[0], par_dims])
    
    ans = get_dirt_pixels_render(is_opt_flow, par_dims)
    render_node = ans[0]
    camera_pos = ans[1]
    bg = ans[2]
    
    sess = tf.Session()
    
    img = np.empty([canvas_height, canvas_width, 3])
    
    # for clip ocean4_00, roughly the height changes from 100 - 200 in the first 270 frames
        
    feed_dict = {}
        
    if True:
        
        if is_opt_flow:
            ini_round_freq = 1
        else:
            ini_round_freq = sample_freq
        for i in range(0, render_par_vals.shape[0], ini_round_freq):
            if par_mode.startswith('ocean'):
                if par_mode == 'ocean4_00':
                    camera_pos_val[[0, 2]] = 0.0
                    camera_pos_val[0] = -i * 0.1
                    camera_pos_val[6] = i / 30
                    if i <= 270:
                        camera_pos_val[1] = 60 * i / 270 + 100
                    else:
                        camera_pos_val[1] = 160
                elif par_mode == 'ocean4_01':
                    camera_pos_val[[0, 2]] = 0.0
                    if i <= 210:
                        z_speed = 2 + (i / 100) * 0.35
                    else:
                        z_speed = 2 + (210 / 100) * 0.35
                    #z_speed = 2
                    camera_pos_val[2] = i * z_speed
                    camera_pos_val[6] = i / 30
                    camera_pos_val[1] = 100
                else:
                    raise

                if is_rand:
                    camera_pos_val[[0, 2]] = (np.random.random(2) - 0.5) * 500.0
                    camera_pos_val[6] = np.random.random() * 180.0
                    camera_pos_val[8] = np.random.random() * 180.0

                camera_pos_val[3:5] = par_vals[i, :2]
                if not cheat_roll:
                    camera_pos_val[5] = par_vals[i, 2]
                camera_pos_val[7] = par_vals[i, -1]
            elif par_mode.startswith('hill'):
                camera_pos_val[:] = par_vals[i].reshape(par_dims)
            else:
                raise

            render_par_vals[i, :] = camera_pos_val[:]
        
            
        opt_flow_spacing = 10
        if is_opt_flow:
            restart = opt_flow_spacing
        else:
            restart = 0
            
        orig_img_shape = None
        
        for i in range(restart, render_par_vals.shape[0], sample_freq):
            camera_pos_val[:] = render_par_vals[i, :]
            #camera_pos_val[:] = render_par_vals[0, :]
            #camera_pos_val[-2] += 0.01 * i
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
        if sample_freq == 1 and (not is_rand):
            os.system('ffmpeg -i %s -r 30 -c:v libx264 -preset slow -crf 0 -r 30 %s'%(os.path.join(outdir, prefix + '_%05d.png'), os.path.join(outdir, 'video.mp4')))
   
    if False:

        # generate test frames along a given path
        nframes = 200
        test_render_par = np.empty([nframes, par_dims])
        for i in range(nframes):
            camera_pos_val[0] = i
            camera_pos_val[1] = 100 * i / nframes + 100
            camera_pos_val[2] = 0.5 * i
            camera_pos_val[3:5] = np.min(par_vals[:, :2], axis=0) + i / nframes * (np.max(par_vals[:, :2], axis=0) - np.min(par_vals[:, :2], axis=0))
            if not cheat_roll:
                camera_pos_val[5] = np.min(par_vals[:, 2], axis=0) + i / nframes * (np.max(par_vals[:, 2], axis=0) - np.min(par_vals[:, 2], axis=0))
            camera_pos_val[6] = i / 30
            camera_pos_val[7] = np.min(par_vals[:, -1], axis=0) + i / nframes * (np.max(par_vals[:, -1], axis=0) - np.min(par_vals[:, -1], axis=0))
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
        
    if False:
        nvideos = 1
        frames_per_video = 100
        # simpliest case: start from frame 0
        # easy case: other values are fixed, only x, z is moving
        test_render_par = np.empty([nvideos, frames_per_video, par_dims])
        for ind_v in range(nvideos):
                        
            camera_pos_val[[0, 2]] = (np.random.random(2) - 0.5) * 500.0
            
            camera_pos_val[3:5] = par_vals[0, :2]
            if not cheat_roll:
                camera_pos_val[5] = par_vals[0, 2]
            camera_pos_val[7] = par_vals[0, -1]
                        
            if par_mode == 'ocean4_00':
                camera_pos_val[1] = 100
                x_speed_ini = -0.1
                x_speed = -np.random.rand() * 0.1
                z_speed_ini = 0.0
                z_speed = 0.0
                y_speed_ini = 60 / 270
            elif par_mode == 'ocean4_01':
                camera_pos_val[1] = 100
                x_speed_ini = 0.0
                x_speed = 0.0
                z_speed_ini = 2
                z_speed = np.random.rand() * 0.8 + 2
                y_speed_ini = 0.0
                
            ang_ini_speed = np.zeros(3)
            ang_ini_speed[:2] = par_vals[1, :2] - par_vals[0, :2]
            if not cheat_roll:
                ang_ini_speed[2] = par_vals[1, 2] - par_vals[0, 2]
            sun_ini_speed = par_vals[1, -1] - par_vals[0, -1]
                
            if novel_mode == 'easy_still':
                ang_speed = np.zeros(3)
                sun_speed = 0.0
            elif novel_mode == 'easy_linear':
                start_par = par_vals[1, :]
                max_par = np.max(par_vals, axis=0)
                min_par = np.min(par_vals, axis=0)
                speed_sign = np.sign((max_par - start_par) - (start_par - min_par))
                max_speed = speed_sign * np.maximum(max_par - start_par, start_par - min_par) / frames_per_video
                gt_speed = np.diff(par_vals, axis=0)
                max_speed = np.clip(max_speed, np.min(gt_speed, axis=0), np.max(gt_speed, axis=0))
                if cheat_roll:
                    ang_speed = np.array([max_speed[0], max_speed[1], 0.0])
                else:
                    ang_speed = max_speed[:3]
                sun_speed = max_speed[-1]
            else:
                raise
                                           
            for ind_f in range(frames_per_video):
                
                camera_pos_val[6] = ind_f / 30
                
                if ind_f > 1:
                    camera_pos_val[0] += x_speed
                    camera_pos_val[2] += z_speed
                    camera_pos_val[3:6] += ang_speed
                    camera_pos_val[7] += sun_speed
                elif ind_f > 0:
                    camera_pos_val[0] += x_speed_ini
                    camera_pos_val[1] += y_speed_ini
                    camera_pos_val[2] += z_speed_ini
                    camera_pos_val[3:6] += ang_ini_speed
                    camera_pos_val[7] += sun_ini_speed
                test_render_par[ind_v, ind_f, :] = camera_pos_val[:]
                img[:] = 0.0
                for _ in range(nsamples):
                    dirt_pixels = sess.run(render_node, feed_dict={camera_pos: camera_pos_val})
                    img += dirt_pixels
                img /= nsamples
                skimage.io.imsave(os.path.join(outdir, 'test_%s_%05d_%05d.png' % (prefix, ind_v, ind_f)), np.clip(img, 0.0, 1.0))
                print(ind_f)
            os.system('ffmpeg -i %s -r 30 -c:v libx264 -preset slow -crf 0 -r 30 %s'%(os.path.join(outdir, 'test_%s_%05d' % (prefix, ind_v) + '_%05d.png'), os.path.join(outdir, 'test_%s_%05d_video.mp4' % (prefix, ind_v))))
        
        np.save(os.path.join(outdir, 'test_%s_camera_pos.npy' % prefix), test_render_par)
        
    
if __name__ == '__main__':
    main()
        