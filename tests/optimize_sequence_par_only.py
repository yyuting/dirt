import numpy as np
import numpy
import tensorflow as tf
import dirt
import skimage.io
import skimage
import skimage.transform
import skimage.color
import time
import os
import scipy
import scipy.optimize
import skimage.measure
from sklearn import linear_model, datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import sys

from optimize_horizon import ransac_fit_with_weights, get_dirt_pixels
linear_model.RANSACRegressor.ransac_fit_with_weights = ransac_fit_with_weights

canvas_width, canvas_height = 960, 640
centre_x, centre_y = 32, 64
square_size = 16

read_1st_round_from_file = False

def main():
    
    dir = '/n/fs/shaderml/deeplab-pytorch/result'
    highlight_dir = '/n/fs/shaderml/drone_videos/drone_frames/ocean3_00/highlight'
    orig_img_dir = '/n/fs/shaderml/drone_videos/drone_frames/ocean3_00'
    out_dir = 'optimize_debug/clip01'

    files = os.listdir(orig_img_dir)
    files = sorted([os.path.join(orig_img_dir, file) for file in files if file.endswith('.png')])
    # for a initial test, use the first 200 consecutive frames
    #files = files[:200]
    
    feed_dict_arr = np.zeros(8)
    feed_dict_arr[1] = 200.0
    feed_dict_arr[7] = 0.9
    img = np.zeros([canvas_height, canvas_width, 3])
    
    nframes = len(files)
    
    all_frames_seg = np.load('all_frames_seg.npy')
    arr = np.load('all_frames_par.npy')
    # normalize par
    min_val = np.min(arr, axis=0)
    max_val = np.max(arr, axis=0)
    all_frames_par = (arr - min_val) / (max_val - min_val)
    
    dim = 16
    x_init = np.zeros(dim)
    zero_tangent = np.zeros(4)

    def h00(t):
        return 2 * t ** 3 - 3 * t ** 2 + 1

    def h10(t):
        return t ** 3 - 2 * t ** 2 + t

    def h01(t):
        return -2 * t ** 3 + 3 * t ** 2

    def h11(t):
        return t ** 3 - t ** 2

    interp_factor = 1 / 3
    
    keyframe_freq = 30
    
    if sys.argv[1] == 'whole_opt':
        num_keyframes = all_frames_par.shape[0] // keyframe_freq + 1
        x_init = np.zeros(4*num_keyframes)
        for i in range(num_keyframes):
            x_init[4*i:4*(i+1)] = all_frames_par[i*keyframe_freq, :]

        def opt_func(x):
            loss = 0.0
            for i in range((num_keyframes-1)*keyframe_freq+1):
                if i % keyframe_freq == 0:
                    # if keyframe
                    keyframe_idx = i // keyframe_freq
                    # UNSAFE assumption: beginning and ending tangent are 0
                    if keyframe_idx == 0:
                        m_before = zero_tangent
                        p_before = x[:4]
                    else:
                        m_before = m_after
                        p_before = p_after
                    if keyframe_idx >= num_keyframes - 2:
                        m_after = zero_tangent
                    else:
                        m_after = m_after = (x[(keyframe_idx+2)*4:(keyframe_idx+3)*4] - p_before) / 2.0 * 3.0
                    if keyframe_idx >= num_keyframes - 1:
                        p_after = p_before
                    else:
                        p_after = x[(keyframe_idx+1)*4:(keyframe_idx+2)*4]
                        
                    current_par = p_before
                else:
                    t = (i % keyframe_freq) / keyframe_freq
                    current_par = h00(t) * p_before + h10(t) * interp_factor * m_before + h01(t) * p_after + h11(t) * interp_factor * m_after

                comp_par = all_frames_par[i, :]
                loss += np.mean((comp_par - current_par) ** 2.0)
            print(loss)
            return loss

        res = scipy.optimize.minimize(opt_func, x_init, method='Powell', options={'disp': True})
        print(res)

        last_result = res.x
        np.save('par_only_whole_%d.npy' % keyframe_freq, last_result)
    elif sys.argv[1] == 'opt':
        start_ind = 0
        end_ind = 30

        while end_ind < all_frames_seg.shape[0]:

            result_filename = 'par_only_%d_%d.npy' % (start_ind, end_ind)
            #if os.path.exists(result_filename):
            if False:
                last_result = np.load(result_filename)
                start_ind += 20
                end_ind += 20
                continue

            if start_ind == 0:
                if not x_init.shape[0] == 16:
                    x_init = np.zeros(16)
                else:
                    x_init[:] = 0.0
                for i in range(4):
                    x_init[i*4:(i+1)*4] = all_frames_par[i*10, :]
            else:
                if not x_init.shape[0] == 12:
                    x_init = np.zeros(12)
                else:
                    x_init[:] = 0.0
                x_init[:4] = last_result[-4:]
                for i in range(1, 3):
                    x_init[i*4:(i+1)*4] = all_frames_par[(i+(start_ind//10)+1)*10, :]

            def opt_func(x):
                loss = 0.0
                for i in range(31):
                    # compute rendering for all 31 frames
                    if i % 10 == 0:
                        # if keyframe
                        keyframe_idx = i // 10
                        # UNSAFE assumption: beginning and ending tangent are 0 for the 31 frame period
                        if start_ind == 0:
                            if keyframe_idx == 0:
                                m_before = zero_tangent
                                p_before = x[:4]
                                m_after = (x[8:12] - p_before) / 2.0 * 3.0
                                p_after = x[4:8]
                            else:
                                m_before = m_after
                                p_before = p_after
                                if keyframe_idx == 1:
                                    m_after = (x[12:] - p_before) / 2.0 * 3.0
                                else:
                                    m_after = zero_tangent
                                p_after = x[(keyframe_idx+1)*4:(keyframe_idx+2)*4]
                        else:
                            if keyframe_idx == 0:
                                m_before = (x[:4] - last_result[-12:-8]) / 2.0 * 3.0
                                p_before = last_result[-8:-4]
                                m_after = (x[4:8] - p_before) / 2.0 * 3.0
                                p_after = x[:4]
                            else:
                                m_before = m_after
                                p_before = p_after
                                if keyframe_idx == 1:
                                    m_after = (x[-4:] - p_before) / 2.0 * 3.0
                                else:
                                    m_after = zero_tangent
                                p_after = x[(keyframe_idx)*4:(keyframe_idx+1)*4]
                        current_par = p_before
                    else:
                        t = (i % 10) / 10
                        current_par = h00(t) * p_before + h10(t) * interp_factor * m_before + h01(t) * p_after + h11(t) * interp_factor * m_after

                    comp_par = all_frames_par[i+start_ind, :]
                    loss += np.mean((comp_par - current_par) ** 2.0)
                print(start_ind, loss, x)
                return loss

            res = scipy.optimize.minimize(opt_func, x_init, method='Powell', options={'disp': True})
            print(res)

            last_result = res.x
            np.save(result_filename, last_result)
            start_ind += 20
            end_ind += 20
            
    elif sys.argv[1] == 'render':
        if len(sys.argv) > 2 and sys.argv[2] == 'whole':
            nframes = 561
            prefix = '_whole'
            x = np.load('par_only_whole_%d.npy' % keyframe_freq)
            x = x.reshape((x.shape[0]//4, 4))
            nframes = (x.shape[0] - 1) * keyframe_freq + 1
        else:
            nframes = 551
            prefix = '_30_seq'
            x = np.zeros((56, 4))
            for start_ind in range(0, 540, 20):
                res_filename = 'par_only_%d_%d.npy' % (start_ind, start_ind + 30)
                current_res = np.load(res_filename)
                if start_ind == 0:
                    x[0, :] = current_res[:4]
                x[(start_ind//10)+1, :] = current_res[-12:-8]
                x[(start_ind//10)+2, :] = current_res[-8:-4]
            x[-1, :] = current_res[-4:]

        optimized_all_par = np.empty((nframes, 4))

        for i in range(nframes):
            if i % keyframe_freq == 0:
                # if keyframe
                keyframe_idx = i // keyframe_freq
                if keyframe_idx == 0:
                    m_before = np.zeros(4)
                    p_before = x[keyframe_idx, :]
                else:
                    m_before = m_after
                    p_before = p_after

                if keyframe_idx >= x.shape[0] - 2:
                    m_after = np.zeros(4)
                else:
                    m_after = (x[keyframe_idx+2, :] - p_before) / 2.0 * 3.0

                if keyframe_idx == x.shape[0] - 1:
                    p_after = None
                else:
                    p_after = x[keyframe_idx+1, :]

                current_par = p_before
            else:
                t = (i % keyframe_freq) / keyframe_freq
                current_par = h00(t) * p_before + h10(t) * interp_factor * m_before + h01(t) * p_after + h11(t) * interp_factor * m_after

            optimized_all_par[i, :] = current_par[:]

        optimized_all_par *= (max_val - min_val)
        optimized_all_par += min_val
        np.save('optimized_all_par%s.npy' % prefix, optimized_all_par)
        #return
        session = tf.Session()
        with session.as_default():

            dirt_node, camera_pos = get_dirt_pixels()
            line_X = np.arange(canvas_width)[:, np.newaxis]

            for i in range(0, nframes, 50):
                feed_dict_arr[3] = optimized_all_par[i, 0]
                feed_dict_arr[4] = optimized_all_par[i, 1]
                feed_dict_arr[5] = optimized_all_par[i, 2]
                feed_dict_arr[7] = optimized_all_par[i, 3]
                current_seg = session.run(dirt_node, feed_dict={camera_pos: feed_dict_arr})
                
                orig_img_name = os.path.join(orig_img_dir, '%05d.png' % i)
                orig_img = skimage.transform.resize(skimage.io.imread(orig_img_name), (img.shape[0], img.shape[1]))

                is_sea_col = np.argmin(current_seg[:, :, 0], axis=0)

                fig = plt.figure()
                plt.imshow(orig_img)
                plt.plot(np.squeeze(line_X), is_sea_col)
                fig.savefig(os.path.join(out_dir, '%05d_opt_par_only%s.png' % (i, prefix)))
                plt.close(fig)

                refl = skimage.transform.resize(skimage.io.imread(os.path.join(highlight_dir, '%05d.png' % i)), (img.shape[0], img.shape[1]))

                comp_refl = np.zeros(img.shape)
                comp_refl[:, :, 2] = current_seg[:, :, 1] * 4.0 - 3.0
                comp_refl[:, :, 1] = refl
                comp_refl = np.clip(comp_refl, 0.0, 1.0)
                skimage.io.imsave(os.path.join(out_dir, '%05d_opt_par_only%s_refl.png' % (i, prefix)), comp_refl)
        
if __name__ == '__main__':
    main()