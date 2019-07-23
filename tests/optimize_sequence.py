
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

from optimize_horizon import ransac_fit_with_weights, get_dirt_pixels
linear_model.RANSACRegressor.ransac_fit_with_weights = ransac_fit_with_weights

canvas_width, canvas_height = 1024, 768
centre_x, centre_y = 32, 64
square_size = 16

read_1st_round_from_file = False

prefix = 'ocean4_clip_00'

def main():
    
    dir = '/n/fs/shaderml/deeplab-pytorch/result'
    highlight_dir = '/n/fs/shaderml/drone_videos/drone_frames/ocean3_00/highlight'
    orig_img_dir = '/n/fs/shaderml/drone_videos/drone_frames/ocean3_00'
    out_dir = 'optimize_debug/clip01'

    files = os.listdir(orig_img_dir)
    files = sorted([os.path.join(orig_img_dir, file) for file in files if file.endswith('.png')])
    
    feed_dict_arr = np.zeros(8)
    feed_dict_arr[1] = 200.0
    feed_dict_arr[7] = 0.9
    img = np.zeros([canvas_height, canvas_width, 3])
    
    nframes = len(files)
        
    session = tf.Session()
    
    ransac = linear_model.RANSACRegressor(stop_probability=0.995, max_trials=200)
    line_X = np.arange(canvas_width)[:, np.newaxis]
    
    _, yv = numpy.meshgrid(numpy.arange(canvas_width), numpy.arange(canvas_height), indexing='ij')
    yv = np.transpose(yv)
    
    last_x = None
        
    with session.as_default():

        dirt_node, camera_pos = get_dirt_pixels(canvas_width, canvas_height)
        # interpolate roughly every 1 sec
        # in order not to re-generate data, interpolate every 3 frames we generated(every 33 frames in video)
        #keyframes = np.arange(0, nframes, 3).tolist()
        
        # interpolate every 10 frames (roughly 0.33 sec)
        keyframes = np.arange(0, nframes, 10).tolist()
        
        if not read_1st_round_from_file:
            keyframe_params = np.empty([len(keyframes), 4])
            all_frames_seg = np.empty([nframes, 2])
            all_frames_par = np.empty([nframes, 4])
            for idx in range(nframes):
                print(files[idx])
                _, filename_short = os.path.split(files[idx])
                filename_only, _ = os.path.splitext(filename_short)

                orig_img_name = files[idx]
                orig_img = skimage.transform.resize(skimage.io.imread(orig_img_name), (img.shape[0], img.shape[1]))

                seg_name = os.path.join(dir, filename_short)
                seg = skimage.transform.resize(skimage.io.imread(seg_name), (img.shape[0], img.shape[1]))[:, :, 0]

                is_sea_col = np.argmin(seg, axis=0)
                ransac.fit(line_X, is_sea_col)
                line_y = ransac.predict(line_X)

                orig_gray = skimage.color.rgb2gray(orig_img)
                sobx = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0, ksize=3)
                soby = cv2.Sobel(orig_gray, cv2.CV_64F, 0, 1, ksize=3)
                sob_coef = sobx / soby
                sob_phi = np.arctan(sob_coef)
                sob_mag = (sobx ** 2.0 + soby ** 2.0)

                ransac_coef = ransac.estimator_.coef_
                ransac_phi = np.arctan(ransac_coef)
                seg_inliers_idx = np.nonzero(ransac.inlier_mask_)

                base_inlier = is_sea_col[ransac.inlier_mask_]
                cand_pts = base_inlier
                ransac_r2_cand = 10
                for i in range(1, ransac_r2_cand + 1):
                    cand_pts = np.concatenate((cand_pts, base_inlier - i, base_inlier + i))

                cand_pts = np.stack((cand_pts, np.tile(line_X[ransac.inlier_mask_, 0], 1 + 2 * ransac_r2_cand)), axis=0).astype('i')
                indices_1d = np.ravel_multi_index(cand_pts, seg.shape)
                cand_mask = np.zeros(seg.shape, dtype=bool)
                cand_mask[cand_pts[0], cand_pts[1]] = True

                similar_dir = np.abs(sob_phi - ransac_phi) <= (np.pi / 36)
                similar_dir *= cand_mask
                similar_dir *= (sob_mag >= 0.1)
                coord_x, coord_y = np.nonzero(similar_dir)

                # an alternative of ransac round 2
                pt_x = np.concatenate((line_X[seg_inliers_idx], np.expand_dims(coord_y, axis=1)))
                pt_y = np.concatenate((is_sea_col[seg_inliers_idx], coord_x))
                ransac_weights = np.ones(pt_y.shape)
                ransac_weights[len(seg_inliers_idx[0]):] = 10.0

                ransac.estimator_.fit(pt_x, pt_y, ransac_weights)
                line_y = ransac.estimator_.predict(line_X)
                is_sea_thre = np.tile(np.expand_dims(line_y, axis=0), [canvas_height, 1])

                seg = (yv <= line_y).astype('f')
                all_frames_seg[idx, 0] = ransac.estimator_.coef_
                all_frames_seg[idx, 1] = ransac.estimator_.intercept_

                if idx % 50 == 0:
                    fig = plt.figure()
                    plt.imshow(orig_img)
                    plt.plot(np.squeeze(line_X), line_y)
                    fig.savefig(os.path.join(out_dir, filename_only + '_sequence_ransac_img_comp.png'))
                    plt.close(fig)

                #if idx not in keyframes:
                #    continue

                refl = skimage.transform.resize(skimage.io.imread(os.path.join(highlight_dir, filename_short)), (img.shape[0], img.shape[1]))

                def opt_func(x):
                    # x is a 2D array that controls ang0 ang ang2
                    feed_dict_arr[3] = x[0]
                    feed_dict_arr[4] = x[1]
                    feed_dict_arr[5] = x[2]
                    feed_dict_arr[7] = x[3]
                    dirt_pixels = session.run(dirt_node, feed_dict={camera_pos: feed_dict_arr})
                    diff_seg = np.clip(dirt_pixels[:, :, 0], 0.0, 1.0) - seg
                    diff_refl = (np.maximum(dirt_pixels[:, :, 1] * 4.0 - 3.0, -1.0) - refl) * (4.5 + 3.5 * np.sign(refl))
                    loss1 = np.mean(diff_seg ** 2.0)
                    loss2 = np.mean(diff_refl ** 2.0) / 16
                    #downsampled_shader = skimage.measure.block_reduce(dirt_pixels[:, :, 1], (64, 64), np.mean)
                    #downsampled_ref = skimage.measure.block_reduce(refl, (64, 64), np.mean)
                    #loss2 = np.mean((downsampled_shader - downsampled_ref) ** 2.0)
                    #loss3 = (np.mean(dirt_pixels[:, :, 1]) - np.mean(refl)) ** 2.0 / (np.mean(refl) ** 2.0)
                    loss3 = 0.0
                    loss = loss1 + loss2 + loss3
                    #print('%.3f, %.3f, %.3f, %.3f' % (loss, loss1, loss2, loss3), x)
                    return loss

                if last_x is None:
                    x_init = np.zeros(4)
                    #x_init[1] = 0.3
                    x_init[3] = 1.9
                else:
                    x_init = last_x
                res = scipy.optimize.minimize(opt_func, x_init, method='Powell', options={'disp': True})
                last_x = res.x

                #keyframe_idx = keyframes.index(idx)
                #keyframe_params[keyframe_idx, :] = res.x[:]
                all_frames_par[idx, :] = res.x[:]

                feed_dict_arr[3] = res.x[0]
                feed_dict_arr[4] = res.x[1]
                feed_dict_arr[5] = res.x[2]
                feed_dict_arr[7] = res.x[3]
                current_seg = session.run(dirt_node, feed_dict={camera_pos: feed_dict_arr})

                if False:
                    comp_seg = np.zeros(current_seg.shape)
                    comp_seg[:, :, 2] = current_seg[:, :, 0]
                    comp_seg[:, :, 1] = seg
                    comp_seg = np.clip(comp_seg, 0.0, 1.0)
                    #comp_seg = 0.5 * current_seg[:, :, 0] + 0.5 * seg

                    skimage.io.imsave(os.path.join(out_dir, filename_only + '_seg.png'), comp_seg)


                    comp_img = np.clip(0.3 * np.expand_dims(current_seg[:, :, 0], 2) + 0.7 * orig_img, 0.0, 1.0)
                    skimage.io.imsave(os.path.join(out_dir, filename_short), comp_img)

                if idx % 50 == 0:
                    comp_refl = np.zeros(img.shape)
                    comp_refl[:, :, 2] = current_seg[:, :, 1] * 4.0 - 3.0
                    comp_refl[:, :, 1] = refl
                    #comp_refl = 0.5 * current_seg[:, :, 1] + 0.5 * refl
                    comp_refl = np.clip(comp_refl, 0.0, 1.0)
                    skimage.io.imsave(os.path.join(out_dir, filename_only + '_refl.png'), comp_refl)

            #np.save('sequence_keyframe_par.npy', keyframe_params)
            np.save('all_frames_seg.npy', all_frames_seg)
            np.save('all_frames_par.npy', all_frames_par)
            
        else:
            keyframe_params = np.load('sequence_keyframe_par.npy')
            all_frames_seg = np.load('sequence_horizon_line.npy')
            
            # initial test 31 frames
            # key frames 0, 10, 20, 30
            # for each key frame, 4 unkonwns (ang0, ang1, ang2, sun_Z)
            
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
            
            start_ind = 0
            end_ind = 30
            
            while end_ind < all_frames_seg.shape[0]:
                
                result_filename = 'res_%d_%d.npy' % (start_ind, end_ind)
                if os.path.exists(result_filename):
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
                        x_init[i*4:(i+1)*4] = keyframe_params[i, :]
                else:
                    if not x_init.shape[0] == 12:
                        x_init = np.zeros(12)
                    else:
                        x_init[:] = 0.0
                    x_init[:4] = last_result[-4:]
                    for i in range(1, 3):
                        x_init[i*4:(i+1)*4] = keyframe_params[i+(start_ind//10)+1, :]

                refls = [None] * 31
                for i in range(31):
                    refls[i] = skimage.transform.resize(skimage.io.imread(os.path.join(highlight_dir, '%05d.png' % (i + start_ind))), (img.shape[0], img.shape[1]))

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

                        coef = all_frames_seg[i+start_ind, 0]
                        intercept = all_frames_seg[i+start_ind, 1]
                        line_y = np.squeeze(coef * line_X + intercept)

                        seg = (yv <= line_y).astype('f')
                        #refl = skimage.transform.resize(skimage.io.imread(os.path.join(highlight_dir, '%05d.png' % i)), (img.shape[0], img.shape[1]))
                        refl = refls[i]

                        feed_dict_arr[3] = current_par[0]
                        feed_dict_arr[4] = current_par[1]
                        feed_dict_arr[5] = current_par[2]
                        feed_dict_arr[7] = current_par[3]

                        dirt_pixels = session.run(dirt_node, feed_dict={camera_pos: feed_dict_arr})
                        diff_seg = np.clip(dirt_pixels[:, :, 0], 0.0, 1.0) - seg
                        diff_refl = (np.maximum(dirt_pixels[:, :, 1] * 4.0 - 3.0, -1.0) - refl) * (4.5 + 3.5 * np.sign(refl))
                        loss1 = np.mean(diff_seg ** 2.0)
                        loss2 = np.mean(diff_refl ** 2.0) / 16
                        loss += loss1 + loss2
                    print(start_ind, loss, x)
                    return loss

                res = scipy.optimize.minimize(opt_func, x_init, method='Powell', options={'disp': True})
                print(res)
                np.save(result_filename, res.x)
                last_result = res.x
                start_ind += 20
                end_ind += 20


if __name__ == '__main__':
    main()

