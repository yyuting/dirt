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

canvas_width, canvas_height = 960, 640
centre_x, centre_y = 32, 64
square_size = 16

read_1st_round_from_file = True

def main():
    
    dir = '/n/fs/shaderml/deeplab-pytorch/result'
    highlight_dir = '/n/fs/shaderml/drone_videos/drone_frames/ocean3_00/highlight'
    orig_img_dir = '/n/fs/shaderml/drone_videos/drone_frames/ocean3_00'
    out_dir = 'horizon_optimize'

    files = os.listdir(orig_img_dir)
    files = sorted([os.path.join(orig_img_dir, file) for file in files if file.endswith('.png')])
    # for a initial test, use the first 200 consecutive frames
    files = files[:200]
    
    feed_dict_arr = np.zeros(8)
    feed_dict_arr[1] = 200.0
    feed_dict_arr[7] = 0.9
    img = np.zeros([640, 960, 3])
    
    nframes = len(files)
        
    session = tf.Session()
    
    ransac = linear_model.RANSACRegressor()
    line_X = np.arange(960)[:, np.newaxis]
    
    _, yv = numpy.meshgrid(numpy.arange(canvas_width), numpy.arange(canvas_height), indexing='ij')
    yv = np.transpose(yv)
    
    last_x = None
        
    with session.as_default():

        dirt_node, camera_pos = get_dirt_pixels()
        #x = np.load('res.npy')
        x = np.zeros((56, 4))
        for start_ind in range(0, 540, 20):
            res_filename = 'res_%d_%d.npy' % (start_ind, start_ind + 30)
            current_res = np.load(res_filename)
            if start_ind == 0:
                x[0, :] = current_res[:4]
            x[(start_ind//10)+1, :] = current_res[-12:-8]
            x[(start_ind//10)+2, :] = current_res[-8:-4]
        x[-1, :] = current_res[-4:]
        optimized_all_par = np.empty((551, 4))
            
        def h00(t):
            return 2 * t ** 3 - 3 * t ** 2 + 1

        def h10(t):
            return t ** 3 - 2 * t ** 2 + t

        def h01(t):
            return -2 * t ** 3 + 3 * t ** 2

        def h11(t):
            return t ** 3 - t ** 2
        interp_factor = 1 / 3
        
        for i in range(551):
            if i % 10 == 0:
                # if keyframe
                keyframe_idx = i // 10
                if keyframe_idx == 0:
                    m_before = np.zeros(4)
                    p_before = x[keyframe_idx, :]
                else:
                    m_before = m_after
                    p_before = p_after
                    
                if keyframe_idx >= 18:
                    m_after = np.zeros(4)
                else:
                    m_after = (x[keyframe_idx+2, :] - p_before) / 2.0 * 3.0
                
                if keyframe_idx == 55:
                    p_after = None
                else:
                    p_after = x[keyframe_idx+1, :]
                    
                current_par = p_before
            else:
                t = (i % 10) / 10
                current_par = h00(t) * p_before + h10(t) * interp_factor * m_before + h01(t) * p_after + h11(t) * interp_factor * m_after
            optimized_all_par[i, :] = current_par[:]
            continue
            feed_dict_arr[3] = current_par[0]
            feed_dict_arr[4] = current_par[1]
            feed_dict_arr[5] = current_par[2]
            feed_dict_arr[7] = current_par[3]

            current_seg = session.run(dirt_node, feed_dict={camera_pos: feed_dict_arr})
            
            orig_img_name = os.path.join(orig_img_dir, '%05d.png' % i)
            if not os.path.exists(orig_img_name):
                continue
            orig_img = skimage.transform.resize(skimage.io.imread(orig_img_name), (img.shape[0], img.shape[1]))
            
            is_sea_col = np.argmin(current_seg[:, :, 0], axis=0)
            
            fig = plt.figure()
            plt.imshow(orig_img)
            plt.plot(np.squeeze(line_X), is_sea_col)
            fig.savefig(os.path.join(out_dir, '%05d_sequence_optimized.png' % i))
            plt.close(fig)
            
            refl = skimage.transform.resize(skimage.io.imread(os.path.join(highlight_dir, '%05d.png' % i)), (img.shape[0], img.shape[1]))
            
            comp_refl = np.zeros(img.shape)
            comp_refl[:, :, 2] = current_seg[:, :, 1] * 4.0 - 3.0
            comp_refl[:, :, 1] = refl
            comp_refl = np.clip(comp_refl, 0.0, 1.0)
            skimage.io.imsave(os.path.join(out_dir, '%05d_sqeuence_optimized_refl.png' % i), comp_refl)
            
        np.save('optimized_all_par_slow.npy', optimized_all_par)
            
            
if __name__ == '__main__':
    main()