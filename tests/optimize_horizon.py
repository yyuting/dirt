
import numpy as np
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

from sklearn.utils import check_random_state, check_array, check_consistent_length
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import has_fit_parameter
import sklearn.linear_model
_dynamic_max_trials = sklearn.linear_model.ransac._dynamic_max_trials

canvas_width, canvas_height = 960, 640
centre_x, centre_y = 32, 64
square_size = 16

def ransac_fit_with_weights(self, X, y, sample_weight=None, residual_threshold=None):
    """
    Modified sklearn.linear_model.RANSACRegressor.fit()
    sample_weight is used in sampling base points, fitting the regressor, and calculating score for candidate model
    """
    X = check_array(X, accept_sparse='csr')
    y = check_array(y, ensure_2d=False)
    check_consistent_length(X, y)

    if self.base_estimator is not None:
        base_estimator = clone(self.base_estimator)
    else:
        base_estimator = LinearRegression()

    if self.min_samples is None:
        # assume linear model by default
        min_samples = X.shape[1] + 1
    elif 0 < self.min_samples < 1:
        min_samples = np.ceil(self.min_samples * X.shape[0])
    elif self.min_samples >= 1:
        if self.min_samples % 1 != 0:
            raise ValueError("Absolute number of samples must be an "
                             "integer value.")
        min_samples = self.min_samples
    else:
        raise ValueError("Value for `min_samples` must be scalar and "
                         "positive.")
    if min_samples > X.shape[0]:
        raise ValueError("`min_samples` may not be larger than number "
                         "of samples: n_samples = %d." % (X.shape[0]))

    if self.stop_probability < 0 or self.stop_probability > 1:
        raise ValueError("`stop_probability` must be in range [0, 1].")

    if residual_threshold is None:
        if self.residual_threshold is None:
            # MAD (median absolute deviation)
            residual_threshold = np.median(np.abs(y - np.median(y)))
        else:
            residual_threshold = self.residual_threshold

    if self.loss == "absolute_loss":
        if y.ndim == 1:
            loss_function = lambda y_true, y_pred: np.abs(y_true - y_pred)
        else:
            loss_function = lambda \
                y_true, y_pred: np.sum(np.abs(y_true - y_pred), axis=1)

    elif self.loss == "squared_loss":
        if y.ndim == 1:
            loss_function = lambda y_true, y_pred: (y_true - y_pred) ** 2
        else:
            loss_function = lambda \
                y_true, y_pred: np.sum((y_true - y_pred) ** 2, axis=1)

    elif callable(self.loss):
        loss_function = self.loss

    else:
        raise ValueError(
            "loss should be 'absolute_loss', 'squared_loss' or a callable."
            "Got %s. " % self.loss)


    random_state = check_random_state(self.random_state)

    try:  # Not all estimator accept a random_state
        base_estimator.set_params(random_state=random_state)
    except ValueError:
        pass

    estimator_fit_has_sample_weight = has_fit_parameter(base_estimator,
                                                        "sample_weight")
    estimator_name = type(base_estimator).__name__
    if (sample_weight is not None and not
            estimator_fit_has_sample_weight):
        raise ValueError("%s does not support sample_weight. Samples"
                         " weights are only used for the calibration"
                         " itself." % estimator_name)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    n_inliers_best = 1
    score_best = -np.inf
    inlier_mask_best = None
    X_inlier_best = None
    y_inlier_best = None
    weight_inlier_best = None
    self.n_skips_no_inliers_ = 0
    self.n_skips_invalid_data_ = 0
    self.n_skips_invalid_model_ = 0

    # number of data samples
    n_samples = X.shape[0]
    sample_idxs = np.arange(n_samples)

    n_samples, _ = X.shape

    self.n_trials_ = 0
    max_trials = self.max_trials
    while self.n_trials_ < max_trials:
        self.n_trials_ += 1

        if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                self.n_skips_invalid_model_) > self.max_skips:
            break

        # choose random sample set
        #subset_idxs = sample_without_replacement(n_samples, min_samples,
        #                                         random_state=random_state)
        
        # use np.random.choice here since it allows sample with prob
        subset_idxs = np.random.choice(n_samples, min_samples, False, sample_weight / np.sum(sample_weight))
        X_subset = X[subset_idxs]
        y_subset = y[subset_idxs]

        # check if random sample set is valid
        if (self.is_data_valid is not None
                and not self.is_data_valid(X_subset, y_subset)):
            self.n_skips_invalid_data_ += 1
            continue

        # fit model for current random sample set
        if sample_weight is None:
            base_estimator.fit(X_subset, y_subset)
        else:
            base_estimator.fit(X_subset, y_subset,
                               sample_weight=sample_weight[subset_idxs])

        # check if estimated model is valid
        if (self.is_model_valid is not None and not
                self.is_model_valid(base_estimator, X_subset, y_subset)):
            self.n_skips_invalid_model_ += 1
            continue

        # residuals of all data for current random sample model
        y_pred = base_estimator.predict(X)
        residuals_subset = loss_function(y, y_pred)

        # classify data into inliers and outliers
        inlier_mask_subset = residuals_subset < residual_threshold
        n_inliers_subset = np.sum(inlier_mask_subset)

        # less inliers -> skip current random sample
        if n_inliers_subset < n_inliers_best:
            self.n_skips_no_inliers_ += 1
            continue

        # extract inlier data set
        inlier_idxs_subset = sample_idxs[inlier_mask_subset]
        X_inlier_subset = X[inlier_idxs_subset]
        y_inlier_subset = y[inlier_idxs_subset]
        if sample_weight is None:
            weight_inlier_subset = None
        else:
            weight_inlier_subset = sample_weight[inlier_idxs_subset]

        # score of inlier data set
        score_subset = base_estimator.score(X_inlier_subset,
                                            y_inlier_subset,
                                            sample_weight[inlier_idxs_subset])

        # same number of inliers but worse score -> skip current random
        # sample
        if (n_inliers_subset == n_inliers_best
                and score_subset < score_best):
            continue

        # save current random sample as best sample
        n_inliers_best = n_inliers_subset
        score_best = score_subset
        inlier_mask_best = inlier_mask_subset
        X_inlier_best = X_inlier_subset
        y_inlier_best = y_inlier_subset
        weight_inlier_best = weight_inlier_subset

        max_trials = min(
            max_trials,
            _dynamic_max_trials(n_inliers_best, n_samples,
                                min_samples, self.stop_probability))

        # break if sufficient number of inliers or score is reached
        if n_inliers_best >= self.stop_n_inliers or \
                        score_best >= self.stop_score:
            break

    # if none of the iterations met the required criteria
    if inlier_mask_best is None:
        if ((self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                self.n_skips_invalid_model_) > self.max_skips):
            raise ValueError(
                "RANSAC skipped more iterations than `max_skips` without"
                " finding a valid consensus set. Iterations were skipped"
                " because each randomly chosen sub-sample failed the"
                " passing criteria. See estimator attributes for"
                " diagnostics (n_skips*).")
        else:
            raise ValueError(
                "RANSAC could not find a valid consensus set. All"
                " `max_trials` iterations were skipped because each"
                " randomly chosen sub-sample failed the passing criteria."
                " See estimator attributes for diagnostics (n_skips*).")
    else:
        if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                self.n_skips_invalid_model_) > self.max_skips:
            warnings.warn("RANSAC found a valid consensus set but exited"
                          " early due to skipping more iterations than"
                          " `max_skips`. See estimator attributes for"
                          " diagnostics (n_skips*).",
                          ConvergenceWarning)

    # estimate final model using all inliers
    base_estimator.fit(X_inlier_best, y_inlier_best, weight_inlier_best)

    self.estimator_ = base_estimator
    self.inlier_mask_ = inlier_mask_best
    return self

linear_model.RANSACRegressor.ransac_fit_with_weights = ransac_fit_with_weights


def get_dirt_pixels(width=canvas_width, height=canvas_height):

    square_vertices = tf.constant([[-1, -1, 0, 1], [-1, 1, 0, 1], [1, 1, 0, 1], [1, -1, 0, 1]], dtype=tf.float32)

    #background = skimage.io.imread('/n/fs/shaderml/datas_oceanic/test_img/test_middle_ground00000.png')
    #background = tf.constant(skimage.img_as_float(background), dtype=tf.float32)
    background = tf.zeros([height, width, 3], dtype=tf.float32)
    
    camera_pos = tf.placeholder(tf.float32, 8)
    
    return dirt.rasterise(
        vertices=square_vertices,
        faces=[[0, 1, 2], [0, 2, 3]],
        vertex_colors=tf.ones([4, 3]),
        background=background,
        camera_pos = camera_pos,
        height=height, width=width, channels=3
    ), camera_pos


def main():
    
    dir = '/n/fs/shaderml/deeplab-pytorch/result'
    highlight_dir = '/n/fs/shaderml/drone_videos/drone_frames/ocean3_00/highlight'
    orig_img_dir = '/n/fs/shaderml/drone_videos/drone_frames/ocean3_00'
    out_dir = 'horizon_optimize'
    
    #files = os.listdir(dir)
    #files = sorted([os.path.join(dir, file) for file in files if 'coco_stuff' not in file])
    files = [os.path.join(dir, '%05d.png' % ind) for ind in range(0, 1860, 11)]
    
    #camera_pos_vals = np.load(os.path.join(dir, 'camera_pos_' + name + '.npy'))
    #render_t  = np.load(os.path.join(dir, 'render_t_' + name + '.npy'))
    #nframes = camera_pos_vals.shape[0]
    feed_dict_arr = np.zeros(8)
    feed_dict_arr[1] = 200.0
    feed_dict_arr[7] = 0.9
    img = np.zeros([640, 960, 3])
    
    nframes = len(files)
        
    session = tf.Session()
    
    ransac = linear_model.RANSACRegressor(stop_probability=0.995, max_trials=200)
    line_X = np.arange(960)[:, np.newaxis]
        
    with session.as_default():

        dirt_node, camera_pos = get_dirt_pixels()
        for idx in range(nframes):
            filename = files[idx]
            print(filename)
            _, filename_short = os.path.split(filename)
            filename_only, _ = os.path.splitext(filename_short)
            
            orig_img_name = os.path.join(orig_img_dir, filename_short)
            if not os.path.exists(orig_img_name):
                raise
            orig_img = skimage.transform.resize(skimage.io.imread(orig_img_name), (img.shape[0], img.shape[1]))

            seg = skimage.transform.resize(skimage.io.imread(filename), (img.shape[0], img.shape[1]))[:, :, 0]
            
            is_sea_col = np.argmin(seg, axis=0)
            ransac.fit(line_X, is_sea_col)
            line_y = ransac.predict(line_X)
            
            fig = plt.figure()
            plt.imshow(orig_img)
            plt.plot(np.squeeze(line_X), line_y)
            fig.savefig(os.path.join(out_dir, filename_only + '_ransac_img_comp.png'))
            plt.close(fig)
            
            fig = plt.figure()
            plt.imshow(seg)
            plt.plot(np.squeeze(line_X), line_y)
            fig.savefig(os.path.join(out_dir, filename_only + '_ransac_seg_comp.png'))
            plt.close(fig)
            
            orig_gray = skimage.color.rgb2gray(orig_img)
            sobx = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0, ksize=3)
            soby = cv2.Sobel(orig_gray, cv2.CV_64F, 0, 1, ksize=3)
            sob_coef = sobx / soby
            sob_phi = np.arctan(sob_coef)
            sob_mag = (sobx ** 2.0 + soby ** 2.0)
            
            ransac_coef = ransac.estimator_.coef_
            ransac_phi = np.arctan(ransac_coef)
            seg_inliers_idx = np.nonzero(ransac.inlier_mask_)
            
            #line_bot = np.floor(line_y)
            #line_up = np.ceil(line_y)
            #cand_pts = np.concatenate((line_bot, line_up))
            #ransec_r2_cand = 10
            #for i in range(1, ransec_r2_cand + 1):
            #    cand_pts = np.concatenate((cand_pts, line_bot - i, line_up + i))
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
            
            ransac_weights = np.ones(seg.shape[1] + coord_x.shape[0])
            # higher weights to pts with high gradient activation that is in similar direction to the line detected in 1st round
            ransac_weights[seg.shape[1]:] = 10.0
            ransac.ransac_fit_with_weights(np.concatenate((line_X, np.expand_dims(coord_y, axis=1))), np.concatenate((is_sea_col, coord_x)), ransac_weights)
            line_y = ransac.predict(line_X)
                        
            fig = plt.figure()
            plt.imshow(orig_img)
            plt.plot(np.squeeze(np.concatenate((line_X, np.expand_dims(coord_y, axis=1)))), np.concatenate((is_sea_col, coord_x)), '.', markersize=2)
            fig.savefig(os.path.join(out_dir, filename_only + '_ransac_debug.png'))
                        
            fig = plt.figure()
            plt.imshow(orig_img)
            plt.plot(np.squeeze(line_X), line_y)
            fig.savefig(os.path.join(out_dir, filename_only + '_ransac_round2_img_comp.png'))
            plt.close(fig)
            
            fig = plt.figure()
            plt.imshow(seg)
            plt.plot(np.squeeze(line_X), line_y)
            fig.savefig(os.path.join(out_dir, filename_only + '_ransac_round2_seg_comp.png'))
            plt.close(fig)
            
            # an alternative of ransac round 2
            pt_x = np.concatenate((line_X[seg_inliers_idx], np.expand_dims(coord_y, axis=1)))
            pt_y = np.concatenate((is_sea_col[seg_inliers_idx], coord_x))
            ransac_weights = np.ones(pt_y.shape)
            ransac_weights[len(seg_inliers_idx[0]):] = 10.0
            #ransac.ransac_fit_with_weights(pt_x, pt_y, ransac_weights, residual_threshold=np.median(np.abs(is_sea_col - np.median(is_sea_col))))
            
            #new_pt_x = pt_x[ransac.inlier_mask_]
            #new_pt_y = pt_y[ransac.inlier_mask_]
            #new_weights = ransac_weights[ransac.inlier_mask_]
            #ransac.estimator_.fit(new_pt_x, new_pt_y, new_weights)
            ransac.estimator_.fit(pt_x, pt_y, ransac_weights)
            line_y = ransac.estimator_.predict(line_X)
            
            fig = plt.figure()
            plt.imshow(orig_img)
            plt.plot(np.squeeze(line_X), line_y)
            fig.savefig(os.path.join(out_dir, filename_only + '_ransac_round3_img_comp.png'))
            plt.close(fig)

            continue
            
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
            
            x_init = np.zeros(4)
            #x_init[1] = 0.3
            x_init[3] = 1.9
            res = scipy.optimize.minimize(opt_func, x_init, method='Powell', options={'disp': True})
            print(res)
            #break
            
            feed_dict_arr[3] = res.x[0]
            feed_dict_arr[4] = res.x[1]
            feed_dict_arr[5] = res.x[2]
            feed_dict_arr[7] = res.x[3]
            current_seg = session.run(dirt_node, feed_dict={camera_pos: feed_dict_arr})
            
            comp_seg = np.zeros(current_seg.shape)
            comp_seg[:, :, 2] = current_seg[:, :, 0]
            comp_seg[:, :, 1] = seg
            comp_seg = np.clip(comp_seg, 0.0, 1.0)
            #comp_seg = 0.5 * current_seg[:, :, 0] + 0.5 * seg
            
            skimage.io.imsave(os.path.join(out_dir, filename_only + '_seg.png'), comp_seg)
            
            
            comp_img = np.clip(0.3 * np.expand_dims(current_seg[:, :, 0], 2) + 0.7 * orig_img, 0.0, 1.0)
            skimage.io.imsave(os.path.join(out_dir, filename_short), comp_img)
            
            comp_refl = np.zeros(comp_seg.shape)
            comp_refl[:, :, 2] = current_seg[:, :, 1] * 4.0 - 3.0
            comp_refl[:, :, 1] = refl
            #comp_refl = 0.5 * current_seg[:, :, 1] + 0.5 * refl
            comp_refl = np.clip(comp_refl, 0.0, 1.0)
            skimage.io.imsave(os.path.join(out_dir, filename_only + '_refl.png'), comp_refl)
            
            

if __name__ == '__main__':
    main()

