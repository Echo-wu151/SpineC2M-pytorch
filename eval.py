import os
import sys
import cv2
import numpy as np

import util.metrics as utils


def main(gt, methods, display_names):
    # read gt image addresses
    gt_names = utils.all_files_under(os.path.join('../', gt), extension='.jpg')

    # read prediction image addresses
    filenames_list = []
    for method in methods:
        filenames = utils.all_files_under(os.path.join('../', method), extension='.jpg')
        filenames_list.append(filenames)

    mae_overall, rmse_overall, psnr_overall, ssim_overall, pcc_overall = [], [], [], [], []
    for idx_method in range(len(methods)):
        # print(methods[idx_method])
        mae_method, rmse_method, psnr_method, ssim_method, pcc_method = [], [], [], [], []

        for idx_name in range(len(gt_names)):
            # read gt and prediction image
            gt_img = cv2.imread(gt_names[idx_name], cv2.IMREAD_GRAYSCALE).astype(np.float32)
            pred_img = cv2.imread(filenames_list[idx_method][idx_name], cv2.IMREAD_GRAYSCALE).astype(np.float32)

            # check gt and prediction image name
            if gt_names[idx_name].split('p0')[-1] != filenames_list[idx_method][idx_name].split('p0')[-1]:
                sys.exit(' [!] Image name can not match!')

            # calcualte mae and psnr
            mae = utils.mean_absoulute_error(pred_img, gt_img)
            rmse = utils.root_mean_square_error(pred_img, gt_img)
            psnr = utils.peak_signal_to_noise_ratio(pred_img, gt_img)
            ssim = utils.structural_similarity_index(pred_img, gt_img)
            pcc = utils.pearson_correlation_coefficient(pred_img, gt_img)

            if np.mod(idx_name, 300) == 0:
                print('Method: {}, idx: {}'.format(methods[idx_method], idx_name))

            # collect each image results
            mae_method.append(mae)
            rmse_method.append(rmse)
            psnr_method.append(psnr)
            ssim_method.append(ssim)
            pcc_method.append(pcc)

        # list to np.array
        mae_method = np.asarray(mae_method)
        rmse_method = np.asarray(rmse_method)
        psnr_method = np.asarray(psnr_method)
        ssim_method = np.asarray(ssim_method)
        pcc_method = np.asarray(pcc_method)

        # collect all methods results
        mae_overall.append(mae_method)
        rmse_overall.append(rmse_method)
        psnr_overall.append(psnr_method)
        ssim_overall.append(ssim_method)
        pcc_overall.append(pcc_method)

    # list to np.array
    mae_overall = np.asarray(mae_overall)
    rmse_overall = np.asarray(rmse_overall)
    psnr_overall = np.asarray(psnr_overall)
    ssim_overall = np.asarray(ssim_overall)
    pcc_overall = np.asarray(pcc_overall)

    # draw boxplot
    utils.draw_box_plot([mae_overall, rmse_overall, psnr_overall, ssim_overall, pcc_overall], display_names)
    # write to csv file
    utils.write_to_csv([mae_overall, rmse_overall, psnr_overall, ssim_overall, pcc_overall], methods, gt_names)


if __name__ == '__main__':
    gt_ = 'gt'
    methods_ = ['pix2pix', 'cyclegan', 'discogan', 'mrgan', 'DC2Anet']
    display_names_ = ['Multi-Channel GAN', 'Deep MR-to-CT', 'DiscoGAN', 'MR-GAN', 'DC2Anet']

    main(gt_, methods_, display_names_)