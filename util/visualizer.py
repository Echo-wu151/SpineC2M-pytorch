
import os
import ntpath
import time
from . import util
import scipy.misc

from io import BytesIO         # Python 3.x


def difference_map(pred, gt, save_fold, img_name, method_name, min_value=0, max_value=500):
    save_fold = os.path.join(save_fold, method_name)
    if not os.path.isdir(save_fold):
        os.makedirs(save_fold)

    raw_img_fold = os.path.join(save_fold, 'raw')
    if not os.path.isdir(raw_img_fold):
        os.makedirs(raw_img_fold)

    pred = (pred[:, :, 1]).astype(np.float32)
    gt = (gt[:, :, 1]).astype(np.float32)

    diff_map = np.abs(pred - gt)
    diff_map[0, 0] = min_value
    diff_map[-1, -1] = max_value

    plt.imshow(diff_map, vmin=min_value, vmax=max_value, cmap='afmhot')
    cb = plt.colorbar(ticks=np.linspace(min_value, max_value, num=3))
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=12)
    plt.axis('off')
    # plt.show()

    plt.savefig(os.path.join(save_fold, img_name), bbox_inches='tight')
    plt.close()

    # save raw pred_img
    cv2.imwrite(os.path.join(raw_img_fold, img_name), pred)
    
    
def draw_box_plot(data_list, method_names):
    filenames = ['MAE', 'RMSE', 'PSNR', 'SSIM', 'PCC']
    expressions = [' (lower is better)', ' (lower is better)', ' (higher is better)', ' (higher is better)',
                   '(higher is better)']
    colors = ['red', 'green', 'blue', 'aquamarine', 'aqua']  # purple

    for idx, data in enumerate(data_list):
        fig1, ax1 = plt.subplots(figsize=(2.5*len(method_names), 6))
        box = ax1.boxplot(np.transpose(data), patch_artist=True, showmeans=True, sym='r+', vert=True)

        # connect mean values
        y = data.mean(axis=1)
        ax1.plot(range(1, len(method_names)+1), y, 'r--')

        for patch, color in zip(box['boxes'], colors):
            patch.set(facecolor=color, alpha=0.5, linewidth=1)

        # scatter draw datapoints
        x_vals, y_vals = [], []
        for i in range(data_list[0].shape[0]):
            # move x coordinate to not overlapping
            x_vals.append(np.random.normal(i + 0.7, 0.04, data.shape[1]))
            y_vals.append(data[i, :].tolist())

        for x_val, y_val, color in zip(x_vals, y_vals, colors):
            ax1.scatter(x_val, y_val, s=5, c=color, alpha=0.5)

        ax1.yaxis.grid()  # horizontal lines
        ax1.set_xticklabels([method_name for method_name in method_names], fontsize=14)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        plt.setp(box['medians'], color='black')
        plt.title(filenames[idx] + expressions[idx], fontsize=14)
        plt.savefig(filenames[idx] + '.jpg', dpi=300)
        plt.close()

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        if opt.isTrain:
            self.log_name = os.path.join(
                opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write(
                    '================ Training Loss (%s) ================\n' %
                    now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        # convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)

        if self.tf_log:  # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                s = BytesIO()
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]
                scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                # Create an Image object
                img_sum = self.tf.Summary.Image(
                    encoded_image_string=s.getvalue(),
                    height=image_numpy.shape[0],
                    width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(
                    self.tf.Summary.Value(
                        tag=label, image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)


    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = value.mean().float()
                summary = self.tf.Summary(
                    value=[
                        self.tf.Summary.Value(
                            tag=tag,
                            simple_value=value)])
                self.writer.add_summary(summary, step)
                
                
    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            # print(v)
            # if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if 'input_label' == key:
                if self.opt.no_label:
                    t = util.tensor2im(t, tile=tile)
                else:
                    t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals
    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if 'input_label' == key:
                if self.opt.no_label:
                    t = util.tensor2im(t, tile=tile)
                else:
                    t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, visuals, image_path):
        visuals = self.convert_visuals_to_numpy(visuals)

        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

    # errors: dictionary of error labels and values
    def get_matrics(self, generated, img_t, step):

        return
