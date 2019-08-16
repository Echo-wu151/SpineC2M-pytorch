
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument(
            '--display_freq',
            type=int,
            default=10000,
            help='frequency of showing training results on screen')
        parser.add_argument(
            '--print_freq',
            type=int,
            default=100,
            help='frequency of showing training results on console')
        parser.add_argument(
            '--save_latest_freq',
            type=int,
            default=10000,
            help='frequency of saving the latest results')
        parser.add_argument(
            '--save_epoch_freq',
            type=int,
            default=200,
            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument(
            '--no_html',
            action='store_true',
            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument(
            '--tf_log',
            action='store_true',
            help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for training
        parser.add_argument(
            '--continue_train',
            action='store_true',
            help='continue training: load the latest model')
        parser.add_argument(
            '--which_epoch',
            type=str,
            default='latest',
            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument(
            '--niter',
            type=int,
            default=500,
            help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument(
            '--niter_decay',
            type=int,
            default=500,
            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument(
            '--beta1',
            type=float,
            default=0.5,
            help='momentum term of adam')
        parser.add_argument(
            '--beta2',
            type=float,
            default=0.9,
            help='momentum term of adam')
        parser.add_argument(
            '--lr',
            type=float,
            default=0.0002,
            help='initial learning rate for adam')
        parser.add_argument(
            '--D_steps_per_G',
            type=int,
            default=1,
            help='number of discriminator iterations per generator iterations.')

        # for discriminators
        parser.add_argument(
            '--ndf',
            type=int,
            default=64,
            help='# of discrim filters in first conv layer')
        parser.add_argument(
            '--lambda_vgg',
            type=float,
            default=10.0,
            help='weight for vgg loss')
        parser.add_argument(
            '--no_vgg_loss',
            action='store_true',
            help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument(
            '--gan_mode',
            type=str,
            default='hinge',
            help='(ls|original|hinge)')
        parser.add_argument(
            '--vgg_mode',
            type=str,
            default='relu',
            help='(pool|relu|2254)')

        parser.add_argument(
            '--lambda_gan',
            type=float,
            default=1.,
            help='weight for gan loss')
        parser.add_argument(
            '--netD',
            type=str,
            default='hybrid',
            help='(hybrid)')
        parser.add_argument(
            '--no_TTUR',
            action='store_true',
            help='Use TTUR training scheme')

        parser.add_argument(
            '--lambda_L1',
            type=float,
            default=1.0,
            help='weight for L1 loss')
        parser.add_argument(
            '--no_L1_loss',
            action='store_true',
            help='if specified, do not use L1 reconstruction loss')

        parser.add_argument(
            '--target_real_label',
            type=float,
            default=1.0,
            help='real label for GAN loss')
        parser.add_argument(
            '--target_fake_label',
            type=float,
            default=0.0,
            help='fake label for GAN loss')
        self.isTrain = True
        return parser
