from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt

class nnUNetLoggerWithClassification(nnUNetLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list(),
            'class_accuracy': list(),
            'class_total': list(),
            'macro_f1': list(),
            'whole_pancreas_dsc': list(),
            'lesion_dsc': list(),
            'micro_dice_score': list()
        }

    def plot_progress_png(self, output_folder):
        # we infer the epoch form our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(3, 1, figsize=(30, 54))
        # regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice",
                 linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)",
                 linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['class_accuracy'][:epoch + 1], color='y', ls='-', label="val class accuracy",
                 linewidth=4)
        
        ax2.plot(x_values, self.my_fantastic_logging['macro_f1'][:epoch + 1], color='y', ls='-', label="macro f1",
                 linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['whole_pancreas_dsc'][:epoch + 1], color='y', ls='-', label="whole pancreas dsc",
                 linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['lesion_dsc'][:epoch + 1], color='y', ls='-', label="lesion dsc",
                 linewidth=4)   
        ax2.plot(x_values, self.my_fantastic_logging['micro_dice_score'][:epoch + 1], color='y', ls='-', label="micro dice score",
                 linewidth=4)

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        ax = ax_all[1]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()