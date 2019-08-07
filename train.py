import sys
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.trainer import Trainer

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer1 = Trainer(opt)
trainer2 = Trainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):

        iter_counter.record_one_iteration()

        data_i['CT'], data_i['MR'] = data_i['CT'].squeeze(
            1), data_i['MR'].squeeze(1)
        
        #aligned
        # train  F (MR to synCT)
        trainer.run_generator_one_step(data_i)
        data_i['synCT'] =  trainer.get_latest_generated()
        # train Df
        trainer.run_discriminator_one_step(data_i)
        # train  G (synCT to synMR) for cycle
        trainer.run_generator_one_step(data_i, is_cylce = True) 
        data_i['cylceMR'] =  trainer.get_latest_generated()
        

        disjoin(data_i)
        
        #unaligned
        # train  G (CT to synMR)
        trainer.run_generator_one_step(data_i, is_forward=False)
        data_i['synMR'] =  trainer.get_latest_generated()
        # train Dg
        trainer.run_discriminator_one_step(data_i, is_forward=False)
        # train  F (synMR to synCT) for cycle
        trainer.run_generator_one_step(data_i, is_forward=False, is_cylce = True) 
        data_i['cylceCT'] =  trainer.get_latest_generated()

        
        
        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(
                losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image',
                                    trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(
                visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    # save the model
    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
