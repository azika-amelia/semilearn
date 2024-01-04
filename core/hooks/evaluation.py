# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
from .hook import Hook


class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training
    """
    
    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")
            eval_dict = algorithm.evaluate('eval')
            algorithm.log_dict.update(eval_dict)

            # update best metrics
            if algorithm.log_dict['eval/balanced_acc'] > algorithm.best_balanced_acc:
                algorithm.best_balanced_acc = algorithm.log_dict['eval/balanced_acc']
                algorithm.best_it = algorithm.it
                
                # Reset consecutive_epochs counter when there's an improvement
                algorithm.consecutive_epochs_without_improvement = 0
            else:
                algorithm.consecutive_epochs_without_improvement += 1
                # Check if early stopping condition is met
                if algorithm.consecutive_epochs_without_improvement >= algorithm.patience:
                    algorithm.print_fn("Early stopping triggered.")
                    algorithm.should_stop_training = True

            
    
    def after_run(self, algorithm):
        
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            algorithm.save_model('latest_model.pth', save_path)

        results_dict = {'eval/best_acc': algorithm.best_balanced_acc, 'eval/best_it': algorithm.best_it}
        if 'test' in algorithm.loader_dict:
            # load the best model and evaluate on test dataset
            best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
            algorithm.load_model(best_model_path)
            test_dict = algorithm.evaluate('test')
            results_dict['test/best_acc'] = test_dict['test/best_balanced_acc']
        algorithm.results_dict = results_dict
