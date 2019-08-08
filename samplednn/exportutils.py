import os
import time

import torch


class ExportUtils():

    def setup_folder_structure(self, export_name):

        if export_name is None:
            export_name = 'experiment_{}'.format(int(time.time()))
            print('WARNING: No export_name was given. Using generic_name \'{}\''.format(
                export_name))

        export_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..',
            'models',
            self.short_name))
        params_base_dir = os.path.abspath(os.path.join(
            export_dir,
            'params'
        ))
        params_dir = os.path.abspath(os.path.join(
            params_base_dir,
            export_name
        ))
        index_dir = os.path.abspath(os.path.join(
            export_dir,
            'index'
        ))

        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(params_base_dir, exist_ok=True)
        os.makedirs(params_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)

        self.folder_structure_setup = True

        return export_name, params_dir, index_dir

    def save_params(self, epoch, export_name, params_dir):

        filename = os.path.abspath(os.path.join(
            params_dir,
            '{}_ep{:03}'.format(export_name, epoch)
        ))
        self.save(filename)
        print('Saved parameters to \'{}\''.format(filename))

    def save_index(self, dataloader, train_info, export_name, index_dir):

        filename = os.path.abspath(os.path.join(
            index_dir,
            '{}.txt'.format(export_name)
        ))

        loader_dict = dataloader.__dict__
        network_dict = self.__dict__['_modules']

        with open(filename, 'w+') as txt:
            def write(string):
                print(string, file=txt)

            write('Configuration of experiment \'{}\''.format(export_name))
            write('=' * (30 + len(export_name)))
            write('\nNetwork architecture\n--------------------\n')
            for key in network_dict:
                write('{}: {}'.format(key, network_dict[key]))
            write('\nData loader information\n-----------------------\n')
            for key in loader_dict:
                write('{}: {}'.format(key, loader_dict[key]))
            write('\nTraining information\n--------------------\n')
            for key in train_info:
                write('{}: {}'.format(key, train_info[key]))


    def load_predict_export(self,  model_name, dataloader, num_samples, export_name):

        export_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..',
            'models',
            self.short_name))
        params_base_dir = os.path.abspath(os.path.join(
            export_dir,
            'params'
        ))
        params_dir = os.path.abspath(os.path.join(
            params_base_dir,
            model_name
        ))
        pred_base_dir = os.path.abspath(os.path.join(
            export_dir,
            'predictions'
        ))
        pred_dir = os.path.abspath(os.path.join(
            pred_base_dir,
            export_name
        ))

        os.makedirs(pred_base_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        for name in os.listdir(params_dir):
            param_file = os.path.abspath(os.path.join(
                params_dir,
                name
            ))
            pred_file = os.path.abspath(os.path.join(
                pred_dir,
                name
            ))
            print('Loading parameters from \'{}\''.format(param_file))
            self.load(param_file)
            # print('Predicting ...')
            predictions = self.predict_all(dataloader, num_samples)
            print('Saving predictions to \'{}\''.format(pred_file))
            torch.save(predictions, pred_file)
            print('-' * 15)
        
        print('Done!')
