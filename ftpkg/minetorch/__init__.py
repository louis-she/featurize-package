import cv2
import minetorch
import pandas as pd
import torch.nn.functional as F

from featurize_jupyterlab.core import BasicModule, DataflowModule, Option, Task
from featurize_jupyterlab.task import env
from featurize_jupyterlab.utils import get_transform_func, image_base64

PYPI_PACKAGE_NAME = 'featurize-package'

class MixinMeta():
    namespace = 'minetorch'


class CorePlugin(minetorch.Plugin):
    """The Minetorch Trainer can be runned independently.
    This plugin activate Trainer with the ability to communicate with the
    Minetorch Server with some basic data collection such as loss.
    """
    def after_init(self, payload):
        env.rpc.create_graph('train_epoch_loss')
        env.rpc.create_graph('val_epoch_loss')
        env.rpc.create_graph('train_iteration_loss')

    def after_epoch_end(self, payload):
        env.rpc.add_point('train_epoch_loss', payload['epoch'], payload['train_loss'])
        env.rpc.add_point('val_epoch_loss', payload['epoch'], payload['val_loss'])

    def after_checkpoint_persisted(self, payload):
        env.rpc.add_checkpoint(payload['modelpath'])

    def after_train_iteration_end(self, payload):
        env.rpc.add_point('train_iteration_loss', payload['iteration'], payload['loss'])


class Minetorch(Task, MixinMeta):
    # create module
    train_transform = DataflowModule(name='Train Transform', component_types=['Dataflow'], multiple=True, required=False)
    val_transform = DataflowModule(name='Validation Transform', component_types=['Dataflow'], multiple=True, required=False)
    dataset = BasicModule(name='Dataset', component_types=['Dataset'])
    model = BasicModule(name='Model', component_types=['Model'])
    loss = BasicModule(name='Loss', component_types=['Loss'])
    metrics = BasicModule(name='Metirc', component_types=['Metric'], required=False)
    optimizer = BasicModule(name='Optimizer', component_types=['Optimizer'])

    # create dependencies
    optimizer.add_dependency(model)
    dataset.add_dependency(train_transform, val_transform)

    def __call__(self):
        train_dataset, val_dataset = self.dataset

        miner = minetorch.Miner(
            alchemistic_directory='./log',
            model=self.model,
            optimizer=self.optimizer,
            train_dataloader=train_dataset,
            val_dataloader=val_dataset,
            loss_func=self.loss,
            drawer=None,
            logger=env.logger,
            plugins=[CorePlugin()]
        )

        try:
            miner.train()
        except Exception as e:
            env.logger.exception(f'unexpected error in training process: {e}')


class ClassificationSampleInference(Task, MixinMeta):

    input_images = Option(name='Predicting images', type='uploader')
    output_activation = Option(name='activation', type='collection', default='None', collection=['None', 'sigmoid', 'softmax'])
    transform = DataflowModule(name='Transform', component_types=['Dataflow'], multiple=True, required=True)
    model = BasicModule(name='Model', component_types=['Model'])

    def __call__(self):
        input_images = [cv2.imread(input_image) for input_image in self.input_images]
        inputs = [self.transform([input_image])[0] for input_image in input_images]

        transform = get_transform_func(inputs[0])

        logits = [self.model(transform(input)).squeeze() for input in inputs]

        if self.output_activation == 'softmax':
            outputs = [F.softmax(logit) for logit in logits]
        elif self.output_activation == 'sigmoid':
            outputs = [F.sigmoid(logit) for logit in logits]
        else:
            outputs = logits

        df = pd.DataFrame(columns=['Image Name', 'Image Preview (ImageBase64)', *[f'Class {klass} score' for klass in range(len(outputs[0]))]])
        for i, image_path in enumerate(self.input_images):
            image_name = image_path.split('/')[-1]
            base64encode = image_base64(image_path)
            row_data = [image_name, base64encode]
            for klass, klass_score in enumerate(outputs[i]):
                row_data.append(klass_score.item())
            df.loc[i] = row_data
        df.to_csv('./output.csv', index=False)
        self.env.rpc.add_file('./output.csv')
