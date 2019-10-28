from featurize_jupyterlab.core import Task, BasicModule, DataflowModule
from featurize_jupyterlab.task import env
import minetorch


class CorePlugin():
    """The Minetorch Trainer can be runned independently.
    This plugin activate Trainer with the ability to communicate with the
    Minetorch Server with some basic data collection such as loss.
    """
    def after_init(self, payload, trainer):
        env.rpc.create_graph('train_epoch_loss')
        env.rpc.create_graph('val_epoch_loss')
        env.rpc.create_graph('train_iteration_loss')

    def after_epoch_end(self, payload, trainer):
        env.rpc.add_point('train_epoch_loss', payload['epoch'], payload['train_loss'])
        env.rpc.add_point('val_epoch_loss', payload['epoch'], payload['val_loss'])

    def after_train_iteration_end(self, payload, trainer):
        env.rpc.add_point('train_iteration_loss', payload['iteration'], payload['loss'])


class Minetorch(Task):

    # create module
    train_dataloader = DataflowModule(name='Train Dataloader', component_types=['Dataflow'], multiple=True, required=False)
    val_dataloader = DataflowModule(name='Validation Dataloader', component_types=['Dataflow'], multiple=True, required=False)
    dataset = BasicModule(name='Dataset', component_types=['Dataset'])
    model = BasicModule(name='Model', component_types=['Model'])
    loss = BasicModule(name='Loss', component_types=['Loss'])
    metrics = BasicModule(name='Metirc', component_types=['Metric'], required=False)
    optimizer = BasicModule(name='Optimizer', component_types=['Optimizer'])

    # create dependencies
    optimizer.add_dependency(model)
    dataset.add_dependency(train_dataloader, val_dataloader)

    def __call__(self):
        train_dataset, val_dataset = self.dataset

        trainer = minetorch.Trainer(
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
            trainer.train()
        except Exception as e:
            env.logger.exception(f'unexpected error in training process: {e}')
