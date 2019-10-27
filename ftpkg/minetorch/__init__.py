from featurize_jupyterlab.core import Task, Option, BasicModule, DataflowModule


class Minetorch(Task):

    train_dataloader = DataflowModule(name='Train Dataloader', component_types=['Dataflow'], multiple=True)
    val_dataloader = DataflowModule(name='Validation Dataloader',component_types=['Dataflow'],multiple=True)
    dataset = BasicModule(name='Dataset', component_types=['Dataset'])
    model = BasicModule(name='Model', component_types=['Model'])
    optimizer = BasicModule(name='Optimizer', component_types=['Optimizer'])
    loss = BasicModule(name='Loss', component_types=['Loss'])
    metrics = BasicModule(name='Metirc', component_types=['Metric'])

    def __call__(self):
        pass
