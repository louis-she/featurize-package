from featurize_jupyterlab.core import Task, BasicModule, DataflowModule, Option
from featurize_jupyterlab.task import env
import minetorch


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
        env.rpc.add_file(payload['modelpath'])

    def after_train_iteration_end(self, payload):
        env.rpc.add_point('train_iteration_loss', payload['iteration'], payload['loss'])



class Inference(Task, MixinMeta):
    uploaded_images = Option(type='upload')
    image_height = Option(type='number', required=True)
    image_width = Option(type='number', required=True)
    transform = DataflowModule(name='Transform', component_types=['Dataflow'], multiple=True, required=False)
    model = BasicModule(name='Model', component_types=['Model'])
    output_activation = Option(name='activation', type='collection', default='None', collection=[['None', 'sigmoid', 'softmax']])

    def __call__(self):
        inference_images = [cv2.imread(img for img in self.uploaded_images)]
        inputs = [self.transform(image) for image in inference_images)]
        outputs = [self.model(input.unsqueeze(0)).squeeze() for input in inputs]
        if self.output_activation == 'None':
            results = outputs
        elif self.output_activation == 'sigmoid':
            results = [torch.sigmoid(output) for output in outputs]
        elif self.output_activation == 'softmax':
            results = [torch.nn.Softmax(dim=0)(output) for output in outputs]
        else:
            env.logger.exception(f'unexpected error in inferencing process.')
        fnames = [i.split('/')[-1] for i in self.uploaded_images]
        tmp = []
        for fname, result in zip(fnames, results):
            tmp.append((fname,result))
        df = pd.DataFrame(tmp, columns=['fname', 'predictions'])
        df.to_csv('./submission.csv', index=False)
        env.rpc.add_file('./submission.csv')