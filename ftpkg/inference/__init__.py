from featurize_jupyterlab.core import Task, BasicModule, DataflowModule, Option
from featurize_jupyterlab.task import env
import minetorch


class MixinMeta():
    namespace = 'inference'


class Inference(Task, MixinMeta):
    output_activation = Option(name='activation', type='collection', default='None', collection=['None', 'sigmoid', 'softmax'])
    pixel_threshold = Option(name='Pixel threshold', type='number', default='0.5')
    dataset = BasicModule(name='Dataset', component_types=['Dataset'])
    transform = DataflowModule(name='Transform', component_types=['Dataflow'], multiple=True, required=False)
    model = BasicModule(name='Model', component_types=['Model'])
    
    def mask2rle(image_logits, pixel_threshold):
        img = image_logits > pixel_threshold
        pixels= img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
   
    def __call__(self):
        fnames = [i.split('/')[-1] for i in self.uploaded_images]
        inference_images = self.dataset
        inputs = [self.transform(image) for image in inference_images]
        outputs = [self.model(input.unsqueeze(0)).squeeze() for input in inputs]
        classes = outputs[0].shape[0]

        if self.output_activation == 'None':
            results = outputs
        elif self.output_activation == 'sigmoid':
            results = [torch.sigmoid(output) for output in outputs]
        elif self.output_activation == 'softmax':
            results = [torch.nn.Softmax(dim=0)(output) for output in outputs]
        else:
            env.logger.exception(f'unexpected error in inferencing process.')
        
        for idx, (fname, result) in enumerate(zip(fnames, results)):
            fig.add_subplot(1, 1, 1)
            img = cv2.imread(self.uploaded_images[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = result > pixel_threshold
            img[mask==1,0] = 128
            plt.imshow(img)
            plt.savefig('./test.png')
            env.rpc.add_file('./submission.csv')