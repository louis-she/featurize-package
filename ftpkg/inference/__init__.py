import cv2
import minetorch
import pandas as pd
import torch.nn.functional as F
import numpy as np
from featurize_jupyterlab.core import BasicModule, DataflowModule, Option, Task
from featurize_jupyterlab.task import env
from featurize_jupyterlab.utils import get_transform_func

class MixinMeta():
    namespace = 'Segmentation Inference'

class SegmentationSampleInference(Task, MixinMeta):
    
    input_images = Option(name='Predicting images', type='uploader')
    output_activation = Option(name='activation', type='collection', default='None', collection=['None', 'sigmoid', 'softmax'])
    transform = DataflowModule(name='Transform', component_types=['Dataflow'], multiple=True, required=True)
    model = BasicModule(name='Model', component_types=['Model'])
    pixel_threshold = Option(name='Pixel Threshold', type='number', default=0.5)
    
    def mask2rle(self, image_logits, pixel_threshold):
        img = image_logits > pixel_threshold
        pixels= img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

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

        df = pd.DataFrame(columns=['Image Name', *[f'Class {klass} score' for klass in range(len(outputs[0]))]])
        for i, image_path in enumerate(self.input_images):
            image_name = image_path.split('/')[-1]
            row_data = [image_name]
            for klass, output in enumerate(outputs[i]):
                row_data.append(self.mask2rle(output, self.pixel_threshold))
            df.loc[i] = row_data
        df.to_csv('./output.csv', index=False)
        self.env.rpc.add_file('./output.csv')
