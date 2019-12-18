import minetorch
import torch
import cv2
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from featurize_jupyterlab.utils import get_transform_func
from featurize_jupyterlab.core import Task, BasicModule, DataflowModule, Option
from featurize_jupyterlab.task import env
from featurize_jupyterlab.utils import get_transform_func, image_base64
from PIL import Image, ImageDraw, ImageFont


colors = [
    (255,255,0),
    (0,255,255),
    (255,0,255),
    (0,0,255),
    (0,255,0),
    (255,0,0)
    ]


def mask2contour(mask, width=1):
    # CONVERT MASK TO ITS CONTOUR
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3) 


class MixinMeta():
    namespace = 'visualize'


class Visualize(Task, MixinMeta):

    input_images = Option(name='Predicting images', type='uploader')
    output_activation = Option(name='activation', type='collection', default='None', collection=['None', 'sigmoid', 'softmax'])
    transform = DataflowModule(name='Transform', component_types=['Dataflow'], multiple=True, required=True)
    model = BasicModule(name='Model', component_types=['Model'])
    pixel_threshold = Option(name='Pixel Threshold', type='number', default=0.5)

    def __call__(self):
        input_images = [cv2.imread(input_image) for input_image in self.input_images]
        inputs = [self.transform([input_image])[0] for input_image in input_images]
        
        shape = input_images[0].shape
        
        transform = get_transform_func(inputs[0])
        
        self.model.eval()
        logits = [self.model(transform(input)).squeeze() for input in inputs]
        
        transform_shape = logits[0].shape
        
        if self.output_activation == 'softmax':
            outputs = [F.softmax(logit) for logit in logits]
        elif self.output_activation == 'sigmoid':
            outputs = [torch.sigmoid(logit) for logit in logits]
        else:
            outputs = logits
        
        df = pd.DataFrame(columns=['Image Name', 'Image Preview (ImageBase64)', 'Image With Masks (ImageBase64)'])
        
        for idx, image_path in enumerate(self.input_images):
            # DISPLAY IMAGES WITH DEFECTS
            image_name = image_path.split('/')[-1]
            plt.figure(figsize=(0.01 * shape[1], 0.01 * shape[0]))
            img = Image.open(image_path)
            img_array = np.array(img)
            patches = []
            for classes in range(len(outputs[idx])):
                try:
                    msk = cv2.threshold(outputs[idx][classes].detach().numpy(), self.pixel_threshold, 1, cv2.THRESH_BINARY)[1]
                    if msk.shape != shape[0:2]:
                        msk = cv2.resize(msk, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                except:
                    msk = np.zeros(shape[0:2])
                msk = mask2contour(msk,width=5)

                img_array[msk==1,0] = colors[classes][0]
                img_array[msk==1,1] = colors[classes][1]
                img_array[msk==1,2] = colors[classes][2]
                patches.append(mpatches.Patch(color=matplotlib.colors.to_rgba(np.array(colors[classes])/255), label=f'Class {classes+1}'))

            plt.legend(handles=patches)
            plt.axis('off') 
            plt.imshow(img_array)
            #plt.subplots_adjust(wspace=0.001)
            plt.savefig(os.path.join('/tmp', image_name), bbox_inches='tight', pad_inches=0.0)
            self.env.rpc.add_file(os.path.join('/tmp', image_name))
            
            base64encode_origin = image_base64(image_path)
            base64encode_processed = image_base64(os.path.join('/tmp', image_name))

            row_data = [image_name, base64encode_origin, base64encode_processed]
            df.loc[idx] = row_data
        df.to_csv('./output.csv', index=False)
        self.env.rpc.add_file('./output.csv')
        

