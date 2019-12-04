import minetorch
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

    def __call__(self):
        input_images = [cv2.imread(input_image) for input_image in self.input_images]
        inputs = [self.transform([input_image])[0] for input_image in input_images]
        
        shape = input_images[0].shape
        
        transform = get_transform_func(inputs[0])

        logits = [self.model(transform(input)).squeeze() for input in inputs]
        
        transform_shape = logits[0].shape
        
        if self.output_activation == 'softmax':
            outputs = [F.softmax(logit) for logit in logits]
        elif self.output_activation == 'sigmoid':
            outputs = [F.sigmoid(logit) for logit in logits]
        else:
            outputs = logits
        
        df = pd.DataFrame(columns=['Image Name', 'Image Preview (ImageBase64)', 'Image With Masks (ImageBase64)'])
        
        for idx, image_path in enumerate(self.input_images):
            # DISPLAY IMAGES WITH DEFECTS
            image_name = image_path.split('/')[-1]
            plt.figure(figsize=(0.01 * shape[1], 0.01 * shape[0]))
            img = Image.open(image_path)
            img_array = np.array(img)
            
            if img_array.shape != (transform_shape[1:3]):
                img_array = cv2.resize(img_array, dsize=(transform_shape[2], transform_shape[1]), interpolation=cv2.INTER_LINEAR)
            
            patches = []
            for classes in range(len(outputs[idx])):
                try:
                    msk = outputs[idx][classes]
                except:
                    msk = np.zeros(shape[1:3])
                msk = mask2contour(msk.detach().numpy(),width=2)

                img_array[msk==1,0] = colors[classes][0]
                img_array[msk==1,1] = colors[classes][1]
                img_array[msk==1,2] = colors[classes][2]
                patches.append(mpatches.Patch(color=matplotlib.colors.to_rgba(np.array(colors[classes])/255), label=classes))

            plt.legend(handles=patches)
            plt.axis('off') 
            plt.imshow(img_array)
            plt.subplots_adjust(wspace=0.05)
            plt.savefig(os.path.join('./images', image_name))
            self.env.rpc.add_file(os.path.join('./images', image_name))
            
            base64encode_origin = image_base64(image_path)
            base64encode_processed = image_base64(os.path.join('./images', image_name))

            row_data = [image_name, base64encode_origin, base64encode_processed]
            df.loc[idx] = row_data
        df.to_csv('./output.csv', index=False)
        self.env.rpc.add_file('./output.csv')
        

