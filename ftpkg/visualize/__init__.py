import minetorch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from featurize_jupyterlab.utils import get_transform_func
from featurize_jupyterlab.core import Task, BasicModule, DataflowModule, Option
from featurize_jupyterlab.task import env
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
    uploaded_images = Option(type='uploader')
    output_activation = Option(name='activation', type='collection', default='None', collection=['None', 'sigmoid', 'softmax'])

    transform = DataflowModule(name='Transform', component_types=['Dataflow'], multiple=True, required=False)
    model = BasicModule(name='Model', component_types=['Model'])

    def __call__(self):
        fnames = [i.split('/')[-1] for i in self.uploaded_images]
        uploaded_images_arrays = [cv2.imread(img) for img in self.uploaded_images]
        transformed_arrays = [self.transform([image])[0] for image in uploaded_images_arrays]
        transform = get_transform_func(transformed_arrays[0])
        outputs = [self.model(transform(input)).squeeze() for input in transformed_arrays]
        shape = outputs[0].shape
        if self.output_activation == 'None':
            results = outputs
        elif self.output_activation == 'sigmoid':
            results = [torch.sigmoid(output) for output in outputs]
        elif self.output_activation == 'softmax':
            results = [torch.nn.Softmax()(output) for output in outputs]
        else:
            env.logger.exception(f'unexpected error in inferencing process.')

        for idx, image_path in enumerate(self.uploaded_images):
            # DISPLAY IMAGES WITH DEFECTS
            plt.figure(figsize=(0.01 * shape[1], 0.01 * shape[0]))
            img = Image.open(image_path)
            img_array = np.array(img)
            patches = []
            for classes in range(len(results[idx])):
                try:
                    msk = results[idx][classes]
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
            plt.savefig(fnames)
            env.rpc.add_file(fnames)
