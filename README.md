# featurize-package
Official packages for featurize.

# How to write a package

A featurize package is also a Python package, which can be installed via `pip`. The source code of official `ftpkg` is a good start to learn how to write your own featurize package.

Featurize by default provides 5 different types of component, which are `Dataset`, `Dataflow`, `Loss`, `Model`, `Optimizer`. With these components, we can build our machine learning training pipeline. All these components are extended from `Component`.

## A solid example

The following code shows how to registed a new `Dataset`,

```Python
from featurize_jupyterlab.core import Dataset, Option

class MyDataset(Dataset):
    """This is my very first dataset component
    """

    # meta information
    name = 'My Dataset'  # Optional, if missing then name will be the class name `MyDataset`
    description = 'This is my very first dataset component'  # Optional, if missing then this will be the doc string
  
    # define component options
    folder = Option(type='string', default='/datasets', help='please input the folder of the dataset files')
    download = Option(type='boolean', default=True)
    datatype = Option(type='collection', collection=[['Imagenet', 'imagenet'], ['MS Coco', 'coco']], default='imagenet')
  
    def initialize(self):
        """If you want to do some thing for initializing the component, write it in
        this method instead of `__init__`. If you really want to override the `__init__`
        function, be sure to know what you are doing.

        In every function except the `__init__`, you can use the self.${option_name} to access
        the user setting, e.g.

        `self.foler` is of type `string`, not `Option`, and the string is the path of the folder the user inputed.
        """
        pass

    def __call__(self):
        """For Dataset component, the return value is a tuple like (train_dataloader, val_dataloader),
        both the dataloader is of type `torch.utils.data.Dataloader`.
        """
        train_dataloader = ...
        val_dataloader = ...
        return (train_dataloader, val_dataloader)
```

Above example basicly shows how to define a `Component`:

1. Required. Define a class extends one of the 5 basic Components.
2. Optional. Define the metainfo if you want, or leave them as default.
3. Optional, but usually necessary. Define the options of this Component. This let the frontend user be able to tweak parameters via the web UI.
4. Required. Define a `__call__` method which is the main logic of the component. The meaning of its parameters and return value is depend on the type of the component.

> Hint: some component type may have predefined options, if you peek the parent class of your component, it may have already defined some `Option`s. You can directly use these option in the `__call__` function.

The following table shows the signature of `__call__` method of different types of component:

| Component Type | Predefined Options | Parameters | Return Value |
| ------------- | ------------- | ------------- | ------------|
| `Dataset` | 1. `train_dataflow` It's a `dataflow` which define train data preprocessing <br/> 2. `val_dataflow` It's a `dataflow` which define validation data preprocessing  | None | A tuple of 2 `torch.utils.data.Dataloader` instance, first for train, second for validation |
| `Model`  | None | None  | A instance of `torch.nn.Module` |
| `Optimizer` | `model` It's the return value of the `Model` component you defined previous | None | A instance of `torch.optim.Optimizer` |
| `Loss` | `trainer` It's a instance of [Minetorch Trainer](https://github.com/minetorch/minetorch), it's useful when you want to hack in to the traning process.  | `data` It's the batched yield value of the dataloader returned by `Dataset` component | A scalar of type `torch.Tensor`, which is the loss |
| `Dataflow` | None | `rowdata` It's a row of data, which is returned by `Dataset` component | A processed row of data |


## Transformation

`Transformation` is a `Dataflow` component. It is mainly used as define the data transform pipeline.
The [test code](https://github.com/louis-she/featurize-jupyterlab/blob/master/featurize_jupyterlab/tests/test_transformation.py) of `featurize-jupyterlab` is a set of good examples to see how to write a `Transformation`.

There are 4 kinds of transformation class which you can chose to inherit based on what you want.

| Transformation Name | Inherit From | Description |
| ------------- | ------------- | ------------- |
| `BasicTransformation` | `Dataflow` | Base class of transformation |
| `BasicImageTransformation` | `BasicTransformation` | Basic image transformation, it's used for single image transoform. |
| `MeaningfulColumnsTransformation` | `BasicTransformation` | A transformation which will define the meaning of selected column and use them in the logic of transform |
| `DualImageTransformation` | `MeaningfulColumnsTransformation` | A image transformation of `MeaningfulColumnsTransformation`, the meaning of the columns have already been defined in the class which is `image`, `mask`, `masks`, `keypoints` and `bboxes` |


Followings are 2 transformation examples with fully comments

```Python
from albumentations import RandomCrop, RandomBrightnessContrast
from featurize_jupyterlab.transform import DualImageTransformation, BasicImageTransformation


class Brightness(BasicImageTransformation):  # This is a single image transformation
    """Add a fixed integer to all pixels of an image
    """
    brightness_limit = Option(type='number', default=0.2)
    contrast_limit = Option(type='number', default=0.2)
    brightness_by_max = Option(type='number')

    def create_aug(self):
        return RandomBrightnessContrast(
            self.brightness_limit,
            self.contrast_limit,
            self.brightness_by_max,
            p=self.probability  # probability is a predefined Option
        )


class Crop(DualImageTransformation):  # This is a dual image transformation
    """Apply random crop to images, masks, keypoints and bounding boxes
    """
    width = Option(type='number')
    height = Option(type='number')

    def create_aug(self):
        return RandomCrop(
            self.height,
            self.width,
            p=self.probability  # again, probability is predefined
        )
```
