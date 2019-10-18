# featurize-package
Official packages for featurize.

# How to write a package

A featurize package is also a Python package, which can be installed via `pip`. The source code of official `ftpkg` is a good start to learn how to write your own featurize package.

Featurize by default provides 5 different types of component, which are `Dataset`, `Dataflow`, `Loss`, `Model`, `Optimizer`. With these components, we can build our machine learning training pipeline. All these components are extend from `Component`.

The following code shows how to registed a new `Dataset`,

```Python
class MyComponent(Dataset):  # MyComponent is a `Dataset`
  """This is my first component
  """
  name = 'My Component'      # Optional, if missing then name will be the class name `MyComponent`
```
