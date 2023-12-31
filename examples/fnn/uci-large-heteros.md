# UCI regression problem with heteroscedasticity

**Author:** [Miquel Florensa](https://www.linkedin.com/in/miquel-florensa/)  
**Date:** 2023/05/23  
**Description:** This example shows how to perform a regression task on the Boston housing prices dataset using a FNN with heteroscedasticity.  

<a href="https://github.com/CivML-PolyMtl/cutagi-doc/tree/main/code/uci_heteros_regression_runner.py" class="github-link">
  <div class="github-icon-container">
    <img src="../../images/GitHub-Mark.png" alt="GitHub" height="32" width="64">
  </div>
  <div class="github-text-container">
    Github Source code
  </div>
</a>

---

## 1. Setup

```python
from python_examples.data_loader import RegressionDataLoader
from python_examples.regression import Regression
from pytagi import NetProp
```

?>Notice that these modules are described [here](modules/modules.md) and the source code is in the *python_examples* directory, in case you have the modules in another directory you must change this paths.

## 2. Prepare the data

In this simple example we will use the Boston housing dataset and we will predict the housing prices along with heteroscedastic uncertainty given the 13 explanatory variables.

```python
# User-input
num_inputs = 13     # 1 explanatory variable
num_outputs = 1     # 1 predicted output
num_epochs = 50     # row for 50 epochs
x_train_file = "./data/UCI/Boston_housing/x_train.csv"
y_train_file = "./data/UCI/Boston_housing/y_train.csv"
x_test_file = "./data/UCI/Boston_housing/x_test.csv"
y_test_file = "./data/UCI/Boston_housing/y_test.csv"
```

**You can find the data used in the [UCI data](https://github.com/lhnguyen102/cuTAGI/tree/main/data/UCI) repository.*

## 3. Create the model

We use a FNN with a simple architecture as defined in the HeterosUCIMLP class wich is suited for this regression problem with heteroscedasticity.

```python
class HeterosUCIMLP(NetProp):
    """Multi-layer preceptron for regression task where the
    output's noise varies overtime"""

    def __init__(self) -> None:
        super().__init__()
        self.layers =       [1, 1, 1, 1]
        self.nodes =        [13, 50, 50, 2]  # output layer = [mean, std]
        self.activations =  [0, 4, 4, 0]
        self.batch_size =   10
        self.sigma_v =      2
        self.sigma_v_min =  0.3
        self.noise_gain =   1.0
        self.noise_type =   "heteros"
        self.init_method =  "He"
        self.device =       "cpu"
```

```python
# Model
net_prop = HeterosUCIMLP()
```

## 4. Load the data

We use of the [RegressionDataLoader](modules/data-loader?id=data-loader) class to load and process the data. The *process_data* function requires the input and output test and training files in a **csv** format.

```python
# Data loader
reg_data_loader = RegressionDataLoader(num_inputs=num_inputs,
                                       num_outputs=num_outputs,
                                       batch_size=net_prop.batch_size)
                                       
data_loader = reg_data_loader.process_data(x_train_file=x_train_file,
                                           y_train_file=y_train_file,
                                           x_test_file=x_test_file,
                                           y_test_file=y_test_file)
```

## 5. Train and evaluate the model

Using the [regression class](modules/regression?id=regression-class) we train and test the model using TAGI. When doing the prediction step we can specify the standard deviation factor defining the confidence interval.

```python
reg_task = Regression(num_epochs=num_epochs,
                      data_loader=data_loader,
                      net_prop=net_prop)

reg_task.train()
reg_task.predict()
```

## 6. Results

At the end of the execution the results are printed in the console as seen below.

> MSE           :  4.66  
> Log-likelihood: -3.84
