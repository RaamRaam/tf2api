# tf2api

The repository provides functional interfaces that allows model building, check-pointing, branching and deploying into production


# Resource

https://www.tensorflow.org/tutorials/quickstart/advanced

# Procedure

* Open example4execute.ipynb in colab
* enable GPU
* Connect to google drive if want to save the models permanently

* Alternately clone with the following command
* _git clone https://github.com/RaamRaam/tf2api.git_
* Open example4execute.ipynb in Jupyter notebook

# Source Files and functionalities
* __train.py__
  * _train_ class
  * _call_
    * Trains the model for provided hyper parameters by taking train and test data as input
  * _save_
    * Saves the model in the path provided. Expects a folder in the name of model in the object
  * _load_
    * Loads the model from the path provided
  * _evaluate_
    * evaluates the accuracy of the model
  * *_deep_learn_*
    * Executes Graph of the model
  * *_linear_lr_*
    * Dynamic LR finder depending on inputs
  * *_savehistory_*
    * Saves history of the model parameters
  * *_savelrhistory_*
    * Saves LR history
  
* __tfrecords.py__
  * _SaveTFRecordSet_
  * _ds_ class
    * _ReadTFRecordSet_
    * _FilterTFRecordSet_
