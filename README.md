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
  > Trains the model for provided hyper parameters by taking train and test data as input
  * _save_
  * _load_
  * _evaluate_
  * *_deep_learn_*
  * *_linear_lr_*
  * *_savehistory_*
  * *_savelrhistory_*
  * *_initialize1_*
  * *_initialize2_*
  
* __tfrecords.py__
  * _SaveTFRecordSet_
  * _ds_ class
    * _ReadTFRecordSet_
    * _FilterTFRecordSet_
