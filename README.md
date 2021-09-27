# Binary_classification_phase_separation
Python code for the paper "Binary classification as a phase separation process", by Rafael Monteiro. Further information can be found in the tutorial website below.

## For version 0.0.2 (from 2021) see below:

This is a data repository for the paper "[Binary classification as a phase separation process](https://arxiv.org/abs/2009.02467)", by [Rafael Monteiro](https://sites.google.com/view/rafaelmonteiro-math/home). )

*The database, split as necessary for model fitting, is available for download at* [Zenodo](https://doi.org/10.5281/zenodo.5525794) 


This is a second version, which I wrote using tensorflow/keras. Several other changes have been added as well. Overall, simulations/tests fit into a much smaller file (5 Gb when decompressed), a remarkable improvement when compared to the more than 100 Gb of the previous version.

The new files are: 

  1. PSBC_BCs.tar.gz
  2. PSBC_classifier_PCA.tar.gz
  3. PSBC_dataset.tar.gz
  4. PSBC_libs_grids_statistics.tar.gz
  5. PSBC_notebooks.tar.gz
  
Their content is explained in the file [README_v2.pdf](https://github.com/rafael-a-monteiro-math/Binary_classification_phase_separation/blob/master/README_v2.pdf)

For usage, see [PSBC_Examples.ipynb](https://github.com/rafael-a-monteiro-math/Binary_classification_phase_separation/blob/master/PSBC_Examples.ipynb)


![Evolution of layers during an epoch while training the model at digits "0" and "1" of the MNIST database.](https://github.com/rafael-a-monteiro-math/Binary_classification_phase_separation/blob/master/figures/Example_layers_snapshots_acc_all-min.gif)

**NOTE) I will keep the content for the previous version available in my Github as well. It is still a "nice exercise" to do all that is done in this new version in numpy, as done there. (Or, I should say, they should be studied as a cautionary tale of what to avoid.)**



## For version 0.0.1 (from 2020) see below:

The old data repository for the paper "Binary classification as a phase separation process", by Rafael Monteiro (v1) is in the folder **PSBC_v1**.

Website with description of this project: https://rafael-a-monteiro-math.github.io/Binary_classification_phase_separation/PSBC_v1/index.html
Therein  you will find

  Examples
  1. 1D toy model examples
  2. Computational statistics
  3. Several trained PSBC on MNIST dataset, with different parameter configurations
  4. Extra simulations, investigating normalization properties, low dimensional models that fail due to "too much" model compression, and comparison among ANNs, KNNs, and the PSBC in 1D

If you want to know how to read the data how to access computational statistics, raw data, and examples how to use the data stored in this data repository see the guide PSBC_v1/README.pdf, where a script that downloads (and organizes) all this data is also available ("download_PSBC.sh).

I did not include a copy of the train-test set (0-1dubset of the MNIST database) in every folder with simulations. But you can find a copy of the normalized dataset in the tar ball "PSBC_Examples.tar.gz"  as data_test_normalized_MNIST.csv and data_train_normalized_MNIST.csv.

