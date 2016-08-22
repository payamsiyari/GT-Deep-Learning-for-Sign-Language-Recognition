# Deep-Learning-for-Sign-Language-Recognition
Learning Hand Features for Sign Language Recognition (GATech Course Project: Deep Learning for Perception CS8803-DL, Spring 2015)

## Installation & Usage
0. The code needs the dataset to run. The dataset could not be provided due to large file size.
The only library needed for running the code is Theano (ver. 0.7)

1. ```image_processor.py``` applies the preprocessing on the data. It needs to be run with the command below:
```python
$ python image_processor.py
```
Customization of parameters should be done in the source code.

2. After running ```image_processor.py```, a .pickle file will be created in the same directory. This .pickle file will be the preprocessed data that should be fed into main algorithms.

3. File ```generateInput.py``` is needed for generating batches from the data. Customization of batch parameters should be done within the source code.

4. Files ```2d_convolutional_net.py``` and ```3d_convolutional_net.py``` are the codes for running 2D-CNN and 3D-CNN model on the extracted .pickle file. All parameters for batch sizes and model parameters should be changed within the source code. In other words, once the .pickle file is generarted via ```image_processor.py```, you can run the models with default settings using the commands below:
```python
$ python 2d_convolutional_net.py
$ python 3d_convolutional_net.py
```

5. File ```visualizeWeights.py``` is used for visualization of the filters.

6. Files ```logistic_sgd.py``` and ```mlp.py``` are complementary and contain optimization algorithms and loss function definition.
