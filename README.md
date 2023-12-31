# Introduction
This is an implementation for "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"

The dataset used in main.py is MovingMnist

https://arxiv.org/abs/1506.04214

The idea behind states_dict is to maintain only relevant states. For example, each layer receives the previous time step's h_{t-1} and c_{t-1}. This requires a data structure to store these values. However, h_{t}, which represents the input from the previous layer in the current time step, can be updated in that moment to update the corresponding h_{t-1} and c_{t-1} for that layer. When I access dict[cell_{i-1}][0], I'm actually retrieving the hidden state h_{t} from the previous layer.

# Usage
you can
```python
from convLSTM import ConvLSTM2d
``` 
it is like how we use `nn.Conv2d`

If you want to run main.py ,first download MovingMnist from https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
Then enter `python main.py` in your terminal to run the code,the output will be put into the folder `output_images`

# Output
(model only be trained two epoch since it really cost my computation resource)

![target](https://github.com/usamimeri/ConvLSTMPytorch/blob/main/images/1_target.gif)
![output](https://github.com/usamimeri/ConvLSTMPytorch/blob/main/images/1_output.gif)

![target](https://github.com/usamimeri/ConvLSTMPytorch/blob/main/images/2epoch_1_target_.gif)
![output](https://github.com/usamimeri/ConvLSTMPytorch/blob/main/images/2epoch_1_output_.gif)

