This is an implementation for "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"

https://arxiv.org/abs/1506.04214

The idea behind states_dict is to maintain only relevant states. For example, each layer receives the previous time step's h_{t-1} and c_{t-1}. This requires a data structure to store these values. However, h_{t}, which represents the input from the previous layer in the current time step, can be updated in that moment to update the corresponding h_{t-1} and c_{t-1} for that layer. When I access dict[cell_{i-1}][0], I'm actually retrieving the hidden state h_{t} from the previous layer.