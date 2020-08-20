
## Repository Structure

- `main.py` is the main file of train and test model
- `utils.py` Utility function
- `data.py` contains data loading and loss function
- `metrics.py` Quantitative analysis predition map
- `Config.py`  is the configuration file of model
- `colornorm` is the python code of H&E stain images normalization. Rewritten from matlab code public in [Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images]. But not involved in this model. Some thing about corlor normalization has discussed in Table 3 in the paper.


## Setup 

To use the models, modify `Config.py` and run with linux commands. 

## Training 

To train the network, the command in liunx is : <br/>
`python main.py --epoch 150 --train 1` <br/>
--train 1 means training --train 0 means testing.The number of GPU number is 1 by default.

## Testing 

When performing the validation process, first set the `self.model_path` variable of `Config.py` to specify the weight to be loaded
To train the network, the command in liunx is : <br/>
`python main.py  --train 0` <br/>
--train 1 means training --train 0 means testing.The number of GPU number is 1 by default.

