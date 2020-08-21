
## Repository Structure

- `main.py` is the main file of train and test model
- `utils.py` utility function
- `data.py` contains data loading and loss function
- `metrics.py` quantitative analysis predition map
- `Config.py`  is the configuration file of model
- `colornorm` is the source code for color deconvolution.


## Setup 

To use the models, modify `Config.py` and run with linux commands. 

## Training 

To train the network, first modify `Config.py` to specify the data to be trained,  <br/>
`self.train_data_path`  :is the dir of training images <br/>
`self.edg_path`         :is the dir of the contour of all the images, including training images and testing images <br/>
`self.label_path`       :is the dir of the ground truth of all the images, including training images and testing images <br/>
Note that the original image and the corresponding GT and contour should have the same file name. <br/>
And then execute the command in linux: <br/>
`python main.py --epoch 150 --train 1` <br/>
`--train 1` means training `--train 0` means testing. The number of GPU number is 1 by default.

## Testing 

When performing the Testing process, first modify `Config.py` to specify the data and the trained model to be loaded, <br/>
`self.test_data_path`  :is the path of testing images <br/>
`self.model_path`      :is the trained model file need to be loaded <br/>
Note that the testing image and the corresponding GT should have the same file name. <br/>
And then execute the command in linux: <br/>
`python main.py  --train 0` <br/>
`--train 1` means training `--train 0` means testing.The number of GPU number is 1 by default. <br/>

