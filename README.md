# Pytorch Implementation of experiments for the paper 'Constructing compressed number lines of latent variables using a cognitive model of memory and deep neural networks'

## Data Generation:

To generate the synthetic datasets for the experiments run -
```
Example-

python generate_datasets.py --gen_type simple --scale_data 1 --num_examples 50 --dataset_dir outputs/datasets
```
- gen_type (str)
    - simple: input predicts label at fixed interval of steps
    - complex: input predicts label under the influence of modulatory inputs (default values are 1 input, 2 modulatory inputs)
    - combined: combines both simple and complex data to build a combined dataset
    - complex_multiple_modulation : input predicts label under the influence of multiple modulatory inputs that consist noise
- scale_data (int) : length of the experiment is equal to 200*scale_data
- num_examples (int) : number of examples to generate
- dataset_dir (str) : directory where the dataset is stored


## Data Format
The dataset is in .npz format. (look at load_datasets function in alpha_train.py for data loading for training, validation and testing)
```
np.savez_compressed(dataset_path, x=x, y=y)
```
x and y  are numpy arrays that represents features of shape- (num_examples, sequence_length, num_of_features)  and labels of shape- (num_examples, sequence_length, num_of_labels). The features that represent the modulatory inputs are always defined as the trailing features in x. Example - if num_of_features are 5 where 3 features represent modulatory inputs and 2 features represent inputs, then features are ordered such that [0:2) are features representing inputs and [2:5) are features representing modulatory inputs (index starting at 0).


## Training Model
To train the model run-
```
Example-

python alpha_train.py --model_type SITH --dataset outputs/datasets/simple_1.npz --train_size 3 --valid_size 12 --test_size 35 --epochs 1000 --lr 0.1 --batch_size 2 --n_taus 50 --tstr_min 0.005 --tstr_max 20.0 --k 8 --g 1 --dt 0.001 --l2_penalty 0 --num_inputs_sith 1 --num_extern_inputs_sith 0 --output_dir outputs
```
Model Parameters -
- model_type (str)
    - SITH: Globally modulated Scale-Invariant Temporal History (G-SITH) along with linear layers to make predictions
    - SITH_F: G-SITH without post inversion along with linear layers to make predictions
    - RNN1FC: Simple RNN and 1 linear layer
    - RNN2FC: Simple RNN and 2 linear layers
    - LSTM1FC: LSTM and 1 linear layer
    - LSTM2FC: LSTM and 2 linear layers
    - GRU1FC: GRU and 1 linear layer
    - GRU2FC: GRU and 2 linear layers
    - LMU : LMU (Legendre Memory Units) and 1 linear layer
    - coRNN : coRNN (Coupled Oscillatory Recurrent Neural Network) and 1 linear layer

- dataset (str) : path of dataset to be used for model training, validation and testing
- train_size (int) : size of training set
- valid_size (int) : size of validation set
- test_size (int) : size of test set
- epochs (int) : number of epochs to train
- lr (float) : learning rate for training
- batch_size (int) : size of each batch for training
- l2_penalty (float) : L2 regularization parameter for training
- bptt_type (str) : type of backprop for training the model (bptt or tbptt)
- k_1 (int) : every k_1 steps, backpropogate through time for the k_1 steps
- loss_type (str) : loss function for training (mse or bce)
- output_dir (str) : directory for storing tensorboard logs of training loss, validation loss and test loss (weighted and unweighted binary cross entropy loss) and model checkpoints.

G-SITH parameters (only for G-SITH) - 
- n_taus (int) : number of taustar nodes in f_tilda
- tstr_min (float) : peak time of the first taustar node
- tstr_max (float) : peak time of the last taustar node
- k (int) : order of the derivative in the inverse laplace transform
- g (int) : (0 or 1) amplitude scaling of nodes in the inverse laplace transform output
- dt (float) : time step of the simulation
- num_inputs_sith (int) : number of inputs in the dataset
- num_extern_inputs_sith (int) : number of modulatory inputs in the dataset

LMU parameters (only for Legendre Memory Units) - 
- order (int) : degree of polynomials
- theta (float) : sliding window length in the memory cell dynamics

coRNN parameters (only for Coupled Oscillatory Recurrent Neural Network) -
- dt_cornn (float) : discretization constant
- gamma_cornn (float) : control parameter for CoRNN
- epsilon_cornn (float) : control parameter for CoRNN

# File Descriptions
- alpha_train.py : loads the respective dataset and contains training, validation and testing procedures using the respective models. The loss values (weighted and unweighted binary cross entropy) for training, validation and testing are logged in tensorboard. The tensorboard logs are stored in the runs folder in the output directory. The model checkpoints during training are stored in model_checkpoints folder in the output directory and the model checkpoint with the lowest validation loss during training is stored in best_model_checkpoint folder in the directory
- generate_datasets.py : generates synthetic data and stores the data in the specified directory
- WM.py : contains WM class that implements Laplace transform and Inverse Laplace transform
- models
    - WMPred.py : contains classes - alpha_prediction, f_tilda_prediction, dim_reduce_network, WMPred
        -   alpha_prediction : neural network for predicting alpha value for each feature given as input to SITH
        - dim_reduce_network : neural network for reducing dimensions of input
        - f_tilda_prediction : neural network to predict the labels from SITH outputs
        - WMPred : combines the above mentioned classes along with WM class to form G-SITH and to make predictions
    - reference.py : contains classes - RNN1FC, RNN2FC, LSTM1FC, LSTM2FC, GRU1FC, GRU2FC, LMU and coRNN
- losses
    - BCEWeighted : contains class for calculating weighted Binary Cross Entropy
    - MSE : Mean Squared Error
- Data_Generation
    -  Data_generator_v3.py : contains class for generating complex datasets.
- outputs/datasets
  - All the datasets used in the paper can be found in this directory in the data format mentioned above
- utils.py : contains functions for loading datasets and functions required for plotting final visualizations and results in the paper
- vis.py : visualization of results after training models

## Visualization and Results
- Use Tensorboard (tensorboard --logdir=runs) in the outputs folder to visualize the model training, validation plots on tensorboard 

To produce test loss values and plots of the test set predictions run - 
```
Example-

python vis.py --model_type "ALL" --train_size 3 --valid_size 12 --test_size 35 --num_inputs_sith 1 --num_extern_inputs_sith 0
```
Parameters for visualization - 
- model_type (str)
  - SITH: Globally modulated Scale-Invariant Temporal History (G-SITH) along with linear layers to make predictions
  - SITH_F: G-SITH without post inversion along with linear layers to make predictions
  - RNN1FC: Simple RNN and 1 linear layer
  - RNN2FC: Simple RNN and 2 linear layers
  - LSTM1FC: LSTM and 1 linear layer
  - LSTM2FC: LSTM and 2 linear layers
  - GRU1FC: GRU and 1 linear layer
  - GRU2FC: GRU and 2 linear layers
  - LMU : LMU (Legendre Memory Units) and 1 linear layer
  - coRNN : coRNN (Coupled Oscillatory Recurrent Neural Network) and 1 linear layer
  - ALL : all trained models in the output folder
- dataset_dir (str) : Path of the dataset to be used
- train_size (int) : size of training set
- valid_size (int) : size of validation set
- test_size (int) : size of test set
- n_taus (int) : number of taustar nodes in f_tilda
- tstr_min (float) : peak time of the first taustar node
- tstr_max (float) : peak time of the last taustar node
- k (int) : order of the derivative in the inverse laplace transform
- g (int) : (0 or 1) amplitude scaling of nodes in the inverse laplace transform output
- dt (float) : time step of the simulation
- num_inputs_sith (int) : number of inputs in the dataset
- num_extern_inputs_sith (int) : number of modulatory inputs in the dataset
- output_dir (str) : directory for storing tensorboard logs of training loss, validation loss and test loss (weighted and unweighted binary cross entropy loss) and model checkpoints.
- loss_type (str) : loss function for testing (mse or bce)
