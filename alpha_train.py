import argparse
import datetime
import functools
import logging
import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split # check if this is needed in this file
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


from losses import BCEWeighted
from models.reference import RNN1FC, RNN2FC, LSTM1FC, LSTM2FC, GRU1FC, GRU2FC, LegendreMemoryUnit, coRNN
from models.WMPred import WMPred, WMPred_with_bptt

from WM import WM
from utils import load_datasets
from matplotlib import pyplot as plt


# Train with truncated Backprop every k_1 steps.
def train_tbptt(model, train_loader, valid_loader, optim, criterion, criterion_unweighted, epochs, writer, model_checkpoints_path, k_1, disable_pbar=False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	pbar = trange(epochs, unit="ep", disable=disable_pbar)
	postfix = {}

	bce = criterion_unweighted # nn.BCEWithLogitsLoss(reduction="mean")

	iter_count = 0
	check_validation_loss = np.inf

	model_checkpoints_p, model_checkpoints_best_p = model_checkpoints_path

	model_checkpoint_file_name = os.path.basename(model_checkpoints_p)

	# Training

	for epoch in pbar:

		train_error_list = []
		train_error_list_unweighted = []

		for x_, y_ in train_loader:
			state=None # Holds the hidden state of the model
			train_loss=0
			unweighted_loss=0
			cnter=0 # counter for num of iterations
			for iter_num,(x,y) in enumerate(zip(x_.split(k_1, dim=1),y_.split(k_1, dim=1))):
				x, y = x.to(device), y.to(device)
				if state!=None:
					if "LSTM" in model_checkpoint_file_name or "CoRNN" in model_checkpoint_file_name or "LMU" in model_checkpoint_file_name:
						state_1,state_2 = state
						state_1 = state_1.detach()
						state_2 = state_2.detach()
						state = (state_1, state_2)
					else:
						state=state.detach()
				
				p, state = model(x, state)
				
				train_loss_iter = criterion(p, y)
				unweighted_loss_iter = bce(p, y)
				train_loss+=train_loss_iter
				unweighted_loss+=unweighted_loss_iter
				optim.zero_grad()
				train_loss_iter.backward()
				optim.step()
				cnter+=1

			postfix["train"] = train_loss.item()/cnter

			train_error_list.append(train_loss.item()/cnter)
			train_error_list_unweighted.append(unweighted_loss.item()/cnter)

			writer.add_scalar("loss/train_iterations", train_loss.item()/cnter, iter_count)
			writer.add_scalar("loss_unweighted/train_iterations", unweighted_loss.item()/cnter, iter_count)
			iter_count += 1

		# adding epoch loss values, weights and gradients to tensorboard
		writer.add_scalar("loss/train", np.mean(train_error_list), epoch)
		writer.add_scalar("loss_unweighted/train", np.mean(train_error_list_unweighted), epoch)

		"""

		for name_, weights in model.named_parameters():
			if weights.requires_grad:
				name = "weights/" + name_
				writer.add_histogram(name, weights, epoch)
				writer.add_histogram(f'{name}.grad', weights.grad, epoch)

		"""

		# Validation Loss Calculation, (note for self: make a function for this block)
		model.eval()
		with torch.no_grad():
			for x, y in valid_loader:
				x, y = x.to(device), y.to(device)
				p, _ = model(x, None)
				valid_loss = criterion(p, y)
				valid_loss_unw = bce(p, y)

			writer.add_scalar("loss/valid", valid_loss.item(), epoch)
			writer.add_scalar("loss_unweighted/valid", valid_loss_unw.item(), epoch)

			postfix["valid"] = valid_loss.item()
		model.train()

		# SAVING MODEL
		# Saving model twice during entire training - 2 checkpoints - middle and at finish
		if np.round(epochs / 2) == epoch or epochs - 1 == epoch:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optim.state_dict(),
				'unweighted_loss': np.mean(train_error_list_unweighted),
				'loss': np.mean(train_error_list),
				'valid_loss': valid_loss.item()
			}, model_checkpoints_p)

		# Saving model for the best validation loss
		if valid_loss.item() < check_validation_loss:
			check_validation_loss = valid_loss.item()
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optim.state_dict(),
				'unweighted_loss': np.mean(train_error_list_unweighted),
				'loss': np.mean(train_error_list),
				'valid_loss': valid_loss.item()
			}, model_checkpoints_best_p)
		pbar.set_postfix(**postfix)


# Train with backprop
def train(model, train_loader, valid_loader, optim, criterion, criterion_unweighted, epochs, writer, model_checkpoints_path, disable_pbar=False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	pbar = trange(epochs, unit="ep", disable=disable_pbar)
	postfix = {}

	bce = criterion_unweighted # nn.BCEWithLogitsLoss(reduction="mean")

	iter_count = 0
	check_validation_loss = np.inf

	model_checkpoints_p, model_checkpoints_best_p = model_checkpoints_path

	# Training

	for epoch in pbar:   
		
		train_error_list = []
		train_error_list_unweighted = []

		for x, y in train_loader:

			x, y = x.to(device), y.to(device)
			p, _ = model(x, None)
			train_loss = criterion(p, y)
			unweighted_loss = bce(p,y)
			postfix["train"] = train_loss.item()

			train_error_list.append(train_loss.item())
			train_error_list_unweighted.append(unweighted_loss.item())

			writer.add_scalar("loss/train_iterations", train_loss.item(), iter_count)
			writer.add_scalar("loss_unweighted/train_iterations", unweighted_loss.item(), iter_count)
			iter_count+=1

			optim.zero_grad()
			train_loss.backward()
			optim.step()

		
		# adding epoch loss values, weights and gradients to tensorboard
		writer.add_scalar("loss/train", np.mean(train_error_list), epoch)
		writer.add_scalar("loss_unweighted/train", np.mean(train_error_list_unweighted), epoch)

		"""
		for name_, weights in model.named_parameters():
			if weights.requires_grad:
				name = "weights/"+name_
				writer.add_histogram(name, weights, epoch)
				writer.add_histogram(f'{name}.grad',weights.grad, epoch)
		"""

		# Validation Loss Calculation
		model.eval()
		with torch.no_grad():
			for x, y in valid_loader:
				x, y = x.to(device), y.to(device)
				p, _ = model(x, None)
				valid_loss = criterion(p, y)
				valid_loss_unw = bce(p,y)

			writer.add_scalar("loss/valid", valid_loss.item(), epoch)
			writer.add_scalar("loss_unweighted/valid", valid_loss_unw.item(), epoch)
			
			postfix["valid"] = valid_loss.item()
		model.train()

		# SAVING MODEL
		# Saving model twice during entire training - 2 checkpoints - middle and at finish
		if np.round(epochs/2) == epoch or epochs-1==epoch:
			torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'unweighted_loss': np.mean(train_error_list_unweighted),
            'loss' : np.mean(train_error_list),
            'valid_loss' : valid_loss.item() 
            }, model_checkpoints_p)

		#Saving model for the best validation loss
		if valid_loss.item()<check_validation_loss:
			check_validation_loss = valid_loss.item()
			torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'unweighted_loss': np.mean(train_error_list_unweighted),
            'loss' : np.mean(train_error_list),
            'valid_loss' : valid_loss.item()
            }, model_checkpoints_best_p)

		pbar.set_postfix(**postfix)


# Testing the model performance on the test set
def test(model, test_loader, criterion_, criterion_unweighted, writer, kind = "test_after_training"):
	x, y = next(iter(test_loader))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device("cpu")
	model.to(device)

	x, y = x.to(device), y.to(device)

	bce = criterion_unweighted
	criterion = criterion_

	model.eval()
	with torch.no_grad():
		p, _ = model(x, None)
		test_loss = criterion(p, y)
		test_loss_unw = bce(p, y)

		writer.add_scalar(f"loss/{kind}", test_loss.item(), -1)
		writer.add_scalar(f"loss_unweighted/{kind}", test_loss_unw.item(), -1)

	return test_loss, test_loss_unw


def main(
	model_type,
	dataset,
	epochs,
	lr,
	batch_size,
	train_size,
	valid_size,
	test_size,
	n_taus,
	tstr_min,
	tstr_max,
	k,
	g,
	dt,
	l2_penalty,
	num_inputs_sith,
	num_extern_inputs_sith,
	order,
	theta,
	dt_cornn,
	gamma_cornn,
	epsilon_cornn,
	output_dir,
	bptt_type,
	k_1,
	loss_type,
	loglevel,
	no_pbar):
	
	
	if loglevel:
		logging.getLogger().setLevel(loglevel)

	torch.set_default_dtype(torch.float64)

	uniqueID=str(random.randint(10000000,99999999)) # UniqueID for the task.

	# torch.autograd.set_detect_anomaly(True)

	# Loading dataset into a data loader.

	dataset_path = Path(dataset)
	dataset_name = dataset_path.stem
	train_set, valid_set, test_set = load_datasets(dataset_path, train_size, valid_size, test_size)
	

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
	valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False, pin_memory=True)
	test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)



	# Assigning variables with size of features and labels
	temp_x, temp_y = next(iter(test_loader))

	n_input = temp_x.size()[2]
	n_output = temp_y.size()[2]

	# Naming output files for tensorboard files and checkpoint files 
	if model_type in ("SITH","SITH_BPTT"):
		hparams_str = f"{model_type}_lr{lr}_k{k}_ntaus{n_taus}_L2{l2_penalty}_batchsize{batch_size}_{bptt_type}_numInputs{num_inputs_sith}_numExtern{num_extern_inputs_sith}_UID{uniqueID}"
	elif model_type in ("LMU"):
		hparams_str = f"{model_type}_lr{lr}_order_{order}_theta_{theta}_L2{l2_penalty}_batchsize{batch_size}_{bptt_type}_{n_input}_UID{uniqueID}"
	elif model_type in ("CoRNN"):
		hparams_str = f"{model_type}_lr{lr}_dt_{dt_cornn}_gamma_{gamma_cornn}_epsilon_{epsilon_cornn}_L2{l2_penalty}_batchsize{batch_size}_{bptt_type}_{n_input}_UID{uniqueID}"
	else:
		hparams_str = f"{model_type}_lr{lr}_L2{l2_penalty}_batchsize{batch_size}_{bptt_type}_{n_input}_UID{uniqueID}"

	start_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
	log_dir = Path(output_dir, "runs", f"{start_time}#{dataset_name}#{hparams_str}")
	model_checkpoints_p = Path(output_dir, "model_checkpoints", f"{start_time}#{dataset_name}#{hparams_str}") # path for saving the model in the middle and the finish of training.
	model_checkpoints_best_p = Path(output_dir, "best_model_checkpoint", f"{start_time}#{dataset_name}#{hparams_str}") # path for saving the best model during training.
	model_checkpoints_path = [model_checkpoints_p, model_checkpoints_best_p]

	writer = SummaryWriter(log_dir=str(log_dir))

	Path(output_dir, "runs").mkdir(parents=True, exist_ok=True) # path for saving tf board files
	Path(output_dir, "model_checkpoints").mkdir(parents=True, exist_ok=True)
	Path(output_dir, "best_model_checkpoint").mkdir(parents=True, exist_ok=True)


	# Loading Model according to arguments passed to the code.
	model_map = {
		"SITH": WMPred,
		"SITH_F":WMPred,
		"RNN1FC": RNN1FC,
		"RNN2FC": RNN2FC,
		"LSTM1FC": LSTM1FC,
		"LSTM2FC": LSTM2FC,
		"GRU1FC": GRU1FC,
		"GRU2FC": GRU2FC,
		"LMU" : LegendreMemoryUnit,
		"CoRNN" : coRNN,
		"Linear_Scaling" : WMPred,
		"SITH_BPTT": WMPred_with_bptt, # This is not being used
		"SITH_F_BPTT": WMPred_with_bptt  # This is not being used
	}

	model_class = model_map[model_type]

	if model_type in ("SITH", "SITH_BPTT"):
		wm = WM(tstr_min=tstr_min, tstr_max=tstr_max, n_taus=n_taus, k=k, g=g, dt=dt, DEBUG_dt_scale=1, batch_first=True)
		model = model_class(wm, n_inputs=num_inputs_sith, n_outputs = n_output, n_extern=num_extern_inputs_sith)
		if "simple" in dataset_name:
			model = model_class(wm, n_inputs=n_input, n_outputs = n_output, n_extern=None)

	elif model_type in ("SITH_F", "SITH_F_BPTT"):
		wm = WM(tstr_min=tstr_min, tstr_max=tstr_max, n_taus=n_taus, k=k, g=g, dt=dt, DEBUG_dt_scale=1, batch_first=True)
		model = model_class(wm, n_inputs=num_inputs_sith, n_outputs = n_output, n_extern=num_extern_inputs_sith, use_F=True)
		if "simple" in dataset_name:
			model = model_class(wm, n_inputs=n_input, n_outputs = n_output, n_extern=None, use_F=True)
	
	elif model_type in ("RNN1FC", "LSTM1FC", "GRU1FC"):
		model = model_class(n_inputs=n_input, n_outputs=n_output, n_rnn_hidden=64)
	
	elif model_type in ("RNN2FC", "LSTM2FC", "GRU2FC"):
		model = model_class(n_inputs=n_input, n_outputs=n_output, n_rnn_hidden=64, n_fc_hidden=50)
	
	elif model_type in ("LMU"):
		model = model_class(input_dim=n_input, output_size=n_output, hidden_size=64, order = order, theta = theta) # try different hidden size

	elif model_type in ("CoRNN"): 
		model = model_class(n_inp = n_input, n_hid = 64, n_out = n_output, dt = dt_cornn, gamma = gamma_cornn, epsilon = epsilon_cornn)

	elif model_type in ("Linear_Scaling"):
		wm = WM(tstr_min=tstr_min, tstr_max=tstr_max, n_taus=n_taus, k=k, g=g, dt=dt, DEBUG_dt_scale=1, batch_first=True, linear_scaling_flag=True)
		model = model_class(wm, n_inputs=num_inputs_sith, n_outputs=n_output, n_extern=num_extern_inputs_sith)
		if "simple" in dataset_name:
			model = model_class(wm, n_inputs=n_input, n_outputs = n_output, n_extern=None)

	else:
		raise NotImplementedError()

	if ("simple" not in dataset_name) and (model_type in ("SITH")):
		print("num inputs - ", num_inputs_sith)
		print("num extern - ", num_extern_inputs_sith)
		print("num outputs - ", n_output)
	else:
		print("num inputs - ", n_input)
		print("num outputs - ", n_output)



	# setting hyperparameters for training
	if not batch_size:
		batch_size = len(train_set)

	optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)

	if loss_type=='bce':
		criterion = BCEWeighted(reduction="mean")
		criterion_unweighted = nn.BCEWithLogitsLoss(reduction="mean")
	elif loss_type=='mse':
		criterion = torch.nn.MSELoss()
		criterion_unweighted = torch.nn.MSELoss()

	# Training and testing the model
	if bptt_type=="bptt":
		train(model, train_loader, valid_loader, optim, criterion, criterion_unweighted, epochs, writer, model_checkpoints_path)
	else:
		train_tbptt(model, train_loader, valid_loader, optim, criterion, criterion_unweighted, epochs, writer, model_checkpoints_path, k_1, disable_pbar=no_pbar)  # LOOK AT THIS, START WORKING FROM HERE---------------------------------------

	test_loss_post_training, test_loss_unw_post_training = test(model, test_loader, criterion, criterion_unweighted, writer, "test_after_training")

	# Test the model when the validation loss was the least
	best_model_checkpoint = torch.load(model_checkpoints_best_p)
	model.load_state_dict(best_model_checkpoint['model_state_dict'])
	test_loss_b, test_loss_unw_b = test(model, test_loader, criterion, criterion_unweighted, writer, "test_best_model")

	# finishing touch
	print("model name - ", model_type)
	print("test loss weighted - ", test_loss_b)
	print("test loss unweighted -", test_loss_unw_b)
	logging.info(f"test post training loss: {test_loss_post_training} unweighted: {test_loss_unw_post_training} best loss: {test_loss_b} unweighted: {test_loss_unw_b} \n")
	writer.close()





if __name__ == "__main__":

	def parse_args():
		parser = argparse.ArgumentParser()
		parser.add_argument(
			"--model_type",
			type=str,
			choices=[
				"SITH",
				"SITH_F",
				"RNN1FC",
				"RNN2FC",
				"LSTM1FC",
				"LSTM2FC",
				"GRU1FC",
				"GRU2FC",
				"LMU",
				"CoRNN",
				"Linear_Scaling",
				"SITH_BPTT"
			],
			help="type of model to train",
		)
		parser.add_argument("--dataset", type=str, help="dataset to load for training")
		parser.add_argument("--train_size", type=int, help="number of examples in training split (None: len(train_set))")
		parser.add_argument("--valid_size", type=int, help="number of examples in validation split (None: len(train_set))")
		parser.add_argument("--test_size", type=int, help="number of examples in testing split (None: len(train_set))")
		parser.add_argument("--epochs", type=int, default=10000, help="number of epochs to train")
		parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
		parser.add_argument("--batch_size", type=int, help="number of examples in each batch (None: len(train_set))")
		
		# SITH/Linear_Scaling arguments--------------
		parser.add_argument("--n_taus", type=int, default=50, help="number of taustar nodes in the inverse Laplace transform")
		parser.add_argument("--tstr_min", type=float, default=0.005, help="peak time of the first taustar node")
		parser.add_argument("--tstr_max", type=float, default=20.0, help="peak time of the last taustar node")
		parser.add_argument("--k", type=int, default=8, help="order of the derivative in the inverse Laplace transform")
		parser.add_argument("--g", type=int, choices=[0, 1], default=1, help="amplitude scaling of nodes in til_f")
		parser.add_argument("--dt", type=float, default=0.001, help="time step of the simulation")
		parser.add_argument("--num_inputs_sith", type=int, default=1, help="number of inputs to sith")
		parser.add_argument("--num_extern_inputs_sith", type=int, default=2, help="number of inputs to calculate alpha")

		# LMU arguments---------------
		#order, theta
		parser.add_argument("--order", type=int, default=128, help="order for LMU")
		parser.add_argument("--theta", type=float, default=5000, help="theta for LMU")

		# CoRNN arguments-------------
		# dt, gamma, epsilon
		parser.add_argument("--dt_cornn", type=float, default=1.6e-2, help="dt for CoRNN")
		parser.add_argument("--gamma_cornn", type=float, default=94.5, help="gamma for CoRNN")
		parser.add_argument("--epsilon_cornn", type=float, default=9.5, help="epsilon for CoRNN")

		parser.add_argument("--l2_penalty", type=float, default=0, help="L2 regularization for model")
		parser.add_argument("--output_dir", type=str, default="outputs", help="directory for output logs")

		parser.add_argument(
			"--bptt_type",
			type=str,
			choices=[
				"bptt",
				"tbptt"
			],
			default="bptt",
			help="type of backprop for training the model, bptt - backpropogation through time, tbptt - truncated backpropogation through time",
		)
		parser.add_argument("--k_1", type=int, default = 5, help = "every k_1 steps backpropagate through k_2 steps in time, at the moment we did not add k_2, so k_1==k_2")

		parser.add_argument(
			"--loss_type",
			type=str,
			choices=[
				"bce",
				"mse"
			],
			default="bce",
			help="type of loss for calculation, if chosen mse, weighted and unweighted losses are both mse",
		)
		parser.add_argument("--no_pbar", action="store_true")
		group = parser.add_mutually_exclusive_group()
		group.add_argument("--debug", "-d", action="store_const", dest="loglevel", const=logging.DEBUG)
		group.add_argument("--verbose", "-v", action="store_const", dest="loglevel", const=logging.INFO)
		args = parser.parse_args()
		return vars(args)

	main(**parse_args())

"""

# Can add tbptt with k_1 and k_2, basically- every k_1 steps backpropogate taking k_2 steps back

def train_tbptt(model, train_loader, valid_loader, optim, criterion, epochs, writer, model_checkpoints_path, k_1, k_2, disable_pbar=False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	pbar = trange(epochs, unit="ep", disable=disable_pbar)
	postfix = {}

	bce = nn.BCEWithLogitsLoss(reduction="mean")

	iter_count = 0
	check_validation_loss = np.inf

	model_checkpoints_p, model_checkpoints_best_p = model_checkpoints_path

	retain_graph = k_1<k_2

	# Training
	model_checkpoint_file_name = os.path.basename(model_checkpoints_p)

	for epoch in pbar:
		if "SITH" in model_checkpoint_file_name:
			init_state = torch.zeros((train_loader.batch_size, model.n_inputs, model.sith.N), dtype=train_loader.dataset[0].dtype, device=device)
		elif "LSTM" in model_checkpoint_file_name:
			init_state = (torch.zeros((1, train_loader.batch_size, 64), dtype=train_loader.dataset[0].dtype, device=device), torch.zeros((1, train_loader.batch_size, model.n_rnn_hidden), dtype=train_loader.dataset[0].dtype, device=device)) # 1 is for num_layers, automate it later. And remove 64 and add model.n_rnn_hidden
		elif "GRU" in model_checkpoint_file_name or "RNN" in model_checkpoint_file_name:
			init_state = torch.zeros((1, train_loader.batch_size, 64), dtype=train_loader.dataset[0].dtype, device=device)

		states = [(None, init_state)]
		train_error_list = []
		train_error_list_unweighted = []
		for x, y in train_loader:

			x, y = x.to(device), y.to(device)

			for time_steps in range(0,x.size()[1]):
				state = states[-1][1].detach()
				state.requires_grad=True
				p, new_state = model(x[:,time_steps:time_steps+1], state)
				states.append((state, new_state))

				while len(states)>k_2:
					del states[0]

				if (time_steps+1)%k_1==0: # Fix things from here, code below this is wrong. Change this.

		pbar.set_postfix(**postfix)
#parser.add_argument("--k_2", type=int, default = 5, help = "every k_1 steps backpropagate taking k_2 back steps")
"""