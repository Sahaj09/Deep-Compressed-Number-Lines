import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from models.reference import RNN1FC, RNN2FC, LSTM1FC, LSTM2FC, GRU1FC, GRU2FC, LegendreMemoryUnit, coRNN
from models.WMPred import WMPred, WMPred_with_bptt
from WM import WM
from utils import *
from losses import BCEWeighted

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split # check if this is needed in this file


def distance_highest_probab_metric(a, b):
	return np.mean(np.abs((np.argmax(a, axis=1) - np.argmax(b, axis=1))))


def test_vis(model, test_loader, loss_type):
	x, y = next(iter(test_loader))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	if loss_type=="bce":
		criterion = BCEWeighted(reduction="mean")
		bce = nn.BCEWithLogitsLoss(reduction="mean")
	elif loss_type=="mse":
		criterion = torch.nn.MSELoss()
		bce = torch.nn.MSELoss()

	model_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	model_total_params = sum(p.numel() for p in model.parameters())

	#print("total number of parameters in the model - ", model_total_params)
	#print("total number of trainable parameters in the model - ", model_total_params_trainable)

	model.eval()
	with torch.no_grad():
		x=x.to(device)
		y=y.to(device)
		#x=x.float()
		p, _ = model(x, None)
		test_loss = criterion(p, y)
		test_loss_unw = bce(p, y)

	"""
	print("test loss - ", test_loss)
	print("input shape - ", x.size())
	print("prediction shape - ", p.size())
	print("label shape - ", y.size())
	"""
	if loss_type=="bce":
		sigmoid = nn.Sigmoid()
		p = sigmoid(p)

	return test_loss, test_loss_unw, x.cpu().numpy(), p.cpu().numpy(), y.cpu().numpy(), model_total_params, model_total_params_trainable


def visualize_loss_plot(dict_values, path, file_name):
	epochs = [i for i in range(0,1000)]
	for i in dict_values:
		mean = []
		error = []

		for j in dict_values[i].T:

			#print(j)

			temp_mean, temp_error, temp_bounds = mean_confidence_interval(j)

			mean.append(temp_mean)
			error.append(temp_error)

		mean = np.array(mean)
		error = np.array(error)
		plt.plot(epochs, mean, label = i)
		plt.fill_between(epochs, (mean-error), (mean+error), alpha = 0.3)
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend()
	#plt.show()
	plt.savefig(os.path.join(path, file_name))
	plt.close()



def visualize(x, p, y, num_inputs_sith, num_extern_inputs_sith, output_directory_path, best_model_file_name):

	Path(output_directory_path, "visualization", best_model_file_name).mkdir(parents=True, exist_ok=True)

	path_to_dir = os.path.join(output_directory_path, "visualization", best_model_file_name)

	for i in range(0,len(x)):
		if num_extern_inputs_sith == None:
			plt.plot(x[i], label = "$f_a$")
		else:
			plt.plot(x[i][..., :num_inputs_sith], label = "$f_a$")

		plt.plot(p[i], '--', label = "prediction")
		plt.plot(y[i], label = "$f_b$")

		plt.xlabel('Time')
		plt.legend()
		temp = str(i) + "-input-prediction-" + "best_model_file_name" + ".png" # best_model_file_name

		plt.savefig(os.path.join(path_to_dir,temp))
		plt.close()


	
	if num_extern_inputs_sith != None and "simple" not in best_model_file_name: # if there is simple in best model file name don't plot external
		for i in range(0,len(x)):
			plt.plot(x[i][...,-num_extern_inputs_sith:])
			plt.xlabel('Time')
			plt.gca.legend_ =None
			temp = str(i) + "-extern-" + "best_model_file_name" + ".png"
			plt.savefig(os.path.join(path_to_dir,temp))
			plt.close()

	print("plots saved in - ", str(path_to_dir))








def main(model_type,
	dataset_dir,
	train_size,
	valid_size,
	test_size,
	n_taus,
	tstr_min,
	tstr_max,
	k,
	g,
	dt,
	order,
	theta,
	dt_cornn,
	gamma_cornn,
	epsilon_cornn,
	num_inputs_sith,
	num_extern_inputs_sith,
	output_dir,
	loss_type
	):

	curr_path = os.getcwd()

	torch.set_default_dtype(torch.float64)

	path_to_tb_files = os.path.join(curr_path, output_dir, "runs")
	path_to_best_model = os.path.join(curr_path, output_dir, "best_model_checkpoint")

	arr_tb_files = os.listdir(path_to_tb_files)
	arr_best_model = os.listdir(path_to_best_model)
	print(path_to_tb_files)

	if num_extern_inputs_sith == 0:
		num_extern_inputs_sith = None
	

	#--------------------------------------------------------------------------------------------------------------------------------------
	# LOADING LOSS VALUES AND FILE NAMES FROM TENSORBOARD FILES

	tab = tabulate_events(path_to_tb_files, output_dir)
	models = ["SITH_F", "SITH","RNN1FC","LSTM1FC","GRU1FC", "RNN2FC","LSTM2FC","GRU2FC", "SITH_BPTT", "SITH_F_BPTT", "CoRNN", "Linear_Scaling", "LMU"]

	"""
	"loss/train"
	"loss_unweighted/train"

	"loss/valid"
	"loss_unweighted/valid"
	"""
	#print(tab)
	#print(np.shape(tab['Jan25_15-40-05#simple_1#SITH_lr1.0_k8_ntaus50_L20_batchsize2_UID73718790'].loc[["loss/train"]].to_numpy()[0][1:1001]))  

	#print(tab)

	
	print("\n \n Unweighted--------------------------------------------------------------------------------------------------------------------\n \n") 

	unw_loss_test_best_model_lossVals, unw_loss_test_best_model_address = collect_loss_values('loss_unweighted/test_best_model', models, tab)
	print("---- test best loss values \n",unw_loss_test_best_model_lossVals,"\n ------------------")
	print("---- test best model address \n",unw_loss_test_best_model_address ,"\n ------------------")
	unw_loss_test_best_model_leastval_across_lr, unw_loss_test_best_model_leastvalIndex_across_lr, unw_loss_test_best_model_leastval_across_runs, unw_loss_test_best_model_leastvalIndex_across_runs, unw_loss_test_best_model_leastval_FileName = collect_addresses_and_loss_for_LeastLoss_BestModel(unw_loss_test_best_model_lossVals, unw_loss_test_best_model_address)

	print("Unweighted--")
	for name_model in models:
		print("model - ", name_model)
		if np.isnan(unw_loss_test_best_model_leastvalIndex_across_lr[name_model]):
			print("NA")
			continue
		lr = -1 - unw_loss_test_best_model_leastvalIndex_across_lr[name_model]
		lr = 10.0**lr
		print("learning rate - ",lr)
		#print(unw_loss_test_best_model_leastvalIndex_across_lr[name_model])
		print("values - ", unw_loss_test_best_model_lossVals[name_model][unw_loss_test_best_model_leastvalIndex_across_lr[name_model]])
		mean, h, check = mean_confidence_interval(unw_loss_test_best_model_lossVals[name_model][unw_loss_test_best_model_leastvalIndex_across_lr[name_model]])
		print("mean -", mean)
		print("h -", h)
		print("check -", check)

	"""
	unw_loss_test_best_model_address[model_name][unw_loss_test_best_model_leastvalIndex_across_lr[model_name]] # will give addresses for the models
	after collecting model - 
	distance = np.mean(np.abs((np.argmax(labels, axis=1) - np.argmax(outputs, axis=1)))
	"""
	print("\n \n Weighted-----------------------------------------------------------------------------------------------------------------------\n \n")
	loss_test_best_model_lossVals, loss_test_best_model_address = collect_loss_values('loss/test_best_model', models, tab)
	print("---- test best loss values \n",loss_test_best_model_lossVals,"\n ------------------")
	print("---- test best model address \n",loss_test_best_model_address ,"\n ------------------")
	loss_test_best_model_leastval_across_lr, loss_test_best_model_leastvalIndex_across_lr, loss_test_best_model_leastval_across_runs, loss_test_best_model_leastvalIndex_across_runs, loss_test_best_model_leastval_FileName = collect_addresses_and_loss_for_LeastLoss_BestModel(loss_test_best_model_lossVals, loss_test_best_model_address)

	print("Weighted--")
	for name_model in models:
		print("model - ", name_model)
		if np.isnan(loss_test_best_model_leastvalIndex_across_lr[name_model]):
			print("NA")
			continue
		lr = -1 - loss_test_best_model_leastvalIndex_across_lr[name_model]
		lr = 10.0**lr
		print("learning rate - ",lr)
		#print(unw_loss_test_best_model_leastvalIndex_across_lr[name_model])
		print("values - ", loss_test_best_model_lossVals[name_model][loss_test_best_model_leastvalIndex_across_lr[name_model]])
		mean, h, check = mean_confidence_interval(loss_test_best_model_lossVals[name_model][loss_test_best_model_leastvalIndex_across_lr[name_model]])
		print("mean -", mean)
		print("h -", h)
		print("check -", check)



	#--------------------------------------------------------------------------------------------------------------------------------------
	# LOADING DATA FOR TEST AND VALIDATION

	if model_type == "ALL":
		model_type_list = models
		flag_for_distance_calc = 1
	else:
		model_type_list = [model_type]
		flag_for_distance_calc = 0

	distance_loss_mean = {} # contains distance loss mean for every model
	distance_loss_confidence = {} # contains distance loss confidence for every model
	
	model_loss_train_weighted = {} # contains weighted training loss across epochs for 3 runs for each model
	model_loss_valid_weighted = {} # contains weighted validation loss across epochs for 3 runs for each model

	model_loss_train_unweighted = {} # contains uweighted training loss across epochs for 3 runs for each model
	model_loss_valid_unweighted = {} # contains unweighted validation loss across epochs for 3 runs for each model

	for model_type in model_type_list:

		print("\n ------ Visualizing results for ", model_type, " --------\n")

		if loss_test_best_model_leastval_FileName[model_type]==None:
			print("The specified model does not have any minimum loss values")
			continue

		dataset_dir_path = Path(dataset_dir)
		temp = loss_test_best_model_leastval_FileName[model_type].split("#")
		dataset_name = temp[1]+".npz"
		dataset_path = Path(os.path.join(dataset_dir_path,dataset_name))

		print("The dataset being loaded is - ", dataset_path)
		
		train_set, valid_set, test_set = load_datasets(dataset_path, train_size, valid_size, test_size)

		train_loader = DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=True)
		valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False, pin_memory=True)
		test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

		temp_x, temp_y = next(iter(test_loader))

		#print("shape -- ",temp_x.size())
		#shcbjqhbcqj
		#print(np.where(temp_x[:,:,0]==1))
		#plt.plot(temp_x[5,:,0])
		#plt.show()
		

		n_input = temp_x.size()[2]
		n_output = temp_y.size()[2]

		#------------------------------------------------------------------------------------------------------------------------------------
		# LOADING MODEL FOR TESTING AND VALIDATION

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
			"SITH_BPTT": WMPred_with_bptt,
			"SITH_F_BPTT": WMPred_with_bptt
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
			model = model_class(wm, n_inputs=num_inputs_sith, n_outputs = n_output, n_extern=num_extern_inputs_sith)
			if "simple" in dataset_name:
				model = model_class(wm, n_inputs=n_input, n_outputs = n_output, n_extern=None)
		
		else:
			raise NotImplementedError()

		#if ("simple" not in dataset_name) and (model_type in ("SITH")):
		#	print("num inputs - ", num_inputs_sith)
		#	print("num extern - ", num_extern_inputs_sith)
		#	print("num outputs - ", n_output)
		#else:
		#	print("num inputs - ", n_input)
		#	print("num outputs - ", n_output)

		# Calculating distance loss -

		if flag_for_distance_calc == 1:

			print("calculating dist metric for - ", unw_loss_test_best_model_address[model_type][unw_loss_test_best_model_leastvalIndex_across_lr[model_type]])

			temp_list_distance = []
			temp_loss_weighted = []
			temp_loss_unweighted = []

			temp_loss_weighted_train_epochs = []
			temp_loss_weighted_valid_epochs = []
			temp_loss_unweighted_train_epochs = []
			temp_loss_unweighted_valid_epochs = []
			for iter_ in range(0,len(unw_loss_test_best_model_address[model_type][unw_loss_test_best_model_leastvalIndex_across_lr[model_type]])):

				model_check = torch.load(os.path.join(path_to_best_model, unw_loss_test_best_model_address[model_type][unw_loss_test_best_model_leastvalIndex_across_lr[model_type]][iter_]), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
				model.load_state_dict(model_check['model_state_dict'])
				test_loss_b, test_loss_unw_b, input_samples, predictions, labels, _, _ = test_vis(model, test_loader, loss_type)
				distance_metric = distance_highest_probab_metric(predictions, labels)
				temp_list_distance.append(distance_metric)
				temp_loss_weighted.append(test_loss_b)
				temp_loss_unweighted.append(test_loss_unw_b)

				# file address unw_loss_test_best_model_address[model_type][unw_loss_test_best_model_leastvalIndex_across_lr[model_type]][iter_]

				temp_loss_weighted_train_epochs.append(tab[unw_loss_test_best_model_address[model_type][unw_loss_test_best_model_leastvalIndex_across_lr[model_type]][iter_]].loc[["loss/train"]].to_numpy()[0][1:1001])
				temp_loss_weighted_valid_epochs.append(tab[unw_loss_test_best_model_address[model_type][unw_loss_test_best_model_leastvalIndex_across_lr[model_type]][iter_]].loc[["loss/valid"]].to_numpy()[0][1:1001])

				temp_loss_unweighted_train_epochs.append(tab[unw_loss_test_best_model_address[model_type][unw_loss_test_best_model_leastvalIndex_across_lr[model_type]][iter_]].loc[["loss_unweighted/train"]].to_numpy()[0][1:1001])
				temp_loss_unweighted_valid_epochs.append(tab[unw_loss_test_best_model_address[model_type][unw_loss_test_best_model_leastvalIndex_across_lr[model_type]][iter_]].loc[["loss_unweighted/valid"]].to_numpy()[0][1:1001])
				"""
				"loss/train"
				"loss_unweighted/train"

				"loss/valid"
				"loss_unweighted/valid"
				"""
				#print(tab)
				#print(np.shape(tab['Jan25_15-40-05#simple_1#SITH_lr1.0_k8_ntaus50_L20_batchsize2_UID73718790'].loc[["loss/train"]].to_numpy()[0][1:1001]))

			model_loss_train_weighted[model_type] = np.array(temp_loss_weighted_train_epochs)
			model_loss_valid_weighted[model_type] = np.array(temp_loss_weighted_valid_epochs)

			model_loss_train_unweighted[model_type] = np.array(temp_loss_unweighted_train_epochs)
			model_loss_valid_unweighted[model_type] = np.array(temp_loss_unweighted_valid_epochs)

			print(np.shape(model_loss_valid_unweighted[model_type]))
			print(np.shape(model_loss_valid_weighted[model_type]))
			print(np.shape(model_loss_train_unweighted[model_type]))
			print(np.shape(model_loss_train_weighted[model_type]))


			print("distance list - ", temp_list_distance)
			print("loss weighted - ", temp_loss_weighted)
			print("loss unweighted - ", temp_loss_unweighted)
			mean_distance, h_distance, _ = mean_confidence_interval(temp_list_distance)

			distance_loss_mean[model_type] = mean_distance
			distance_loss_confidence[model_type] = h_distance





		print("the model path is - ", os.path.join(path_to_best_model, loss_test_best_model_leastval_FileName[model_type]))

		best_model_checkpoint = torch.load(os.path.join(path_to_best_model, loss_test_best_model_leastval_FileName[model_type]), map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
		model.load_state_dict(best_model_checkpoint['model_state_dict'])
		

		#test(model, test_loader, torch.nn.MSELoss(), torch.nn.MSELoss(), 0, kind = "test_after_training")  
		test_loss_b, test_loss_unw_b, input_samples, predictions, labels, model_total_params, model_total_params_trainable = test_vis(model, test_loader, loss_type)

		print("minimum test loss for ",model_type, " is ", test_loss_b)
		print("minimum test_unweighted loss for ",model_type, " is ", test_loss_unw_b)

		print("total number of parameters in the model - ", model_total_params)
		print("total number of trainable parameters in the model - ", model_total_params_trainable)

		visualize(input_samples, predictions, labels, num_inputs_sith, num_extern_inputs_sith, os.path.join(curr_path, output_dir), loss_test_best_model_leastval_FileName[model_type])

	print("distance_loss_mean - ",distance_loss_mean)
	print("distance_loss_confidence - ", distance_loss_confidence)

	#model_loss_train_weighted
	#model_loss_valid_weighted
	#model_loss_train_unweighted 
	#model_loss_valid_unweighted

	if model_loss_train_weighted:

		#for model_name in model_loss_train_weighted:
		visualize_loss_plot(model_loss_train_weighted, output_dir, "train_weighted.png")

		visualize_loss_plot(model_loss_valid_weighted, output_dir,"valid_weighted.png")

		visualize_loss_plot(model_loss_train_unweighted, output_dir, "train_unweighted.png")

		visualize_loss_plot(model_loss_valid_unweighted, output_dir, "valid_unweighted.png")




	





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
				"SITH_BPTT",
				"SITH_F_BPTT",
				"ALL"
			],
			help="type of model to visualize, enter all to visualize all models",
		)
		parser.add_argument("--dataset_dir", type=str, default="outputs/datasets", help="dataset directory")

		parser.add_argument("--train_size", type=int, help="number of examples in training split (None: len(train_set))")
		parser.add_argument("--valid_size", type=int, help="number of examples in validation split (None: len(train_set))")
		parser.add_argument("--test_size", type=int, help="number of examples in testing split (None: len(train_set))")
		
		# SITH arguments ----
		parser.add_argument("--n_taus", type=int, default=50, help="number of taustar nodes in the inverse Laplace transform")
		parser.add_argument("--tstr_min", type=float, default=0.005, help="peak time of the first taustar node")
		parser.add_argument("--tstr_max", type=float, default=20.0, help="peak time of the last taustar node")
		parser.add_argument("--k", type=int, default=8, help="order of the derivative in the inverse Laplace transform")
		parser.add_argument("--g", type=int, choices=[0, 1], default=1, help="amplitude scaling of nodes in til_f")
		parser.add_argument("--dt", type=float, default=0.001, help="time step of the simulation")

		#LMU arguments --
		#order, theta
		parser.add_argument("--order", type=int, default=128, help="order for LMU")
		parser.add_argument("--theta", type=float, default=5000, help="theta for LMU")

		# CoRNN arguments-------------
		# dt, gamma, epsilon
		parser.add_argument("--dt_cornn", type=float, default=1.6e-2, help="dt for CoRNN")
		parser.add_argument("--gamma_cornn", type=float, default=94.5, help="gamma for CoRNN")
		parser.add_argument("--epsilon_cornn", type=float, default=9.5, help="epsilon for CoRNN")

		parser.add_argument("--num_inputs_sith", type=int, default=1, help="number of inputs to sith")
		parser.add_argument("--num_extern_inputs_sith", type=int, default=2, help="number of inputs to calculate alpha")
		parser.add_argument("--output_dir", type=str, default="outputs", help="directory for output logs")
		
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
		args = parser.parse_args()
		return vars(args)

	main(**parse_args())