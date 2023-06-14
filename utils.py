import argparse
import datetime
import functools
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict


from sklearn.preprocessing import MinMaxScaler

import numpy as np
import scipy.stats


# load dataset
def load_datasets(file: Path, n_train: int, n_valid: int, n_test: int):
	data = np.load(file)  
	x = torch.tensor(data["x"])
	x = x.type(torch.DoubleTensor)
	print("x shape - ", x.shape)
	#print(x.type())
	y = torch.tensor(data["y"])
	y = y.type(torch.DoubleTensor)
	print("y shape - ", y.shape)
	#print(y.type())
	
	logging.info(f"Loaded {len(x)} examples from '{file.resolve()}'")
	assert len(x) == n_train + n_valid + n_test

	dataset = TensorDataset(x, y)
	train_set, valid_set, test_set = random_split(dataset, [n_train, n_valid, n_test], generator=torch.Generator().manual_seed(42))
	logging.debug(f"train/valid/test = {len(train_set)}/{len(valid_set)}/{len(test_set)}")
	return train_set, valid_set, test_set



# plotting .95 confidence bounds on multiple runs
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    if n>1:
        se = scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        check = scipy.stats.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=scipy.stats.sem(data))
    else:
        h = 0
        check = 0     
    return m, h, check 


# returns tabular data from tfboard files
def tabulate_events(dpath, output_dir):
	# dpath - path of the files where tensorboard files are saved
	# add output directory to check if files with the same name already exist.
	Path(output_dir, "runs_pkl_files").mkdir(parents=True, exist_ok=True)
	save_directory = os.path.join(output_dir,"runs_pkl_files")

	files_already_converted = os.listdir(save_directory)

	#print("converted - ",files_already_converted)

	final_out = {}
	file_names = os.listdir(dpath)
	#print(file_names)
	for dname in file_names:
		
		if dname+".pkl" in files_already_converted:
			final_out[dname] = pd.read_pickle(os.path.join(save_directory, dname+".pkl"))  
			print(dname, " already converted")
			continue

		print(f"Converting run {dname}",end="")
		#print(os.path.join(dpath, dname)	)
		ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
		tags = ea.Tags()['scalars']

		out = {}

		for tag in tags:
			tag_values=[]
			#wall_time=[]
			steps=[]

			for event in ea.Scalars(tag):
				tag_values.append(event.value)
				#wall_time.append(event.wall_time)
				steps.append(event.step)

			#out[tag]=pd.DataFrame(data=dict(zip(steps,np.array([tag_values,wall_time]).transpose())), columns=steps,index=['value','wall_time'])
			out[tag]=pd.DataFrame(data=dict(zip(steps,tag_values)), columns=steps,index=['value'])

		if len(tags)>0:      
			df= pd.concat(out.values(),keys=out.keys())
			df.to_pickle(os.path.join(save_directory, dname+".pkl"))
			print("- Done")
		else:
			print('- Not scalers to write')

		final_out[dname] = df
		#print(df)


	return final_out


# collect loss values across mutiple runs and arranges them in dictionaries. This function returns 2 dictionaries - one for the loss values of various models across multiple learning rates for multiple runs, the other for their respective model file address in the directory
def collect_loss_values(row_heading, model_names, tab):

	# returns loss values for all models for all learning rates in a single dictionary

	test_best_model_dict = {}
	test_best_model_address_dict = {}

	for name in model_names:
		test_best_model_dict[name] = [[],[],[],[]] # because we have three learning rates
		test_best_model_address_dict[name] = [[],[],[],[]]

	for i in tab:
		row_headings = [k[0] for k in tab[i].index]
			
		for model_name in model_names:
			if model_name+"_lr" in i:
				if "lr0.1" in i:
					test_best_model_address_dict[model_name][0].append(i)

					if "loss/test_best_model" not in row_headings:
						test_best_model_dict[model_name][0].append(np.nan)
					else:
						test_best_model_dict[model_name][0].append(tab[i].loc[[row_heading]].to_numpy()[0][0])
				
				elif "lr0.01" in i:
					test_best_model_address_dict[model_name][1].append(i)

					if "loss/test_best_model" not in row_headings:
						test_best_model_dict[model_name][1].append(np.nan)
					else:
						test_best_model_dict[model_name][1].append(tab[i].loc[[row_heading]].to_numpy()[0][0])
				
				elif "lr0.001" in i:
					test_best_model_address_dict[model_name][2].append(i)

					if "loss/test_best_model" not in row_headings:
						test_best_model_dict[model_name][2].append(np.nan)
					else:
						test_best_model_dict[model_name][2].append(tab[i].loc[[row_heading]].to_numpy()[0][0])
				elif "lr1.0" in i:
					test_best_model_address_dict[model_name][3].append(i)

					if "loss/test_best_model" not in row_headings:
						test_best_model_dict[model_name][3].append(np.nan)
					else:
						test_best_model_dict[model_name][3].append(tab[i].loc[[row_heading]].to_numpy()[0][0])
				break
	return test_best_model_dict, test_best_model_address_dict


# Takes the two dictionaries with loss values and addresses across models-learning rates-runs and produces dictionaries for best model addresses and loss values across learning rates, runs and also returns the overall best model file name.
def collect_addresses_and_loss_for_LeastLoss_BestModel(test_best_model_dict, test_best_model_address):
	min_values_across_runs = {} # for each learning rate, stores the minimum value across runs
	min_addresses_across_runs = {} # for each learning rate, stores the address of run which produces minimum value

	min_values_across_lr = {} # for each model, stores the minimum value
	min_addresses_across_lr = {} # for each model, stores the address of the learning rate with minimum value

	file_name_of_best_model = {} # for each model points to the file name that has the least loss, if there is no file, then it stores None


	for i in test_best_model_dict:
		min_values_across_runs[i] = [np.nanmin(k) if len(k)!=0 and np.isnan(k).all()==False else np.nan for k in test_best_model_dict[i]]
		min_addresses_across_runs[i] = [np.nanargmin(k) if len(k)!=0 and np.isnan(k).all()==False else np.nan for k in test_best_model_dict[i]]

		#temp_vals_lr = [np.nanmin(k) if len(k)!=0 and np.isnan(k).all()==False else np.nan for k in test_best_model_dict[i]]
		#temp_addr_lr = [np.nanmin(k) if len(k)!=0 and np.isnan(k).all()==False else np.nan for k in test_best_model_dict[i]]

		if np.isnan(min_values_across_runs[i]).all()==False:
			min_values_across_lr[i] = np.nanmin(min_values_across_runs[i])
			min_addresses_across_lr[i] = np.nanargmin(min_values_across_runs[i])
		else:
			min_values_across_lr[i] = np.nan
			min_addresses_across_lr[i] = np.nan

	for i in test_best_model_dict:
		if np.isnan(min_addresses_across_lr[i]):
			file_name_of_best_model[i] = None
		else:
			file_name_of_best_model[i] =   test_best_model_address[i][min_addresses_across_lr[i]][min_addresses_across_runs[i][min_addresses_across_lr[i]]]	


	print("all loss values - ", test_best_model_dict)
	print("\n----------------------------------------\n")
	print("min_values_across_runs - ", min_values_across_runs)
	print("\n----------------------------------------\n")
	print("min_addresses_across_runs - ", min_addresses_across_runs)
	print("\n----------------------------------------\n")
	print("min_values_across_lr - ", min_values_across_lr)
	print("\n----------------------------------------\n")
	print("min_addresses_across_lr - ", min_addresses_across_lr)
	print("\n----------------------------------------\n")
	print("file address - ", file_name_of_best_model)
	print("\n----------------------------------------\n")



	return min_values_across_lr, min_addresses_across_lr, min_values_across_runs, min_addresses_across_runs, file_name_of_best_model