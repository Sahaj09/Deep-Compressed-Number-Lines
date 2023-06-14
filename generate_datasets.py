import argparse
import contextlib
import logging
from pathlib import Path

import numpy as np

from Data_Generation.Data_generator_v3 import data_gen_V3
from Data_Generation.Data_generator_NonLinear import data_gen_non_linear


def create_simple_data(num_examples: int, trial_len: int, interval: int):
    x = np.zeros((num_examples, trial_len, 1))
    y = np.zeros((num_examples, trial_len, 1))
    

    stim_end = trial_len - interval
    stim_step = int(stim_end / num_examples)

    for ex, stim_time in enumerate(range(0, stim_end, stim_step)):
        x[ex, stim_time, 0] = 1
        y[ex, stim_time + interval, 0] = 1

    return x, y, None 




def create_complex_data(num_examples: int, trial_len: int, interval: int, seed: int = 4):
    np.random.seed(seed)  # fix seed for data generation
    gen = data_gen_V3(
        num_stimulus=None,  # see data_matrix
        num_rewards=None,  # see data_matrix
        num_external_stimulus=None,  # see C_scale_matrix
        total_length_of_trial=trial_len,
        min_max_time_units_to_reward=None,
        external_stim_scaling_range=None,
        stim_reward_relation=0,
        scaling_relation=None,
        min_max_num_extern_stim_occurs_in_trial=[1, 2],  # 1
        min_max_range_for_extern_stim_to_stay=[9, 10],  # num_of_pivots  (9 times)
        min_max_num_times_stim_occurs_in_trial=[1, 2],  # 1 time
        data_matrix=np.full((1, 1), interval),
        C_scale_matrix=np.array([[0.25], [0.75]]),
        steps_in_extern_stim=[
            0,
            0.1,
            0.15,
            0.20,
            0.25,
            0.30,
            0.45,
            1,
            1.25,
            1.5,
            1.75,
            2,
            2.25,
            2.5,
            2.75,
            3,
            3.25,
            5,
        ],
    )

    x = []
    y = []
    z = []
    for i in range(num_examples):
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            ext_stim, stim, rewards, check = gen.generate_trial()
        else:
            with contextlib.redirect_stdout(None):  # silence prints
                ext_stim, stim, rewards, check = gen.generate_trial()

        x.append(stim)
        y.append(rewards)
        z.append(ext_stim)

    x = np.array(x).transpose((0, 2, 1))  # (batch, seq, feature)
    y = np.array(y).transpose((0, 2, 1))  # (batch, seq, feature)
    z = np.array(z).transpose((0, 2, 1))  # (batch, seq, feature)

    return x, y, z



def create_complex_data_multiple_modulatory_inputs(num_examples: int, trial_len: int, interval: int, seed: int = 4, mode: str = "linear"):
	# produces single stimulus with multiple modulatory signals

    np.random.seed(seed)  # fix seed for data generation

    values_for_mod_input_weights = [[0.25],[0.50],[0.75],[0.01],[0.035],[0.80]] + [[0] for i in range(0,44)]
    num_of_modulatory_inputs = 6  # this should be argument

    if mode == "non-linear":
    	# w1*c1 + (w1*c1)*(w2*c2) + .. (wi-1*ci-1)*(wi*ci) 
    	gen = data_gen_non_linear(
        num_stimulus=None,  # see data_matrix
        num_rewards=None,  # see data_matrix
        num_external_stimulus=None,  # see C_scale_matrix
        total_length_of_trial=trial_len,
        min_max_time_units_to_reward=None,
        external_stim_scaling_range=None,
        stim_reward_relation=0,
        scaling_relation=None,
        min_max_num_extern_stim_occurs_in_trial=[1, 2],  # 1
        min_max_range_for_extern_stim_to_stay=[9, 10],  # num_of_pivots  (9 times)
        min_max_num_times_stim_occurs_in_trial=[1, 2],  # 1 time
        data_matrix=np.full((1, 1), interval),
        C_scale_matrix= np.array(values_for_mod_input_weights), #np.array([[np.random.choice(values_for_mod_input_weights)] for i in range(0,num_of_modulatory_inputs)]),
        steps_in_extern_stim= [
            0,
            0.01,
            0.015,
            0.020,
            0.025,
            0.030,
            0.45,
            0.60,
            0.80,
            0.90,
            1,
            1.25,
            1.5,
            1.75,
            3.25,
            5,
        ]
    	)

    else:
    	gen = data_gen_V3(
        num_stimulus=None,  # see data_matrix
        num_rewards=None,  # see data_matrix
        num_external_stimulus=None,  # see C_scale_matrix
        total_length_of_trial=trial_len,
        min_max_time_units_to_reward=None,
        external_stim_scaling_range=None,
        stim_reward_relation=0,
        scaling_relation=None,
        min_max_num_extern_stim_occurs_in_trial=[1, 2],  # 1
        min_max_range_for_extern_stim_to_stay=[9, 10],  # num_of_pivots  (9 times)
        min_max_num_times_stim_occurs_in_trial=[1, 2],  # 1 time
        data_matrix=np.full((1, 1), interval),
        C_scale_matrix= np.array(values_for_mod_input_weights), #np.array([[np.random.choice(values_for_mod_input_weights)] for i in range(0,num_of_modulatory_inputs)]),
        steps_in_extern_stim= [
            0,
            0.01,
            0.015,
            0.020,
            0.025,
            0.030,
            0.45,
            0.60,
            0.80,
            0.90,
            1,
            1.25,
            1.5,
            1.75,
            3.25,
            5,
        ]
    	)

    x = []
    y = []
    z = []
    
    # best to use while here with checks for empty stims

    counter = 0 

    while counter<num_examples:
        print(counter)
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            ext_stim, stim, rewards, check = gen.generate_trial()
        else:
            with contextlib.redirect_stdout(None):  # silence prints
                ext_stim, stim, rewards, check = gen.generate_trial()

        if 1 not in np.array(stim):
        	continue
        x.append(stim)
        y.append(rewards)
        z.append(ext_stim)
        counter+=1

    x = np.array(x).transpose((0, 2, 1))  # (batch, seq, feature)
    y = np.array(y).transpose((0, 2, 1))  # (batch, seq, feature)
    z = np.array(z).transpose((0, 2, 1))  # (batch, seq, feature)

    

    return x, y, z

#def combine_multiple_modulatory_inputs() finish this

def create_combined_data(num_examples: int, trial_len: int, interval: int):
    assert num_examples % 2 == 0

    x1, y1, z1 = create_simple_data(num_examples // 2, trial_len, interval)
    x2, y2, z2 = create_complex_data(num_examples // 2, trial_len, interval)

    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    z = np.concatenate((z1, z2), axis=0)

    return x, y, z


def main(gen_type, scale_data, num_examples, dataset_dir, loglevel):
    if loglevel:
        logging.getLogger().setLevel(loglevel)

    trial_len = 200 * scale_data
    interval = 50 * scale_data

    x, y, z, = None, None, None
    if gen_type == "simple":
        x, y, _ = create_simple_data(num_examples, trial_len, interval)
    elif gen_type == "complex":
        x, y, z = create_complex_data(num_examples, trial_len, interval)
        x = np.concatenate((x,z), axis = 2)
    elif gen_type == "combined":
        x, y, z = create_combined_data(num_examples, trial_len, interval)
        x = np.concatenate((x,z), axis = 2)
    elif gen_type == "complex_multiple_modulation":
    	x, y, z = create_complex_data_multiple_modulatory_inputs(num_examples, trial_len, interval)
    	x = np.concatenate((x,z), axis = 2)
    elif gen_type == "complex_multiple_modulation_non_linear":
    	x, y, z = create_complex_data_multiple_modulatory_inputs(num_examples, trial_len, interval, mode = "non-linear")
    	x = np.concatenate((x,z), axis = 2)


    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset_dir, f"{gen_type}_{scale_data}.npz")

    if dataset_path.exists():
        logging.info(f"Skipped data generation - '{dataset_path.resolve()}' already exists.")
        return

    np.savez_compressed(dataset_path, x=x, y=y)

    logging.info(f"Saved {num_examples} examples to '{dataset_path.resolve()}'")



# add default trial length
# add default interval length

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--gen_type",
            type=str,
            choices=["simple", "complex", "combined", "complex_multiple_modulation", "complex_multiple_modulation_non_linear"],
            help="type of data to generate",
        )
        parser.add_argument(
            "--scale_data",
            type=int,
            default=1,
            help="scale of data",
        )
        parser.add_argument(
            "--num_examples",
            type=int,
            default=50,
            help="number of examples to generate",
        )
        parser.add_argument(  # remove this argument - dataset_directory to always be set to outputs/datasets
            "--dataset_dir",
            type=str,
            default="outputs/datasets",
            help="directory to save generated datasets",
        )
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--debug", '-d', action='store_const', dest="loglevel", const=logging.DEBUG)
        group.add_argument("--verbose", '-v', action='store_const', dest="loglevel", const=logging.INFO)
        args = parser.parse_args()
        return vars(args)


    main(**parse_args())
