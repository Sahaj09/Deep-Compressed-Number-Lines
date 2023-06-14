import numpy as np


class data_gen_V3:
  def __init__(self, num_stimulus, num_rewards, num_external_stimulus, total_length_of_trial, min_max_time_units_to_reward, external_stim_scaling_range, stim_reward_relation, scaling_relation, min_max_num_extern_stim_occurs_in_trial, min_max_range_for_extern_stim_to_stay, min_max_num_times_stim_occurs_in_trial, data_matrix= None,C_scale_matrix = None, steps_in_extern_stim = [0.5,1, 2, 3]):
    
    self.total_length_of_trial = total_length_of_trial
    self.num_rewards = num_rewards
    self.num_stimulus = num_stimulus
    self.num_external_stimulus = num_external_stimulus
    self.stim_reward_relation = stim_reward_relation
    self.min_max_num_extern_stim_occurs_in_trial = min_max_num_extern_stim_occurs_in_trial
    self.min_max_range_for_extern_stim_to_stay = min_max_range_for_extern_stim_to_stay
    self.min_max_num_times_stim_occurs_in_trial = min_max_num_times_stim_occurs_in_trial
    self.steps_in_extern_stim = steps_in_extern_stim

    #------------------------- Stim-reward relational matrix  
    if stim_reward_relation == 0 and data_matrix is None: # one-to-one mapping b/w stimulus and reward
      if num_stimulus != num_rewards:
        print("Warning, number of stimulus and rewards are not equal in order to form one-to-one relationship between each stimulus and reward")
      data_matrix = np.zeros((num_stimulus,num_rewards))
      temp_values = np.random.randint(min_max_time_units_to_reward[0], min_max_time_units_to_reward[1], num_stimulus)
      np.fill_diagonal(data_matrix, temp_values)  

    elif (stim_reward_relation == 1 or stim_reward_relation == 2) and data_matrix is None: # many-to-one (1) and many-to-many (2) mapping b/w stimulus and reward.
      data_matrix=[]
      for i in range(0,num_stimulus):
        if stim_reward_relation==2:
          temp = np.random.randint(1,num_rewards+1)
          temp_arr = np.zeros(num_rewards)
          for j in range(0,temp):
            temp_val = np.random.randint(0,num_rewards)
            temp_arr[temp_val] = np.random.randint(min_max_time_units_to_reward[0], min_max_time_units_to_reward[1])
        else:
          temp_arr = np.zeros(num_rewards)
          temp_val = np.random.randint(0,num_rewards)
          temp_arr[temp_val] = np.random.randint(min_max_time_units_to_reward[0], min_max_time_units_to_reward[1])
      
        data_matrix.append(temp_arr)
      data_matrix = np.array(data_matrix)
    #--------------------------------------External-stim relational matrix

    if C_scale_matrix is None and scaling_relation ==0: # one-one betwen external stimulus and stimulus
      if num_external_stimulus < num_stimulus:
        print("Warning, num of stimulus is greater than num of external stimulus, cannot form on-to-one relation between them for all stimulus, the last few will stimulus will be ignored")
      C_scale_matrix = []
      temp_matrix = data_matrix.copy()
      temp_matrix[temp_matrix>0]=1
      for i in range(0,num_external_stimulus):
        temp_1 = np.zeros((num_stimulus,num_rewards))
        temp_2 = -np.random.uniform(-external_stim_scaling_range[1], external_stim_scaling_range[0],(num_stimulus,num_rewards))
        index = i%num_stimulus
        temp_1[index] = temp_matrix[index]
        temp_1 = temp_1 * temp_2
        C_scale_matrix.append(temp_1)
      C_scale_matrix = np.array(C_scale_matrix)

    elif C_scale_matrix is None and (scaling_relation == 1 or scaling_relation == 2): # many-to-many and all-to-all
      C_scale_matrix = []
      for i in range(0,num_external_stimulus):
        if scaling_relation == 2:
          temp_arr = -np.random.uniform(-external_stim_scaling_range[1], external_stim_scaling_range[0], (num_stimulus, num_rewards))
        else:   
          temp = np.random.randint(1,(num_stimulus*num_rewards)+1)
          temp_arr = np.zeros((num_stimulus, num_rewards))
          for j in range(0,temp):
            temp_val_i = np.random.randint(0,num_stimulus)
            temp_val_j = np.random.randint(0,num_rewards)
            temp_arr[temp_val_i, temp_val_j] = -np.random.uniform(-external_stim_scaling_range[1], external_stim_scaling_range[0])
        C_scale_matrix.append(temp_arr)
      C_scale_matrix = np.array(C_scale_matrix)

    elif C_scale_matrix is None and scaling_relation == 3: 
      if num_stimulus != num_rewards:
        print("Warning, number of stimulus and rewards are not equal in order to form one-to-one relationship between each stimulus and reward, and one-to-one relationship between extern. stim. and reward")
      C_scale_matrix = np.zeros((num_external_stimulus,num_rewards))
      temp_values = -np.random.uniform(-external_stim_scaling_range[1], external_stim_scaling_range[0], num_external_stimulus)
      np.fill_diagonal(C_scale_matrix, temp_values)

    elif C_scale_matrix is None and scaling_relation == 4:
      C_scale_matrix= -np.random.uniform(-external_stim_scaling_range[1], external_stim_scaling_range[0], (num_external_stimulus, num_rewards))
    
    
    self.data_matrix = data_matrix
    self.C_scale_matrix = C_scale_matrix
    #self.C_scale_matrix[self.C_scale_matrix==0] = 1
    #self.C_scale_matrix = np.ceil(self.C_scale_matrix) # reasons to be told. 0.33*23=7.59, 0.33*24 =7.92, 0.33*25=8.25 (scaling factor* time to rewards), in all the cases the resulting time to reward is 8 units away.  

    if C_scale_matrix is not None:  # Can write the following lines in a cleaner way.
      self.num_external_stimulus = np.shape(C_scale_matrix)[0]
    if data_matrix is not None:
      self.num_stimulus = np.shape(data_matrix)[0]
      self.num_rewards = np.shape(data_matrix)[1]


  def generate_trial(self):
    
    num_extern_occurs = np.random.randint(self.min_max_num_extern_stim_occurs_in_trial[0], self.min_max_num_extern_stim_occurs_in_trial[1])
    num_stim_occurs = np.random.randint(self.min_max_num_times_stim_occurs_in_trial[0],self.min_max_num_times_stim_occurs_in_trial[1])

    
    C = np.ones((self.num_external_stimulus,self.total_length_of_trial))

    check_list = []

    for i in range(0,len(self.C_scale_matrix)):
      num_pivots = np.random.randint(self.min_max_range_for_extern_stim_to_stay[0],self.min_max_range_for_extern_stim_to_stay[1])
      time_of_pivots = np.random.choice(range(1,self.total_length_of_trial-1), num_pivots, replace=False) #np.random.randint(0,self.total_length_of_trial-duration)
      time_of_pivots = np.sort(time_of_pivots)

      step = np.random.choice(self.steps_in_extern_stim, num_pivots+1, replace=False)
      
      print("num pivots -", num_pivots)
      print("pivots timestep - ",time_of_pivots)
      print("values of extern - ",step)

      check_list.append([num_pivots, list(time_of_pivots), list(step)])

      for k in range(0,len(time_of_pivots)):
        if k==0:
          C[i][0:time_of_pivots[k]] =  step[k]
        else:
          C[i][temp_var:time_of_pivots[k]] = step[k]
        temp_var = time_of_pivots[k]
      
      C[i][temp_var:] = step[num_pivots]
    
    
    Stimulus = np.zeros((self.num_stimulus, self.total_length_of_trial))
    Rewards  = np.zeros((self.num_rewards, self.total_length_of_trial))
    
    count=0

    
    list_stim_occur = [[mm for mm in range(0,int(self.total_length_of_trial - np.max(self.data_matrix[kk])))] for kk in range(0,self.num_stimulus)] # all available empty time indices for every stimulus across a trial, to be used only by 1-1,many-1.
    

    while count<num_stim_occurs:
      stim = np.random.randint(0,self.num_stimulus) # Sampling stimulus  (Note for self:change this to random.choice() in the future and remove stim from choice in below condition for efficiency).
      
      if len(list_stim_occur[stim])==0: # If length of available empty time indices for a stim is 0. Then it checks if it is the same for other stimulus as well. Breaks the loop if it.
        temp = [len(mm) for mm in list_stim_occur]
        for i in temp:
          if i>0:
            continue
        print("Cannot sample any more stimulus because there are no empty time units left to choose for its occurence")
        break
      
      if self.stim_reward_relation == 0 or self.stim_reward_relation == 1:

        time_of_occurance = np.random.choice(list_stim_occur[stim]) # selects time randomly from available time indices for stimulus.
        list_stim_occur[stim].pop(list_stim_occur[stim].index(time_of_occurance)) # removes it from the set of all the empty indices for a stimulus.


        # LOOK FROM HERE--------------------------------------------------------------
        
        
        unscaled_time_to_reward = np.max(self.data_matrix[stim])
        index_of_reward = np.argmax(self.data_matrix[stim])

        c_time = time_of_occurance
        c_time_check = time_of_occurance + 1

        time_to_reward=0 

        while c_time_check <= time_of_occurance+unscaled_time_to_reward:
          
          c_time+=1
          time_to_reward+=1

          if time_of_occurance+time_to_reward >= self.total_length_of_trial:
            break
          
          new_data_matrix = np.ones(np.shape(self.data_matrix))
          m = np.zeros(np.shape(self.C_scale_matrix[0]))
          
          #if 1 in C[:,c_time]:
          for i in range(0,len(C[:,c_time])):
            m = m + (C[:,c_time][i]*self.C_scale_matrix[i])
          
          new_data_matrix = new_data_matrix*m

          scaling_factor = new_data_matrix[stim][index_of_reward]
          
          c_time_check+=scaling_factor
          

        if time_of_occurance+time_to_reward >= self.total_length_of_trial:
          continue
        
        if time_to_reward!=0:
          reward_i = np.argmax(self.data_matrix[stim])
          Rewards[reward_i][int(time_of_occurance+time_to_reward)] = 1
        
        Stimulus[stim][time_of_occurance] = 1
        check_list.append([time_of_occurance, time_of_occurance+time_to_reward])
        count+=1
      else: # This is if we need many to many relationship between stimulus and reward
        print("TBA, if required")
    return C, Stimulus, Rewards, check_list