from matplotlib import pyplot as plt 
import numpy as np 


data = np.load("complex_multiple_modulation_non_linear_10.npz")
x= data["x"]
y=data["y"]

print(np.shape(x))
print(np.shape(y))


for i in range(0,100):
	#print(x[i][:,0])
	#print(y[i][:,0])
	ind = np.where(x[i][:,0])[0][0]
	ind_y = np.where(y[i])[0][0]
	print(ind_y-ind)

	if ind_y-ind==0:
		print("problem with index - ", i)
		break

for i in range(0,15):

	plt.plot(x[i][:,0])
	plt.plot(y[i])
	plt.show()

	plt.plot(x[i][:,1:])
	plt.show()