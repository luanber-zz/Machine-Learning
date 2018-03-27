import numpy as np


file = open('ex1data1.txt', 'r');
ll = file.read();

data = ll.split('\n');

X= []
Y= []

for i in range(len(data)-1):

	if data[i].split(',') != '':
		M = data[i].split(',');
		X.append(float(M[0]));
		Y.append(float(M[1]));

print("\nX Matrix:\n\n", X);
print("\nY Matrix:\n\n", Y);

Xn = np.transpose( np.array([X]) );
y  = np.transpose( np.array([Y]) );

print("\nXn shape:\t", Xn.shape);
print("\ny shape:\t", y.shape);


one = np.ones( (Xn.shape[0], 1) );

print("\none shape:\t", one.shape);

x = np.concatenate((one, Xn), axis = 1);

print('\nx shape:\t', x.shape);
print('\nx matrix line sample:\n\n', x[2]);


def make_theta(y):
	return np.zeros((x.shape[1], 1));

theta = make_theta(y);

print('\ntheta shape:\t', theta.shape);

# Cost function

def Cost_function(X, y, h, theta):
	sum_dist = 0;
	for i in range(h.shape[0]) :
		sum_dist += (h[i] - y[i]) **2;

	J = sum_dist / float(2 * h.shape[0]);

	print("\nthe cost function returned:\t", J);
	return J;


#Cost_function(x, y, np.dot(x, theta), theta);

def Grad_Descent(X, y, alpha, ite):
	print("\nrunning gradient descent...\n");
	
	h = np.dot(x, theta);
	t_storage = np.zeros((1,1));

	def theta_storage(t, theta):
		return np.concatenate((t_storage, theta), axis = 0);

	for m in range(ite):
		j_t_0 = 0;
		j_t_1 = 0;
		for i in range(h.shape[0]) :
			j_t_0 += (h[i] - y[i])
			j_t_1 += (h[i] - y[i]) * x[i,1];

		j_t_0 /= float( h.shape[0]);
		j_t_1 /= float( h.shape[0]);

		theta[0,0] += alpha * j_t_0;			#evaluating theta 0
		theta[1,0] += alpha * j_t_1;			#evaluating theta 1

		t_storage = theta_storage(t_storage, theta);	#saving theta in a history matrix

	print('\ntheta_storage values:\n', t_storage);
	print("\nthe gradient descent found the values:\n");
	print("theta 0:\t", theta[0,0]);
	print("theta 1:\t", theta[1,0]);

	Cost_function(x, y, h, theta);

	return theta;


Grad_Descent(x, y, 0.0001, 1000);


print('\n');
