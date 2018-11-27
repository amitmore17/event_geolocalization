from utils.utils import *
from baselines.baseline import *
from baselines.cvx import getCvx
from baselines.pl import pgm
import pickle

'''
following are all possible methods to choose from:
Cvx, PoiT, PoiTAll, PoiH1, PoiH2, PoiH3, PoiR, Pgm, PgmV, PgmG, PgmGV
'''

def get_event(method, x_, y_, phi_, Nc, T, V, H, misc=None):
	sensor_data = x_, y_, phi_, Nc, T
	if method == 'Cvx':
		lambdas = [2., 5., 1., 0.0]
		sensor_data = x_, y_, phi_, Nc, T, V, H
		solution = getCvx(sensor_data,lambdas[0],lambdas[1],lambdas[2],lambdas[3])
		
	elif method == 'PoiT':
		solution = getPOIThomee(sensor_data, None)

	elif method == 'PoiTAll':
		solution = getPOIThomeeAll(sensor_data, None)

	elif method == 'PoiH1':
		solution = getPOIHao1(sensor_data, None)

	elif method == 'PoiH2':
		solution = getPOIHao2(sensor_data, None) 
	
	elif method == 'PoiH3':
		solution = getPOIHao3(sensor_data, None) 

	elif method == 'PoiR':
		sensor_data = x_, y_, phi_, Nc, T, V
		solution = getPOIRobin(sensor_data, None)
	
	elif method == 'Pgm':
		sensor_data = x_, y_, phi_, Nc, T, V, H
		solution = pgm(sensor_data, misc, VasVariable=False, Gaussian=False)
	
	elif method == 'PgmV':
		sensor_data = x_, y_, phi_, Nc, T, V, H
		solution = pgm(sensor_data, misc, VasVariable=True, Gaussian=False)
	
	elif method == 'PgmG':
		sensor_data = x_, y_, phi_, Nc, T, V, H
		solution = pgm(sensor_data, misc, VasVariable=False, Gaussian=True)
	
	elif method == 'PgmGV':
		sensor_data = x_, y_, phi_, Nc, T, V, H
		solution = pgm(sensor_data, misc, VasVariable=True, Gaussian=True)
		
	return solution

def plot_events(X, Y, name='./event.png', Xg=0, Yg=0, gt_available=False, ):
	plt.close('all')
	plt.plot(X,Y,'b',label='estimate')
	if gt_available:
		plt.plot(Xg,Yg,'g',label='ground_truth')
	plt.grid()
	plt.savefig(name)
	plt.close('all')
	
if __name__ == '__main__':
	solutions = {}
	method_list = 'PoiT', 'PoiH1', 'PoiH2', 'PoiH3', 'PoiR', 'Cvx', 'Pgm', 'PgmV'
	dataset_path = '../../dataset/'
	exp_number = 1
	exp_file_name = dataset_path+'exp_'+str(exp_number)+'.pk'
	file_object = open(exp_file_name,'rb') 
	exp_data = pickle.load(file_object)
	file_object.close()

	Nc = exp_data['number_of_cameras']
	T = exp_data['number_of_timestamps']
	gt_available = exp_data['is_event_track_gps_available']
	Xg, Yg, Err = exp_data['event_track']
	x_, y_, phi_, H, V = exp_data['sensor_data']
	cvx_sol = None
	
	for method in method_list:
		solution = get_event(method, x_, y_, phi_, Nc, T, V, H, misc=cvx_sol)
		if method == 'PoiTAll':
			raise ValueError('Not Implemented.')
		elif method == 'Cvx':
			X, Y = solution[0], solution[1]
			cvx_sol = solution
		elif method == 'PgmV':	#proposed
			X, Y = solution[0], solution[1]
		elif method == 'Pgm':		#proposed2
			X, Y = solution[0], solution[1]
		else:
			X, Y = solution
		plot_events(X, Y, './'+method+'.png', Xg, Yg, gt_available)






