from __future__ import division
from utils.utils import *
from skimage import measure as measure
from scipy import sparse
from scipy import signal
from scipy.signal import convolve2d as Convolve2d
from scipy.ndimage.filters import gaussian_filter as GaussianFilter
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
	
'''
This module contains the functions defined for Baseline methods!
'''
def GetEventIndexSet(x,y):
			return (x-1,x-1,x-1,x,x,x,x+1,x+1,x+1,x-2,x+2,x,x),(y-1,y,y+1,y-1,y,y+1,y-1,y,y+1,y,y,y-2,y+2)

def SparseRowColFilter( Mat, MaxRow, MaxCol ):
	Indx = np.where(Mat.row>=MaxRow)
	Mat.row = np.delete(Mat.row,Indx)
	Mat.col = np.delete(Mat.col,Indx)
	Mat.data = np.delete(Mat.data,Indx)

	Indx = np.where(Mat.col>=MaxCol)
	Mat.row = np.delete(Mat.row,Indx)
	Mat.col = np.delete(Mat.col,Indx)
	Mat.data = np.delete(Mat.data,Indx)

	Indx = np.where(Mat.row<0)
	Mat.row = np.delete(Mat.row,Indx)
	Mat.col = np.delete(Mat.col,Indx)
	Mat.data = np.delete(Mat.data,Indx)

	Indx = np.where(Mat.col<0)
	Mat.row = np.delete(Mat.row,Indx)
	Mat.col = np.delete(Mat.col,Indx)
	Mat.data = np.delete(Mat.data,Indx)
	return Mat

def getPOIThomee(sensor_data, misc=None):
	'''
	Reference:
	Bart Thomee, Ioannis Arapakis, and David A Shamma. 2016. Finding social points of interest from georeferenced and oriented online photographs.
ACM Trans. on Multimedia Computing, Communications, and Applications 12, 2 (2016), 36.
	These are BEST SETTINGS for location & orientation variance parameters as per our validation study
	Sigma = 0.0 
	SigmaPhi = 0.3
	'''
	Sigma = 0.0 
	SigmaPhi = 0.3
	x_, y_, phi_, Nc, T = sensor_data
	
	Density = np.zeros([ScoreSize,ScoreSize])
	DensityList = []
	X = np.empty(T)
	Y = np.empty(T)
	CamTemplet = np.radians(CamTempletAngle2)
	Weight = np.ones([ScoreSize,ScoreSize])

	def GetOrientationList(Sigma, SigmaPhi):
		FileName = './ThomeeOrientationListSigma'+str(Sigma)+'SigmaPhi'+str(SigmaPhi)+'.txt'
		if os.path.isfile(FileName):
			FileHandle = open(FileName,'r')	
			OrientationList = pickle.load(FileHandle)
			FileHandle.close()
			return OrientationList
		OrientationList = []
		CamTemplet = deepcopy(np.radians(CamTempletAngle2))
		for phi in range(0,360):
			print 'Phi:'+str(phi)
			Orientation = CamTemplet - np.radians(phi)
			Orientation = np.sqrt(2/np.pi)/SigmaPhi * np.exp(- (np.pi-np.abs(np.abs(Orientation)-np.pi))**2/(2*SigmaPhi**2))
			if Sigma == .0:
				Orientation = Orientation*CamTempletBoolean
			else:
				Orientation = GaussianFilter(Orientation,Sigma*5.5*10e-4)*CamTempletBoolean
			OrientationList.append(sparse.coo_matrix(Orientation))
		FileHandle = open(FileName,'w')
		pickle.dump(OrientationList,FileHandle)
		FileHandle.close()
		return OrientationList

	OrientationList = GetOrientationList(Sigma, SigmaPhi)
	for t in range(T):
		print 't:'+str(t)
		Density *= 0.0
		for i in range(Nc):
			Angle = np.int(np.round(np.degrees(phi_[t,i])))%360
			Weight = deepcopy(OrientationList[Angle])
			Weight.row = Weight.row - int(round(y_[t,i]))
			Weight.col = Weight.col + int(round(x_[t,i]))
			Weight = SparseRowColFilter(Weight, ScoreSize, ScoreSize)
			Density += Weight.toarray()
		Max = np.max(Density)
		indices = np.argwhere(Density==Max)
		r = np.mean(map(lambda x: x[0],indices))
		c = np.mean(map(lambda x: x[1],indices))
		EventIndexSet = GetEventIndexSet(int(round(r)),int(round(c)))
		Density[EventIndexSet] = Max-Density[EventIndexSet]

		r = r - ScoreSizeBy2
		c = c - ScoreSizeBy2
		X[t] = c
		Y[t] = -r
		DensityList.append(deepcopy(Density))
	DensityList = map(lambda Density: Density*250/np.max(Density),DensityList )
	return X,Y 
	#end getPOIThomee()

def getPOIThomeeAll(sensor_data, misc=None):
	'''
	Reference:
	Bart Thomee, Ioannis Arapakis, and David A Shamma. 2016. Finding social points of interest from georeferenced and oriented online photographs.
ACM Trans. on Multimedia Computing, Communications, and Applications 12, 2 (2016), 36.

	Range of Signma = 0.00, 0.50, 1.00, 2.50, 5.00 *10E-4
	Range of SigmaPhi = 0.00, 0.05, 0.10, 0.20, 0.30

	Range of Signma = 0.0-1.0 (*10E-4) for <=10 photos!
	Range of Signma = 2.5-5.0 (*10E-4) for >=20 photos!

	Range of SigmaPhi = 0.20, 0.30 for =5 photos!
	Range of SigmaPhi = 0.0-0.1 >=10 photos!
	'''
	x_, y_, phi_, Nc, T = sensor_data
	def GetOrientationList(Sigma, SigmaPhi):
		FileName = 'PickleFiles/ThomeeOrientationListSigma'+str(Sigma)+'SigmaPhi'+str(SigmaPhi)+'.txt'
		if os.path.isfile(FileName):
			FileHandle = open(FileName,'r')	
			OrientationList = pickle.load(FileHandle)
			FileHandle.close()
			return OrientationList
		OrientationList = []
		CamTemplet = deepcopy(np.radians(CamTempletAngle2))
		for phi in range(0,360):
			print 'Phi:'+str(phi)
			Orientation = CamTemplet - np.radians(phi)
			Orientation = np.sqrt(2/np.pi)/SigmaPhi * np.exp(- (np.pi-np.abs(np.abs(Orientation)-np.pi))**2/(2*SigmaPhi**2))
			if Sigma == .0:
				Orientation = Orientation*CamTempletBoolean
			else:
				Orientation = GaussianFilter(Orientation,Sigma*5.5*10e-4)*CamTempletBoolean
			OrientationList.append(sparse.coo_matrix(Orientation))
		FileHandle = open(FileName,'w')
		pickle.dump(OrientationList,FileHandle)
		FileHandle.close()
		return OrientationList

	def GetXY(OrientationList):
		Weight = np.ones([ScoreSize,ScoreSize])
		Density = np.zeros([ScoreSize,ScoreSize])
		X = np.empty(T)
		Y = np.empty(T)
		for t in range(T):
			print 't:'+str(t)
			Density *= 0.0
			for i in range(Nc):
				Angle = np.int(np.round(np.degrees(phi_[t,i])))%360
				Weight = deepcopy(OrientationList[Angle])
				Weight.row = Weight.row - int(round(y_[t,i]))
				Weight.col = Weight.col + int(round(x_[t,i]))
				Weight = SparseRowColFilter(Weight, ScoreSize, ScoreSize)
				Density += Weight.toarray()
			Max = np.max(Density)
			indices = np.argwhere(Density==Max)
			r = np.mean(map(lambda x: x[0],indices))
			c = np.mean(map(lambda x: x[1],indices))
			EventIndexSet = GetEventIndexSet(int(round(r)),int(round(c)))
			Density[EventIndexSet] = Max-Density[EventIndexSet]
			r = r - ScoreSizeBy2
			c = c - ScoreSizeBy2
			X[t] = c
			Y[t] = -r
		return X, Y

	Tracks = {}
	for Sigma in [0.00, 0.50, 1.00, 2.50, 5.00]:
		for SigmaPhi in [0.001, 0.05, 0.10, 0.20, 0.30]:
			print 'Sigma:'+str(Sigma)
			print 'SigmaPhi:'+str(SigmaPhi)
			Tracks['Sigma'+str(Sigma)+'SigmaPhi'+str(SigmaPhi)] = GetXY(GetOrientationList(Sigma, SigmaPhi))
	return Tracks 
	#end getPOIThomeeAll()

def getPOIHao1(sensor_data, misc=None):
	x_, y_, phi_, Nc, T = sensor_data
	X = np.zeros(T)
	Y = np.zeros(T)
	for t in range(T):
		XX=[]
		YY=[]
		for i in range(Nc):
			for j in range(i+1,Nc):
				phi = phi_[t,i]
				xi = x_[t,i]
				yi = y_[t,i]

				phj = phi_[t,j]
				xj = x_[t,j]
				yj = y_[t,j]

				x = ((yj-yi)+(xi/np.tan(phi)-xj/np.tan(phj)))/(1/np.tan(phi)-1/np.tan(phj))
				y = ((xj-xi)+(yi*np.tan(phi)-yj*np.tan(phj)))/(np.tan(phi)-np.tan(phj))

				#Check if points are too far from camera
				if np.linalg.norm([x-xi,y-yi])<=MaxVisibleDistance and np.linalg.norm([x-xj,y-yj])<=MaxVisibleDistance:
					#Check if this point is in front of the camera
					if np.sign(np.cos(phi)) == np.sign((y-yi)) and np.sign(np.cos(phj)) == np.sign((y-yj)):
						XX.append(x)
						YY.append(y)
		X[t] = X[t-1]  if not XX else np.mean(XX)
		Y[t] = Y[t-1] if not YY else np.mean(YY)

	FovAngle = np.radians(1.)
	DensityList = []
	Density = np.zeros([ScoreSize,ScoreSize])
	CamTemplet = np.radians(CamTempletAngle2)
	Orientation = np.zeros([ScoreSize, ScoreSize])
	OrientationList = []
	Weight = np.ones([ScoreSize,ScoreSize])
	for phi in range(0,360):
		print 'Phi:'+str(phi)
		Orientation = np.cos(CamTemplet - np.radians(phi))
		Orientation[Orientation>=np.cos(FovAngle)] = 1
		Orientation[Orientation<np.cos(FovAngle)] = 0
		Orientation = Orientation*CamTempletBoolean
		OrientationList.append(sparse.coo_matrix(Orientation))

	for t in range(T):
		Density *= 0.0
		for i in range(Nc):	
			Angle = np.int(np.round(np.degrees(phi_[t,i])))%360
			Weight = deepcopy(OrientationList[Angle])
			Weight.row = Weight.row - int(round(y_[t,i]))
			Weight.col = Weight.col + int(round(x_[t,i]))
			Weight = SparseRowColFilter(Weight, ScoreSize, ScoreSize)
			Density += Weight.toarray()
		Max = np.max(Density)
		c = int(round(X[t]+ScoreSizeBy2))
		r = int(round(-Y[t]+ScoreSizeBy2))
		EventIndexSet = GetEventIndexSet(int(round(r)),int(round(c)))
		Density[EventIndexSet] = Max-Density[EventIndexSet]
		DensityList.append(deepcopy(Density))
	DensityList = map(lambda Density: Density*250/np.max(Density),DensityList )
	return X,Y 
	#end getPOIHao1()

def getPOIHao2(sensor_data, misc=None): #camera field of view is modeled as a cone
	x_, y_, phi_, Nc, T = sensor_data
	FovAngle = np.radians(25.)
	Density = np.zeros([ScoreSize,ScoreSize])
	DensityList = []
	X = np.empty(T)
	Y = np.empty(T)
	CamTemplet = np.radians(CamTempletAngle2)
	Orientation = np.zeros([ScoreSize, ScoreSize])
	OrientationList = []
	Weight = np.ones([ScoreSize,ScoreSize])

	for phi in range(0,360):
		print 'Phi:'+str(phi)
		Orientation = np.cos(CamTemplet - np.radians(phi))
		Orientation[Orientation>=np.cos(FovAngle)] = 1
		Orientation[Orientation<np.cos(FovAngle)] = 0
		Orientation = Orientation*CamTempletBoolean
		OrientationList.append(sparse.coo_matrix(Orientation))
	for t in range(T):
		print 't:'+str(t)
		Density *= 0.0
		for i in range(Nc):	
			Angle = np.int(np.round(np.degrees(phi_[t,i])))%360
			Weight = deepcopy(OrientationList[Angle])
			Weight.row = Weight.row - int(round(y_[t,i]))
			Weight.col = Weight.col + int(round(x_[t,i]))
			Weight = SparseRowColFilter(Weight, ScoreSize, ScoreSize)
			Density += Weight.toarray()
		Max = np.max(Density)
		indices = np.argwhere(Density==Max)
		r = np.mean(map(lambda x: x[0],indices))
		c = np.mean(map(lambda x: x[1],indices))
		EventIndexSet = GetEventIndexSet(int(round(r)),int(round(c)))
		Density[EventIndexSet] = Max-Density[EventIndexSet]
		r = r - ScoreSizeBy2
		c = c - ScoreSizeBy2
		X[t] = c
		Y[t] = -r
		DensityList.append(deepcopy(Density))
	DensityList = map(lambda Density: Density*250/np.max(Density),DensityList )
	return X,Y 
	#end getPOIHao2() #Fov Cone

def getPOIHao3(sensor_data, misc=None): #camera field of view is modeled as a line
	x_, y_, phi_, Nc, T = sensor_data
	FovAngle = np.radians(1.)
	DensityList = []
	Density = np.zeros([ScoreSize,ScoreSize])
	X = np.empty(T)
	Y = np.empty(T)
	CamTemplet = np.radians(CamTempletAngle2)
	Orientation = np.zeros([ScoreSize, ScoreSize])
	OrientationList = []
	Weight = np.ones([ScoreSize,ScoreSize])
	for phi in range(0,360):
		print 'Phi:'+str(phi)
		Orientation = np.cos(CamTemplet - np.radians(phi))
		Orientation[Orientation>=np.cos(FovAngle)] = 1
		Orientation[Orientation<np.cos(FovAngle)] = 0
		Orientation = Orientation*CamTempletBoolean
		OrientationList.append(sparse.coo_matrix(Orientation))

	for t in range(T):
		print 't:'+str(t)
		Density *= 0.0
		for i in range(Nc):	
			Angle = np.int(np.round(np.degrees(phi_[t,i])))%360
			Weight = deepcopy(OrientationList[Angle])
			Weight.row = Weight.row - int(round(y_[t,i]))
			Weight.col = Weight.col + int(round(x_[t,i]))
			Weight = SparseRowColFilter(Weight, ScoreSize, ScoreSize)
			Density += Weight.toarray()
		Max = np.max(Density)
		indices = np.argwhere(Density==Max)
		r = np.mean(map(lambda x: x[0],indices))
		c = np.mean(map(lambda x: x[1],indices))
		EventIndexSet = GetEventIndexSet(int(round(r)),int(round(c)))
		Density[EventIndexSet] = Max-Density[EventIndexSet]
		r = r - ScoreSizeBy2
		c = c - ScoreSizeBy2
		X[t] = c
		Y[t] = -r
		DensityList.append(deepcopy(Density))
	DensityList = map(lambda Density: Density*250/np.max(Density),DensityList )
	return X,Y
	#end getPOIHao3() #Fov Line

def getPOIRobin(sensor_data, misc=None):
	x_, y_, phi_, Nc, T, V = sensor_data
	from cvxopt import matrix as cvxmatrix
	from cvxopt.solvers import lp
	Alpha = np.radians(10.)
	dm = MaxDistance
	DensityList = []
	Density = np.zeros([ScoreSize,ScoreSize])
	Weight = np.ones([Nc,ScoreSize,ScoreSize])
	X = np.empty(T)
	Y = np.empty(T)
	Xo = np.empty(T)
	Yo = np.empty(T)
	S = V**.5 #Standard Deviation!
	XCoordinates = CamTemplY
	YCoordinates = -CamTemplX

	def LineDistance(x1,y1,x2,y2,x,y):
		return (y2-y1)*(x-x1)-(y-y1)*(x2-x1)

	IndSet = []
	for t in range(T):
		print 't:'+str(t)
		Density *= 0.0
		Weight[:,:,:] = 1.0
		for i in range(Nc):
			phi = phi_[t,i]
			x = x_[t,i]
			y = y_[t,i]
			a = S[t,i]

			theta = np.pi/2-phi

			b = a + (dm+2*a)*np.tan(Alpha)

			xb = x + a*np.cos(np.pi + theta)
			yb = y + a*np.sin(np.pi + theta)

			xt = x + (a + dm)*np.cos(theta)
			yt = y + (a + dm)*np.sin(theta)

			x1 = xb + a*np.cos( np.pi/2 + theta)
			y1 = yb + a*np.sin( np.pi/2 + theta)
			x2 = xb + a*np.cos( 3*np.pi/2 + theta)
			y2 = yb + a*np.sin( 3*np.pi/2 + theta)
			x3 = xt + b*np.cos( 3*np.pi/2 + theta)
			y3 = yt + b*np.sin( 3*np.pi/2 + theta)
			x4 = xt + b*np.cos( np.pi/2 + theta)
			y4 = yt + b*np.sin( np.pi/2 + theta)
			#Line l1l4
			Weight[i][LineDistance(x1,y1,x4,y4,XCoordinates,YCoordinates)<0] = 0
			#Line l3l2
			Weight[i][LineDistance(x3,y3,x2,y2,XCoordinates,YCoordinates)<0] = 0
			#Line l2l1
			Weight[i][LineDistance(x2,y2,x1,y1,XCoordinates,YCoordinates)<0] = 0
			#Line l4l3
			Weight[i][LineDistance(x4,y4,x3,y3,XCoordinates,YCoordinates)<0] = 0
			Density += Weight[i]/np.sum(Weight[i])

		Max = np.max(Density)

		def GetCentroid(Properties):
			Areas = []
			for props in Properties:
				Areas.append(props.area)
			MaxIndx = np.argmax(np.array(Areas))
			return Properties[MaxIndx].centroid
				
		BoolDensity = Density*0.0
		BoolDensity[Density==Max] = 1.0
		LabelDensity = measure.label(BoolDensity,connectivity=1)
		Properties = measure.regionprops(LabelDensity)
		NoRegions=len(Properties)
		if NoRegions==1:
			row,col = Properties[0].centroid
		else:
			row,col = GetCentroid(Properties)
		r = row - ScoreSizeBy2
		c = col - ScoreSizeBy2
		X[t] = c
		Y[t] = -r
		Xo[t] = c
		Yo[t] = -r

		#Creating index set of cameras which will contribute for event location!
		IndSetEntry = []
		row = np.int(np.round(row))
		col = np.int(np.round(col))
		for i in range(Nc):
			if Weight[i][row,col] == 1.:
				IndSetEntry.append(i)
		IndSet.append(IndSetEntry)
		DensityList.append(deepcopy(Density))
	DensityList = map(lambda Density: Density*250/np.max(Density),DensityList )
		
	Gamma = 1.0
	tList = []
	for t in range(T):
		print 't:',t
		Ind = IndSet[t]
		N = len(Ind)
		P = cvxmatrix(0.0,(3*N,3*N+2))
		w = 1/S[t,Ind]
		W = np.diag(w)
		P[:N,2:2+N] = W
		P[N:2*N,2+N:2+2*N] = W
		P[2*N:,2+2*N:] = Gamma*np.eye(N,N)

		q = cvxmatrix(0.0,(3*N,1))
		q[:N] = w*x_[t,Ind]
		q[N:2*N] = w*y_[t,Ind]

		AA = cvxmatrix(0.0,(N,3*N+2))
		cos = np.cos(phi_[t,Ind])	
		sin = np.sin(phi_[t,Ind])
		AA[:,0] = -cos
		AA[:,1] = sin
		AA[:,2:2+N] = np.diag(cos)
		AA[:,2+N:2+2*N] = np.diag(-sin)
		AA[:,2+2*N:] = -np.eye(N,N)

		c = cvxmatrix(0.0,(6*N+2,1))
		c[3*N+2::] = np.ones(3*N)

		G = cvxmatrix(0.0,(6*N,6*N+2))
		G[:3*N,:3*N+2] = P
		G[3*N:,:3*N+2] = -P
		G[:3*N,3*N+2:] = -np.eye(3*N,3*N)
		G[3*N:,3*N+2:] = -np.eye(3*N,3*N)

		h = cvxmatrix(0.0,(6*N,1))
		h[:3*N] = q
		h[3*N:] = -q

		A = cvxmatrix(0.0,(N,6*N+2))
		A[:,:3*N+2] = AA
		b = cvxmatrix(0.0,(N,1))
		
		if N>1:
			Solution = lp(c,G,h,A,b)
			if not Solution['status'] == 'optimal':
				tList.append(t)
			Sol = Solution['x']
			X[t] = Sol[0]
			Y[t] = Sol[1]
			print 'X,Y:',X[t],Y[t]

	for t in range(T):
		print 't:',t
		c = int(round(X[t]+ScoreSizeBy2))
		r = int(round(-Y[t]+ScoreSizeBy2))
		print 'r,c:',r,c
		EventIndexSet = GetEventIndexSet(int(round(r)),int(round(c)))
		print 'EventIndexSet:',EventIndexSet
		DensityList[t][EventIndexSet] = np.max(DensityList[t])-DensityList[t][EventIndexSet]
	return X,Y
	#end getPOIRobin()

def dbscan(XX,YY):
	PLT = True
	PLT = False

	X = np.asarray(map(lambda x,y: [x,y],XX,YY))
	#db = DBSCAN(eps=0.3, min_samples=10).fit(X)
	db = DBSCAN(eps=20., min_samples=1).fit(X)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	print('Estimated number of clusters: %d' % n_clusters_)

	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	ClusterLen = 0
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Black used for noise.
			col = 'k'
		class_member_mask = (labels == k)
		xy = X[class_member_mask & core_samples_mask]
		if ClusterLen<len(xy):
			ClusterLen = len(xy)
			meanx = np.mean(XX[class_member_mask & core_samples_mask])
			meany = np.mean(YY[class_member_mask & core_samples_mask])
		if PLT == True:
			plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=16)
			xy = X[class_member_mask & ~core_samples_mask]
			plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
	if PLT == True:
		plt.plot(meanx, meany, '*', markerfacecolor='k', markeredgecolor='k', markersize=18)
		plt.plot(np.mean(XX),  np.mean(YY), '*', markerfacecolor='r', markeredgecolor='k', markersize=20)
		plt.title('Estimated number of clusters: %d' % n_clusters_)
		plt.show()
	if ClusterLen <=1.:
		meanx = np.mean(XX)
		meany = np.mean(YY)
	return meanx,meany






