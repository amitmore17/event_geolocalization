from __future__ import division
from utils.utils import *

def ParseSolution(Solution):
	Len = len(Solution)
	N = len(Solution[0])-1
	X = np.zeros(Len)
	Y = np.zeros(Len)
	x = np.zeros((Len,N))
	y = np.zeros((Len,N))
	phi = np.zeros((Len,N))
	t=0
	for sol in Solution:
		X[t] = sol[0][1]
		Y[t] = sol[0][2]
		i = 0
		for ssol in sol[1:]:
			x[t,i] = ssol[0]
			y[t,i] = ssol[1]
			phi[t,i] = np.radians(ssol[2]%360)
			i+=1
		t+=1
	return [ X, Y, x, y, phi ] #End of ParseSolution()
	
def getCvx(sensor_data,Lambda1=2.,Lambda2=5.,Lambda3=1.,Lambda4=0.0,Batch=1,OverLap=False,NumberOfIteration=1):
	'''
	This function returns Solution to the problem in following format
	It computes solutions iteratively
	Solution=
	[
		[	[Time X Y] [x1 y1 Ph1 d1] [x2 y2 Ph2 d2] ... [xN yN PhN dN] 	]
		[	[Time X Y] [x1 y1 Ph1 d1] [x2 y2 Ph2 d2] ... [xN yN PhN dN] 	]
				:
				:
				:
		[	[Time X Y] [x1 y1 Ph1 d1] [x2 y2 Ph2 d2] ... [xN yN PhN dN] 	]
	]
	Since its a iterative scheme, it has to rely on previous solutions. So we create dummy solution to start with, for the first interation
	'''
	x_, y_, phi_, N, T, V, H = sensor_data
	Number=0
	MiddlePoint = int(Batch/2.)+1
	#Creating zero solution, for compatibility with existing code
	Solution = []
	for t in range(T):
		Sol = []
		#Sol.append([0,0,0])
		Sol.append([0,np.random.uniform(),np.random.uniform()])
		for n in range(N):
			Sol.append([0,0,0,0])
		Solution.append(Sol)

	#Repeat this for NumberOfIteration number of times!
	while not NumberOfIteration == 0:
		if OverLap:
			pass
		else:#No overlap
			#First Loop
			NewSolution = GetConvergentPointForBatch(sensor_data,0,Batch,Solution[0:Batch],Solution[0],Solution[0],'First',Lambda1,Lambda2,Lambda3,Lambda4)
			Solution[0:Batch] = NewSolution[:]
			Number=Number+1
			#Intermediate Loops
			for t in range(Batch,T,Batch):
				print 't:'+str(t)
				tn = t+Batch
				if t == 0:
					PrevSol = Solution[0]
				else:
					PrevSol = Solution[t-1]
				if tn >= T:
					NextSol = Solution[T-1]
				else:
					NextSol = Solution[tn]
				NewSolution = GetConvergentPointForBatch(sensor_data,t,tn,Solution[t:tn],PrevSol,NextSol,'Intermediate',Lambda1,Lambda2,Lambda3,Lambda4)
				Number=Number+1
				Solution[t:tn] = NewSolution[:]

		NumberOfIteration = NumberOfIteration - 1

	#Here Solution is in xdi, ydi form
	#Converting it to phi, di form!
	for Sol in Solution:
		for n in range(N):
			if not Sol[n+1] == [None]:
				xdi = Sol[n+1][2]
				ydi = Sol[n+1][3]
				Sol[n+1][2] = np.degrees(-np.arctan2(ydi,xdi))+90
				Sol[n+1][3] = np.sqrt(xdi**2+ydi**2)

	X, Y, x, y, phi = ParseSolution(Solution)
	return X, Y, x, y, phi

def GetConvergentPointForBatch(sensor_data,t1,t2,OldSol,PrevSol,NextSol,New=None,Lambda1=None,Lambda2=None,Lambda3=None,Lambda4=None):
	x_, y_, phi_, N, T, V, H = sensor_data
	T = t2-t1
	TN = T*N
	DMax = MaxVisibleDistance
	DMin = MinVisibleDistance
	Cx = np.zeros([T,N])
	Cy = np.zeros([T,N])
	Err = np.zeros([T,N])
	Phi = np.zeros([T,N])
	MagneticField = np.zeros([T,N])

	L0 = np.ones([T,N])
	L1 = Lambda1*np.ones([T+1,1])
	L2 = Lambda2*np.ones([T+1,N])
	L3 = Lambda3*np.ones([T+1,N])
	L4 = Lambda4*np.ones([T,N])

	#Zeroing last entris of L1,2,3 !
	if New == 'First':
		#Ignoring prev solution for Fisrt Batch
		L1[0] = 0.0
		L2[0,:] = 0.0
		L3[0,:] = 0.0
		#Ignoring next solution for Fisrt Batch
		L1[T] = 0.0
		L2[T,:] = 0.0
		L3[T,:] = 0.0

	if New == 'Intermediate' or New == 'Last':
		#Ignoring next solution for all Batches
		L1[T] = 0.0
		L2[T,:] = 0.0
		L3[T,:] = 0.0
		
	Cx = x_[t1:t2,:]
	Cy = y_[t1:t2,:]
	Phi = np.degrees(phi_[t1:t2,:])
	Err = V[t1:t2,:]**.5/4.
	MagneticField = H[t1:t2,:]
				
	PhiError = (GetPhiError(MagneticField.flatten())).reshape(T,N)
	PhiMin = np.radians(Phi-PhiError)
	PhiMax = np.radians(Phi+PhiError)
	Phi = np.radians(Phi)

	def GetIniSolFromOldSol(OldSol):
		InitialSolution = cvxmatrix(0.0,(2*T+4*TN,1))
		for t in range(T):
			InitialSolution[t] = OldSol[t][0][1]								#X
			InitialSolution[T+t] = OldSol[t][0][2]								#Y
			for n in range(N):
				if not OldSol[t][n+1] == [None]:
					InitialSolution[2*T+t*N+n] = OldSol[t][n+1][0]			#xi
					InitialSolution[2*T+TN+t*N+n] = OldSol[t][n+1][1]			#yi
					InitialSolution[2*T+2*TN+t*N+n] = OldSol[t][n+1][2]		#xdi
					InitialSolution[2*T+3*TN+t*N+n] = OldSol[t][n+1][3]		#ydi
		
		return InitialSolution

	InitialSolution = GetIniSolFromOldSol(OldSol)

	#Extracting data from Prev/NextSol
	X0 = PrevSol[0][1]*L1[0]
	Y0 = PrevSol[0][2]*L1[0]

	x0 = np.zeros([N,1])
	y0 = np.zeros([N,1])
	xd0 = np.zeros([N,1])
	yd0 = np.zeros([N,1])

	X_1 = NextSol[0][1]*L1[T]
	Y_1 = NextSol[0][2]*L1[T]

	x_1 = np.zeros([N,1])
	y_1 = np.zeros([N,1])
	xd_1 = np.zeros([N,1])
	yd_1 = np.zeros([N,1])
	
	for n in range(N):
		x0[n] = PrevSol[n+1][0]*L2[0,n]
		y0[n] = PrevSol[n+1][1]*L2[0,n]
		xd0[n] = PrevSol[n+1][2]*L3[0,n]
		yd0[n] = PrevSol[n+1][3]*L3[0,n]

		x_1[n] = NextSol[n+1][0]*L2[T,n]
		y_1[n] = NextSol[n+1][1]*L2[T,n]
		xd_1[n] = NextSol[n+1][2]*L3[T,n]
		yd_1[n] = NextSol[n+1][3]*L3[T,n]
	
	#Creating cost function matrices!
	DX = np.zeros([TN,T])
	for t in range(T):
		DX[t*N:t*N+N,t] = L0[t,:]
	DX0 = np.zeros([TN,T])
	Dx = np.diag(-L0.flatten())
	Dx0 = np.zeros([TN,TN])
	F0 = np.bmat([[DX,DX0,Dx,Dx0,Dx,Dx0],[DX0,DX,Dx0,Dx,Dx0,Dx]])

	D1X = np.zeros([T+1,T])
	D1X[0:-1,0::] = np.diag(L1[0:-1].flatten())
	D1X[1::,0::] = D1X[1::,0::] - np.diag(L1[1::].flatten())
	D1X0 = np.zeros([T+1,T])
	ZeroT1x4T = np.zeros([T+1,4*TN])
	F1 = np.bmat([[D1X,D1X0,ZeroT1x4T],[D1X0,D1X,ZeroT1x4T]])

	b1 = np.zeros([2*T+2,1])
	b1[0] = -X0
	b1[T] = +X_1
	b1[T+1] = -Y0
	b1[2*T+1] = +Y_1

	ZeroTNNx2T = np.zeros([TN+N,2*T])
	D2X0 = np.zeros([TN+N,TN])
	D2X = np.zeros([TN+N,TN])
	D2X[0:-N,0::] = np.diag(L2[0:-1,:].flatten())
	D2X[N::,0::] = D2X[N::,0::] - np.diag(L2[1::,:].flatten())
	F2 = np.bmat([[ZeroTNNx2T,D2X,D2X0,D2X0,D2X0],[ZeroTNNx2T,D2X0,D2X,D2X0,D2X0]])

	b2 = np.zeros([2*(TN+N),1])
	b2[0:N] = -x0
	b2[TN:TN+N] = +x_1
	b2[TN+N:TN+2*N] = -y0
	b2[2*TN+N:2*TN+2*N] = +y_1

	D3X0 = np.zeros([TN+N,TN])
	D3X = np.zeros([TN+N,TN])
	D3X[0:-N,0::] = np.diag(L3[0:-1,:].flatten())
	D3X[N::,0::] = D3X[N::,0::] - np.diag(L3[1::,:].flatten())
	F3 = np.bmat([[ZeroTNNx2T,D3X0,D3X0,D3X,D3X0],[ZeroTNNx2T,D3X0,D3X0,D3X0,D3X]])

	b3 = np.zeros([2*(TN+N),1])
	b3[0:N] = -xd0
	b3[TN:TN+N] = +xd_1
	b3[TN+N:TN+2*N] = -yd0
	b3[2*TN+N:2*TN+2*N] = +yd_1

	ZeroTNx2T = np.zeros([TN,2*T])
	D4X0 = np.zeros([TN,TN])
	D4X = np.diag(L4.flatten())
	F4 = np.bmat([[ZeroTNx2T,D4X0,D4X0,D4X,D4X0],[ZeroTNx2T,D4X0,D4X0,D4X0,D4X]])

	ZerothTerm = 2*np.dot(F0.T,F0)
	FirstTerm = 2*np.dot(F1.T,F1)
	SecondTerm = 2*np.dot(F2.T,F2)
	ThirdTerm = 2*np.dot(F3.T,F3)
	#Negative of 4th term!
	ForthTerm = (-1)*2*np.dot(F4.T,F4)
	Hf = ZerothTerm + FirstTerm + SecondTerm + ThirdTerm + ForthTerm

	#Here we will remove constraints from phi if phierror is larger than 90 degrees
	PhiMask = deepcopy(PhiError.flatten())
	PhiErrorMax = 50
	PhiMask[PhiError.flatten()>PhiErrorMax]=0
	PhiMask[PhiError.flatten()<=PhiErrorMax]=1

	#1st Constraint
	h1 = cvxmatrix(0.0,(TN,1),tc='d')
	G1 = cvxmatrix(0.0,(TN,2*T+4*TN))
	G1[:,2*T+2*TN:2*T+3*TN] = -np.diag(PhiMask*np.cos(PhiMin).flatten())
	G1[:,2*T+3*TN::] = np.diag(PhiMask*np.sin(PhiMin).flatten())

	#2nd Constraint
	h2 = cvxmatrix(0.0,(TN,1),tc='d')
	G2 = cvxmatrix(0.0,(TN,2*T+4*TN))
	G2[:,2*T+2*TN:2*T+3*TN] = np.diag(PhiMask*np.cos(PhiMax).flatten())
	G2[:,2*T+3*TN::] = -np.diag(PhiMask*np.sin(PhiMax).flatten())

	#3rd Constraint
	h3 = cvxmatrix(-DMin*PhiMask,(TN,1),tc='d')
	G3 = cvxmatrix(0.0,(TN,2*T+4*TN))
	G3[:,2*T+2*TN:2*T+3*TN] = -np.diag(PhiMask*np.sin(Phi).flatten())
	G3[:,2*T+3*TN::] = -np.diag(PhiMask*np.cos(Phi).flatten())

	h = cvxmatrix([h1, h2, h3])
	G = cvxmatrix([G1, G2, G3])

	def Getf(X,n):
		'''
		Returns function & Constraints evaluated at X
		'''
		X = np.array(X)
	
		x = X[0:T]
		y = X[T:2*T]
		xi = X[2*T:2*T+TN].reshape(T,N)
		yi = X[2*T+TN:2*T+2*TN].reshape(T,N)
		xdi = X[2*T+2*TN:2*T+3*TN].reshape(T,N)
		ydi = X[2*T+3*TN::].reshape(T,N)

		if n == 0:
			'''
			Cost Function
			'''
			F0X=np.dot(F0,X)
			F1X=np.dot(F1,X) + b1
			F2X=np.dot(F2,X) + b2
			F3X=np.dot(F3,X) + b3
			F4X=np.dot(F4,X)

			ZerothTerm = np.dot(F0X.T,F0X)
			FirstTerm = np.dot(F1X.T,F1X)
			SecondTerm = np.dot(F2X.T,F2X)
			ThirdTerm = np.dot(F3X.T,F3X)
			#Negative of 4th term!
			ForthTerm = (-1)*np.dot(F4X.T,F4X)

			Cost = ZerothTerm + FirstTerm + SecondTerm + ThirdTerm + ForthTerm
			return Cost

		elif n<=TN:
			'''
			GPS Constraints
			'''
			n=n-1 #Since camera index starts with 0
			t=int(n/N)
			n=n%N
			Cost = (xi[t,n]-Cx[t,n])**2+(yi[t,n]-Cy[t,n])**2-Err[t,n]**2
			return Cost

		elif n<=2*TN:
			'''
			Compass Constraints
			'''
			n=n-1-TN
			t=int(n/N)
			n=n%N
			Cost = (xdi[t,n])**2+(ydi[t,n])**2-DMax**2
			return Cost
		else:
			raise ValueError('There are no more constraints!')

	def GetDf(X,n):
		'''
		Returns gradient of function and constraints as a column vector evaluated at X
		'''

		X = np.array(X)
		x = X[0:T]
		y = X[T:2*T]
		xi = X[2*T:2*T+TN].reshape(T,N)
		yi = X[2*T+TN:2*T+2*TN].reshape(T,N)
		xdi = X[2*T+2*TN:2*T+3*TN].reshape(T,N)
		ydi = X[2*T+3*TN::].reshape(T,N)

		if n==0:
			'''
			Cost Function
			'''
			F0X=np.dot(F0,X)
			F1X=np.dot(F1,X)
			F2X=np.dot(F2,X)
			F3X=np.dot(F3,X)
			F4X=np.dot(F4,X)

			ZerothTerm = 2*np.dot(F0.T,F0X)
			FirstTerm = 2*np.dot(F1.T,F1X+b1)
			SecondTerm = 2*np.dot(F2.T,F2X+b2)
			ThirdTerm = 2*np.dot(F3.T,F3X+b3)
			#Negative of 4th term!
			ForthTerm = (-1)*2*np.dot(F4.T,F4X)

			Df = ZerothTerm + FirstTerm + SecondTerm + ThirdTerm + ForthTerm
			return Df

		elif n<=TN:
			'''
			GPS Constraints
			'''
			n=n-1 #Since camera index starts with 0
			t=int(n/N)
			n=n%N
			Dfn = cvxmatrix(0.0,(2*T+4*TN,1))
			Dfn[2*T+(t*N+n)]=2*(xi[t,n]-Cx[t,n])
			Dfn[2*T+TN+(t*N+n)]=2*(yi[t,n]-Cy[t,n])
			return Dfn

		elif n<=2*TN:
			'''
			Compass Constraints
			'''
			n=n-1-TN
			t=int(n/N)
			n=n%N
			Dfn = cvxmatrix(0.0,(2*T+4*TN,1))
			Dfn[2*T+2*TN+(t*N+n)]=2*xdi[t,n]
			Dfn[2*T+3*TN+(t*N+n)]=2*ydi[t,n]
			return Dfn
			
		else:
			raise ValueError('There are no more constraints!')

	def GetH(X,n):
		'''
		Returns Hessian of a function & constraints
		'''
		H = cvxmatrix(0.0,(2*T+4*TN,2*T+4*TN))

		if n==0:
			'''
			Cost Function
			'''
			H=Hf
			
		elif n<=TN:
			'''
			GPS Constraints
			'''
			n=n-1 #Since camera index starts with 0
			t=int(n/N)
			n=n%N

			H[2*T+(t*N+n),2*T+(t*N+n)]=2
			H[2*T+TN+(t*N+n),2*T+TN+(t*N+n)]=2

		elif n<=2*TN:
			'''
			Compass Constraints
			'''
			n=n-1-TN
			t=int(n/N)
			n=n%N

			H[2*T+2*TN+(t*N+n),2*T+2*TN+(t*N+n)]=2
			H[2*T+3*TN+(t*N+n),2*T+3*TN+(t*N+n)]=2

		else:
			raise ValueError('There are no more constraints!')
		
		#Here we have computed Upper triangle of Hessian matrix
		#We need only lower traingular part of Hessian matrix
		return H.T

	def F(X=None, Z=None):
		if X is None:
			return 2*TN, InitialSolution

		if max(abs(X))>=ScoreSize:
			return None

		f = cvxmatrix(Getf(X,0))
		for n in range(1,2*TN+1):
			f = cvxmatrix([[f],[Getf(X,n)]])
		f = f.T #Column vector
	
		Df = cvxmatrix(GetDf(X,0))
		for n in range(1,2*TN+1):
			Df = cvxmatrix([[Df],[GetDf(X,n)]])
		Df = Df.T #Row vectors are gradients 

		if Z is None:
			return f, Df

		H = Z[0]*GetH(X,0)
		for n in range(1,2*TN+1):
			H = H+Z[n]*GetH(X,n)
		H=cvxmatrix(H)
		return f, Df, H

	solvers.options['maxiters']=100
	solvers.options['show_progress'] = False
	sol = np.array(cp(F,G,h)['x'])
	#Times = np.array(TimeLineIndicatorPart[:,0]).reshape(T,1)
	Times = np.arange(t1,t2).reshape(T,1)
	

	X = sol[0:T]
	Y = sol[T:2*T]
	x = sol[2*T:2*T+TN].reshape(T,N)
	y = sol[2*T+TN:2*T+2*TN].reshape(T,N)
	xd = sol[2*T+2*TN:2*T+3*TN].reshape(T,N)
	yd = sol[2*T+3*TN::].reshape(T,N)

	TXY = np.concatenate((Times,X,Y)).reshape(3,T).T
	NewSolution=[]
	for t in range(T):
		NewSol=[]
		NewSol.append(TXY[t])
		for n in range(N):
			NewSol.append([x[t,n],y[t,n],xd[t,n],yd[t,n]])
		NewSolution.append(NewSol)
	return NewSolution #end GetConvergentPointForBatch


