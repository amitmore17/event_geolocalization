from __future__ import division
from utils.utils import *
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.special import iv as i
from scipy.special import i0
from scipy.special import i1


'''
#Parameters to be estimated
Vc,Vd,Ve,Kh0,Kh1,Kc,Kd

#Posterior Expectations
EXX = np.zeros([T])
EX = np.zeros([T])
EYY = np.zeros([T])
EY = np.zeros([T])

Exx = np.zeros([T,Nc])
Ex = np.zeros([T,Nc])
Eyy = np.zeros([T,Nc])
Ey = np.zeros([T,Nc])

EXx = np.zeros([T,Nc])
EYy = np.zeros([T,Nc])

#Parameters for angles
Kh0,Kh1,Kh
Theta,K = Parameters of prior vM
Theta1,K1 = Parameters of posterior vM
'''
#VasVariable is True for proposed and False for proposed2

def V2M(Vec):
	return np.tile(Vec,(Nc,1)).transpose()
	
def I(k,p):
	if k<700:
		if p==0:
			return i0(k)
		elif p==1:
			return i1(k)
		else:
			return i(p,k)
	else:
		print 'k:'+str(k)
		raise ValueError('Bessel function is very high for this k. Evaluate log of bessel function instead!')

def LogI(K,p):
	def logI(k,p):
		if k<700:
			return np.log(I(k,p))
		else:
			return -0.5*np.log(2*np.pi*k)+k+np.log( 1 - (4*p*p-1)/(8*k) + (4*p*p-1)*(4*p*p-9)/(128*k*k) - (4*p*p-1)*(4*p*p-9)*(4*p*p-25)/(3072*k*k*k) ) #Refer A.4 from Directional Statistics by Mardia
	if hasattr(K,'__iter__'):
		#return np.asarray(map(lambda k: logI(k,p),K))
		return np.asarray(map(lambda k: map(lambda kk: logI(kk,p),k) if hasattr(k,'__iter__') else logI(k,p),K))
	else:
		return logI(K,p)

def A(K):
	def a(k):
		if k<700:
			return i1(k)/i0(k)
		else:
			k=k*1.0
			a1 = 1-1/(2*k)-1/(8*k*k)-1/(8*k*k*k) #Refer 3.5.34 from Directional Statistics by Mardia
			#a2 = np.exp(LogI(k,1)-LogI(k,0))
			#print 'a1:'+str(a1)
			#print 'a2:'+str(a2)
			#print 'diff:'+str(abs(a1-a2))
			return a1
	if hasattr(K,'__iter__'):
		#return np.asarray(map(lambda k: a(k),K))
		return np.asarray(map(lambda k: map(lambda kk: a(kk), k) if hasattr(k,'__iter__') else a(k),K))
	else:
		return a(K)

def A1(k):
	#A11 = ( I(k,2)*i0(k) - i1(k)**2 ) / i0(k)**2
	A12 = A(k)*(np.exp(LogI(k,2)-LogI(k,1))-A(k))
	#print 'A11:'+str(A11)
	#print 'A12:'+str(A12)
	#print 'Diff:'+str(abs(A11-A12))
	return A12
	
def GetInitialSolution(cvx_sol, Phi_, Nc, T,x_,y_,H):
	X, Y, x, y, Phi = cvx_sol
	PsiE=np.zeros([T,Nc])#Must be in radiance!
	d=np.zeros([T,Nc])
	for t in range(T):
		for i in range(Nc):
			d[t,i]= abs( np.cos(Phi[t,i])*(x[t,i]-X[t])+np.sin(Phi[t,i])*(Y[t]-y[t,i]) )
			PsiE[t,i] = arctan2( X[t]-x[t,i],Y[t]-y[t,i] )%(2*np.pi)

	M, m = GetLocationMeans( X, x, Nc, T )
	N, n = GetLocationMeans( Y, y, Nc, T )
	Psi = GetOrientationMeans( X, Y, x, y, Phi, Nc, T )

	Vc = .5*( np.mean((x-m)**2) + np.mean((y-n)**2))
	Vd = np.mean(d**2)
	Ve = .5*( np.mean((X-M)**2) + np.mean((Y-N)**2))
	Vo = .5*( np.mean((x-x_)**2) + np.mean((y-y_)**2))

	def GetK(Phi,Theta):
		DelPhi = Phi-Theta
		SBar = np.mean(np.sin(DelPhi))
		CBar = np.mean(np.cos(DelPhi))
		ThetaBar = np.arctan2(SBar,CBar)
		R = (CBar**2+SBar**2)**.5
		R = R*np.cos(ThetaBar)
		if R<.53:
			k = 2*R+R**3+5*(R**5)/6
		if .53<=R and R<.85:
			k = -.4+1.39*R+.43/(1-R)
		else:
			k = 1/(2.*(1-R))
		return k

	Kc = GetK(Phi.flatten(),Psi.flatten())
	Kd = GetK(Phi.flatten(),PsiE.flatten())
	def GetKh1Kh0(phi,phi_,h):
		k = np.array(map(lambda theta,theta_:GetK(theta,theta_),phi,phi_))
		A = np.ones((T*Nc,2))
		A[:,1] = -(h-42.)**2
		b = np.log(k)
		X = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.transpose(),A)),A.transpose()),b)
		Kh1 = np.exp(X[0])
		Kh0 = 1/X[1]
		return Kh1, Kh0
	
	Kh1, Kh0 = GetKh1Kh0(Phi.flatten(),Phi_.flatten(),H.flatten())
	print 'Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0'
	print str([Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0])
	return Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, X, Y, x, y, Phi#End of GetInitialSolution()

def GetInitialSolution2(cvx_sol, Phi_, Nc, T,x_,y_,H):
	Vc, Vd, Ve, Vo = abs(np.random.rand(4))
	Kc, Kd, Kh1, Kh0 = abs(np.random.rand(4))*10.
	print 'Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0'
	print str([Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0])
	X, Y = GetEventLocations(x_,y_,Phi_)
	return Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, X, Y, x_, y_, Phi_#End of GetInitialSolution2()

def GetEventLocations(x, y, Phi):
	'''
	Unconstrained event estimate!
	'''
	SPhi = np.sin(Phi)
	CPhi = np.cos(Phi)
	def GetSingleEventLocation(xt, yt, SPhit, CPhit):
		a = np.sum( CPhit**2 )
		b = -np.sum( SPhit*CPhit )
		c = np.sum( -xt*CPhit**2 + yt*SPhit*CPhit )

		p = -np.sum( SPhit*CPhit )
		q = np.sum( SPhit**2 )
		r = np.sum( xt*SPhit*CPhit - yt*SPhit**2 )

		Xt = (b*r-c*q)/(a*q-b*p)
		Yt = (c*p-a*r)/(a*q-b*p)
		return [Xt,Yt]
	XY = np.asarray( map(lambda xt, yt, SPhit, CPhit:GetSingleEventLocation(xt,yt,SPhit,CPhit), x, y, SPhi, CPhi) )
	X = XY[:,0]
	Y = XY[:,1]
	#D = np.abs(CPhi*(x-V2M(X))+SPhi*(V2M(Y)-y))
	return X,Y #End of GetEventLocations()

def ParseData(VideoList,TimeLineIndicator,GroundTruthAvailable):
	global Nc
	global T
	Nc = len(VideoList)
	if GroundTruthAvailable:
		Nc -=1
	
	T = len(TimeLineIndicator)
	x_=np.zeros([T,Nc])
	y_=np.zeros([T,Nc])
	Phi_=np.zeros([T,Nc])#Must be in radiance!
	H=np.zeros([T,Nc])
	V=np.zeros([T,Nc])
	for t in range(T):
		for n in range(Nc):
			Indx = TimeLineIndicator[t][n+1]
			x_[t,n] = VideoList[n].sensor.TYXYE[Indx][2]
			y_[t,n] = VideoList[n].sensor.TYXYE[Indx][3]
			Phi_[t,n] = np.radians( VideoList[n].sensor.TYXYE[Indx][1] )%(2*np.pi)
			H[t,n] = VideoList[n].sensor.MagneticData[Indx][3]
			V[t,n] = ( VideoList[n].sensor.TYXYE[Indx][4] )**2 
	return x_, y_, Phi_, H, V, Nc, T #End of ParseData()

#Various parameters like markove chain means.
def GetAverage(X,Length = None):
	if Length == None:
		Length = 1
	MeanX=deepcopy(X)
	T = len(X)
	for i in range(Length):
		MeanX[i] =  np.mean(X[:i+Length+1])
	for i in range(Length,T-Length):
		MeanX[i] =  np.mean(X[i-Length:i+Length+1])
	for i in range(T-Length,T):
		MeanX[i] =  np.mean(X[i-Length:])
	return MeanX

def GetPeriodicAverage(X,Length = None):
	tao = 2*np.pi
	if Length == None:
		Length = 1
	MeanX=deepcopy(X)
	T = len(X)
	for i in range(Length):
		MeanX[i] = ( arctan2( np.sum(np.sin(X[:i+Length+1])), np.sum(np.cos(X[:i+Length+1])) ) )%tao
	for i in range(Length,T-Length):
		MeanX[i] = ( arctan2( np.sum(np.sin(X[i-Length:i+Length+1])), np.sum(np.cos(X[i-Length:i+Length+1])) ) )%tao
	for i in range(T-Length,T):
		MeanX[i] = ( arctan2( np.sum(np.sin(X[i-Length:])), np.sum(np.cos(X[i-Length:])) ) )%tao
	return MeanX

def GetMean(X):
	'''Xt = (Xt-1+Xt+1)/2.'''
	MeanX=deepcopy(X)
	T = len(X)
	for i in range(1,T-1):
		MeanX[i] =  0.5*(X[i-1]+X[i+1])
	MeanX[[0,-1]] = X[[1,-2]]
	return MeanX

def GetMeanPrime(X):
	'''Xt = 2Xt+1-Xt+2'''
	MeanX=deepcopy(X)
	T = len(X)
	for i in range(T-2):
		MeanX[i] =  2*X[i+1]-X[i+2]
	return MeanX

def GetMeanPrimePrime(X):
	'''Xt = 2Xt-1-Xt-2'''
	MeanX=deepcopy(X)
	T = len(X)
	for i in range(2,T):
		MeanX[i] =  2*X[i-1]-X[i-2]
	return MeanX

def GetPeriodicMean(X):
	'''Xt = (Xt-1+Xt+1)/2.'''
	tao = 2*np.pi
	MeanX=deepcopy(X)
	T = len(X)
	for i in range(1,T-1):
		MeanX[i] = ( arctan2( np.sin(X[i-1])+np.sin(X[i+1]), np.cos(X[i-1])+np.cos(X[i+1]) ) )%tao
	MeanX[[0,-1]] = X[[1,-2]]
	return MeanX

def GetPeriodicMeanPrime(X):
	'''Xt = 2Xt+1-Xt+2'''
	tao = 2*np.pi
	MeanX=deepcopy(X)
	T = len(X)
	for i in range(T-2):
		MeanX[i] = ( arctan2( 2*np.sin(X[i+1])-np.sin(X[i+2]), 2*np.cos(X[i+1])-np.cos(X[i+2]) ) )%tao
	return MeanX

def GetPeriodicMeanPrimePrime(X):
	'''Xt = 2Xt-1-Xt-2'''
	tao = 2*np.pi
	MeanX=deepcopy(X)
	T = len(X)
	for i in range(2,T):
		MeanX[i] = ( arctan2( 2*np.sin(X[i-1])-np.sin(X[i-2]), 2*np.cos(X[i-1])-np.cos(X[i-2]) ) )%tao
	return MeanX

def GetLocationMeans( X, x, Nc, T, FiltLen = None):
	if FiltLen == None:
		FiltLen = 3
	#M = GetAverage(X,int((FiltLen-1)/2.))
	M = GetMean(X)
	m=np.zeros([T,Nc])
	for i in range(Nc):
		#m[:,i] = GetAverage(x[:,i],int((FiltLen-1)/2.))
		m[:,i] = GetMean(x[:,i])
	return M,m #End GetLocationMeans()

def GetLocationMeansPrime( X, x, Nc, T, FiltLen = None):
	M1 = GetMeanPrime(X)
	m1=np.zeros([T,Nc])
	for i in range(Nc):
		m1[:,i] = GetMeanPrime(x[:,i])
	return M1,m1 #End GetLocationMeansPrime()

def GetLocationMeansPrimePrime( X, x, Nc, T, FiltLen = None):
	M11 = GetMeanPrimePrime(X)
	m11=np.zeros([T,Nc])
	for i in range(Nc):
		m11[:,i] = GetMeanPrimePrime(x[:,i])
	return M11,m11 #End GetLocationMeansPrimePrime()

def GetOrientationMeans( X, Y, x, y, Phi, Nc, T, FiltLen = None ):
	tao = 2*np.pi
	if FiltLen == None:
		FiltLen = 3
	PsiM=np.zeros([T,Nc])	#This is a variable Psi in the derivation
	for i in range(Nc):
		#PsiM[:,i] = GetPeriodicAverage(Phi[:,i],int((FiltLen-1)/2.))
		PsiM[:,i] = GetPeriodicMean(Phi[:,i])
	return PsiM%tao #End GetOrientationMeans()

def GetOrientationMeansPrime( X, Y, x, y, Phi, Nc, T, FiltLen = None ):
	tao = 2*np.pi
	PsiM1=np.zeros([T,Nc])	#This is a variable Psi in the derivation
	for i in range(Nc):
		PsiM1[:,i] = GetPeriodicMeanPrime(Phi[:,i])
	return PsiM1%tao #End GetOrientationMeansPrime()

def GetOrientationMeansPrimePrime( X, Y, x, y, Phi, Nc, T, FiltLen = None ):
	tao = 2*np.pi
	PsiM11=np.zeros([T,Nc])	#This is a variable Psi in the derivation
	for i in range(Nc):
		PsiM11[:,i] = GetPeriodicMeanPrimePrime(Phi[:,i])
	return PsiM11%tao #End GetOrientationMeansPrimePrime()

def GetVM(Theta1, K1, Theta2, K2, Gaussian):
	if Gaussian:
		K = K1+K2
		Theta = (Theta1*K1+Theta2*K2)/K
		return Theta, K
		
	Del = Theta1-Theta2
	K = (K1**2+K2**2+2*K1*K2*np.cos(Del))**0.5
	Theta = Theta1 + np.arctan2(-np.sin(Del),K1/K2+np.cos(Del))
	return Theta, K #end of GetVM()

#Posterior Expectations
def GetPosterierX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model ):
	Vc +=Vmin
	Vd +=Vmin
	Ve +=Vmin
	Vo +=Vmin
	#Adding all common terms for t=0...T-1
	ax = np.sum(Alphax**2,1)/(-2*Vd)
	bx = 1/(-2*Vo) + Alphax**2/(-2*Vd)
	cx = Alphax**2/Vd
	dx = np.sum(Alphax*Betax,1)/Vd 
	ex = x_/Vo + Alphax*Betax/(-Vd)
	#Adding M terms for t=1...T-2
	ax[1:-1] += 1/(-2*Ve)
	bx[1:-1] += 1/(-2*Vc)
	dx[1:-1] += M[1:-1]/Ve
	ex[1:-1] += m[1:-1]/Vc
	if Model=='1stOrder':
		print 'Model:'+str(Model)
		ax[[0,-1]] += 1/(-4*Ve)
		bx[[0,-1]] += 1/(-4*Vc)
		dx[[0,-1]] += M[[0,-1]]/(2*Ve)
		ex[[0,-1]] += m[[0,-1]]/(2*Vc)
	else:	#Model=='2ndOrder'
		#Adding MPrime terms for t=0...T-3
		ax[:-2] += 1/(-8*Ve)
		bx[:-2] += 1/(-8*Vc)
		dx[:-2] += M1[:-2]/(4*Ve)
		ex[:-2] += m1[:-2]/(4*Vc)
		#Adding MPrimePrime terms for t=2...T-1
		ax[2:] += 1/(-8*Ve)
		bx[2:] += 1/(-8*Vc)
		dx[2:] += M11[2:]/(4*Ve)
		ex[2:] += m11[2:]/(4*Vc)

	MuXt = ( np.sum(cx*ex/(2*bx),1) - dx )/( 2*ax-np.sum(cx*cx/( 2*bx ),1) )
	VXt = 1/( np.sum(cx*cx/(2*bx),1)-2*ax )
	Vxt = 1/(-2*bx)
	
	EXX = MuXt**2 + VXt
	EX = MuXt
	Exx = (cx**2 * V2M(EXX) + 2*cx*ex*V2M(MuXt)+ex**2)/(4*bx**2) + Vxt
	Ex = ( cx*V2M(MuXt) + ex )/(-2*bx)
	EXx = ( cx*V2M(EXX)+ex*V2M(MuXt) )/(-2*bx)
	#return EXX, EX, Exx, Ex, EXx, VXt, Vxt #End GetPosterierX()
	return EXX, EX, Exx, Ex, EXx #End GetPosterierX()

def GetPosterierY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model ):
	Vc +=Vmin
	Vd +=Vmin
	Ve +=Vmin
	Vo +=Vmin
	#Adding all common terms for t=0...T-1
	ay = np.sum(Alphay**2,1)/(-2*Vd)
	by = 1/(-2*Vo) + Alphay**2/(-2*Vd)
	cy = Alphay**2/Vd
	dy = np.sum(Alphay*Betay,1)/(-Vd)
	ey = y_/Vo + Alphay*Betay/Vd
	#Adding M terms for t=1...T-2
	ay[1:-1] += 1/(-2*Ve)
	by[1:-1] += 1/(-2*Vc)
	dy[1:-1] += N[1:-1]/Ve
	ey[1:-1] += n[1:-1]/Vc
	if Model=='1stOrder':
		ay[[0,-1]] += 1/(-4*Ve)
		by[[0,-1]] += 1/(-4*Vc)
		dy[[0,-1]] += N[[0,-1]]/(2*Ve)
		ey[[0,-1]] += n[[0,-1]]/(2*Vc)
	else:	#Model=='2ndOrder'
		#Adding MPrime terms for t=0...T-3
		ay[:-2] += 1/(-8*Ve)
		by[:-2] += 1/(-8*Vc)
		dy[:-2] += N1[:-2]/(4*Ve)
		ey[:-2] += n1[:-2]/(4*Vc)
		#Adding MPrimePrime terms for t=2...T-1
		ay[2:] += 1/(-8*Ve)
		by[2:] += 1/(-8*Vc)
		dy[2:] += N11[2:]/(4*Ve)
		ey[2:] += n11[2:]/(4*Vc)

	MuYt = ( np.sum(cy*ey/(2*by),1)-dy )/( 2*ay-np.sum(cy*cy/( 2*by ),1) )
	VYt = 1/( np.sum(cy*cy/(2*by),1)-2*ay )
	Vyt = 1/(-2*by)
	
	EYY = MuYt**2 + VYt
	EY = MuYt
	Eyy = (cy**2 * V2M(EYY) + 2*cy*ey*V2M(MuYt)+ey**2)/(4*by**2) + Vyt
	Ey = ( cy*V2M(MuYt) + ey )/(-2*by)
	EYy = ( cy*V2M(EYY)+ey*V2M(MuYt) )/(-2*by)
	#return EYY, EY, Eyy, Ey, EYy, VYt, Vyt #End GetPosterierY()
	return EYY, EY, Eyy, Ey, EYy #End GetPosterierY()

#def GetPosterierPhi( Kh0, Kh1, Ke, Km, PsiE, PsiM, Phi_, H ):
def GetPosterierPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model, Gaussian ):
	Kh = Kh1#*np.exp( -H42Sq/Kh0 )

	Kc = Kc/(Kc/Kmax+1)
	Kd = Kd/(Kd/Kmax+1)
	Kh = Kh/(Kh/Kmax+1)

	#Common terms for t=0...T-1
	Theta, K = GetVM(Phi_, Kh, PsiE, Kd, Gaussian)
	if Gaussian:
		K = np.tile(K,Theta.shape)
		
	#Adding M terms for t=1...T-2
	Theta[1:-1], K[1:-1] = GetVM(Theta[1:-1], K[1:-1], Psi[1:-1], Kc, Gaussian)
	if Model=='1stOrder':
		#Adding M terms for t=1...T-2
		Theta[[0,-1]], K[[0,-1]] = GetVM(Theta[[0,-1]], K[[0,-1]], Psi[[0,-1]], Kc/2., Gaussian)
	else:
		#Adding MPrime terms for t=0...T-3
		Theta[:-2], K[:-2] = GetVM(Theta[:-2], K[:-2], Psi1[:-2], Kc/4., Gaussian)
		#Adding MPrimePrime terms for t=2...T-1
		Theta[2:], K[2:] = GetVM(Theta[2:], K[2:], Psi11[2:], Kc/4., Gaussian)
	#return K1, Theta1%tao
	return K, Theta%tao #End GetPosterierPhi()

#Map Estimates
def GetMapX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model ):
	Vc +=Vmin
	Vd +=Vmin
	Ve +=Vmin
	Vo +=Vmin
	#Adding all common terms for t=0...T-1
	ax = np.sum(Alphax**2,1)/(-2*Vd)
	bx = 1/(-2*Vo) + Alphax**2/(-2*Vd)
	cx = Alphax**2/Vd
	dx = np.sum(Alphax*Betax,1)/Vd 
	ex = x_/Vo + Alphax*Betax/(-Vd)
	#Adding M terms for t=1...T-2
	ax[1:-1] += 1/(-2*Ve)
	bx[1:-1] += 1/(-2*Vc)
	dx[1:-1] += M[1:-1]/Ve
	ex[1:-1] += m[1:-1]/Vc
	if Model=='1stOrder':
		ax[[0,-1]] += 1/(-4*Ve)
		bx[[0,-1]] += 1/(-4*Vc)
		dx[[0,-1]] += M[[0,-1]]/(2*Ve)
		ex[[0,-1]] += m[[0,-1]]/(2*Vc)
	else:	#Model=='2ndOrder'
		#Adding MPrime terms for t=0...T-3
		ax[:-2] += 1/(-8*Ve)
		bx[:-2] += 1/(-8*Vc)
		dx[:-2] += M1[:-2]/(4*Ve)
		ex[:-2] += m1[:-2]/(4*Vc)
		#Adding MPrimePrime terms for t=2...T-1
		ax[2:] += 1/(-8*Ve)
		bx[2:] += 1/(-8*Vc)
		dx[2:] += M11[2:]/(4*Ve)
		ex[2:] += m11[2:]/(4*Vc)

	MuXt = ( np.sum(cx*ex/(2*bx),1)-dx )/( 2*ax-np.sum(cx*cx/( 2*bx ),1) )
	EX = MuXt
	Ex = ( cx*V2M(MuXt) + ex )/(-2*bx)
	return EX, Ex #End GetMapX()

def GetMapY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model ):
	Vc +=Vmin
	Vd +=Vmin
	Ve +=Vmin
	Vo +=Vmin
	#Adding all common terms for t=0...T-1
	ay = np.sum(Alphay**2,1)/(-2*Vd)
	by = 1/(-2*Vo) + Alphay**2/(-2*Vd)
	cy = Alphay**2/Vd
	dy = np.sum(Alphay*Betay,1)/(-Vd)
	ey = y_/Vo + Alphay*Betay/Vd
	#Adding M terms for t=0...T-1
	ay[1:-1] += 1/(-2*Ve)
	by[1:-1] += 1/(-2*Vc)
	dy[1:-1] += N[1:-1]/Ve
	ey[1:-1] += n[1:-1]/Vc
	if Model=='1stOrder':
		ay[[0,-1]] += 1/(-4*Ve)
		by[[0,-1]] += 1/(-4*Vc)
		dy[[0,-1]] += N[[0,-1]]/(2*Ve)
		ey[[0,-1]] += n[[0,-1]]/(2*Vc)
	else:	#Model=='2ndOrder'
		#Adding MPrime terms for t=0...T-3
		ay[:-2] += 1/(-8*Ve)
		by[:-2] += 1/(-8*Vc)
		dy[:-2] += N1[:-2]/(4*Ve)
		ey[:-2] += n1[:-2]/(4*Vc)
		#Adding MPrimePrime terms for t=2...T-1
		ay[2:] += 1/(-8*Ve)
		by[2:] += 1/(-8*Vc)
		dy[2:] += N11[2:]/(4*Ve)
		ey[2:] += n11[2:]/(4*Vc)

	MuYt = ( np.sum(cy*ey/(2*by),1)-dy )/( 2*ay-np.sum(cy*cy/( 2*by ),1) )
	EY = MuYt
	Ey = ( cy*V2M(MuYt) + ey )/(-2*by)
	return EY, Ey #End GetMapY()

def GetMapPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model, Gaussian ):
	Kh = Kh1#*np.exp( -H42Sq/Kh0 )
	Kc = Kc/(Kc/Kmax+1)
	Kd = Kd/(Kd/Kmax+1)
	Kh = Kh/(Kh/Kmax+1)

	#Common terms for t=0...T-1
	Theta, K = GetVM(Phi_, Kh, PsiE, Kd, Gaussian)
	if Gaussian:
		K = np.tile(K,Theta.shape)
		
	#Adding M terms for t=1...T-2
	Theta[1:-1], K[1:-1] = GetVM(Theta[1:-1], K[1:-1], Psi[1:-1], Kc, Gaussian)
	if Model=='1stOrder':
		#Adding M terms for t=1...T-2
		Theta[[0,-1]], K[[0,-1]] = GetVM(Theta[[0,-1]], K[[0,-1]], Psi[[0,-1]], Kc/2., Gaussian)
	else:
		#Adding MPrime terms for t=0...T-3
		Theta[:-2], K[:-2] = GetVM(Theta[:-2], K[:-2], Psi1[:-2], Kc/4., Gaussian)
		#Adding MPrimePrime terms for t=2...T-1
		Theta[2:], K[2:] = GetVM(Theta[2:], K[2:], Psi11[2:], Kc/4., Gaussian)
	return Theta%tao #End GetMapPhi()

def GetMarginalX( Vc, Vd, Ve, Alphax, Betax, M, m, M1, m1, M11, m11, Model ):
	#Adding all common terms for t=0...T-1
	ax = np.sum(Alphax**2,1)/(-2*Vd)
	bx = Alphax**2/(-2*Vd)
	cx = Alphax**2/Vd
	dx = np.sum(Alphax*Betax,1)/Vd 
	ex = Alphax*Betax/(-Vd)
	#Adding M terms for t=0...T-1
	ax[1:-1] += 1/(-2*Ve)
	bx[1:-1] += 1/(-2*Vc)
	dx[1:-1] += M[1:-1]/Ve
	ex[1:-1] += m[1:-1]/Vc
	if Model=='1stOrder':
		ax[[0,-1]] += 1/(-4*Ve)
		bx[[0,-1]] += 1/(-4*Vc)
		dx[[0,-1]] += M[[0,-1]]/(2*Ve)
		ex[[0,-1]] += m[[0,-1]]/(2*Vc)
	else:	#Model=='2ndOrder'
		#Adding MPrime terms for t=0...T-3
		ax[:-2] += 1/(-8*Ve)
		bx[:-2] += 1/(-8*Vc)
		dx[:-2] += M1[:-2]/(4*Ve)
		ex[:-2] += m1[:-2]/(4*Vc)
		#Adding MPrimePrime terms for t=2...T-1
		ax[2:] += 1/(-8*Ve)
		bx[2:] += 1/(-8*Vc)
		dx[2:] += M11[2:]/(4*Ve)
		ex[2:] += m11[2:]/(4*Vc)

	MuX = ( np.sum(cx*ex/(2*bx),1)-dx )/( 2*ax-np.sum(cx*cx/( 2*bx ),1) )
	VX = 1/( np.sum(cx*cx/(2*bx),1)-2*ax )
	Vx = 1/(-2*bx)
	return MuX, VX, Vx, ax, bx, cx, dx, ex #End GetMarginalX()

def GetMarginalY( Vc, Vd, Ve, Alphay, Betay, N, n, N1, n1, N11, n11, Model ):
	#Adding all common terms for t=0...T-1
	ay = np.sum(Alphay**2,1)/(-2*Vd)
	by = Alphay**2/(-2*Vd)
	cy = Alphay**2/Vd
	dy = np.sum(Alphay*Betay,1)/(-Vd)
	ey = Alphay*Betay/Vd
	#Adding M terms for t=1...T-2
	ay[1:-1] += 1/(-2*Ve)
	by[1:-1] += 1/(-2*Vc)
	dy[1:-1] += N[1:-1]/Ve
	ey[1:-1] += n[1:-1]/Vc
	if Model=='1stOrder':
		ay[[0,-1]] += 1/(-4*Ve)
		by[[0,-1]] += 1/(-4*Vc)
		dy[[0,-1]] += N[[0,-1]]/(2*Ve)
		ey[[0,-1]] += n[[0,-1]]/(2*Vc)
	else:	#Model=='2ndOrder'
		#Adding MPrime terms for t=0...T-3
		ay[:-2] += 1/(-8*Ve)
		by[:-2] += 1/(-8*Vc)
		dy[:-2] += N1[:-2]/(4*Ve)
		ey[:-2] += n1[:-2]/(4*Vc)
		#Adding MPrimePrime terms for t=2...T-1
		ay[2:] += 1/(-8*Ve)
		by[2:] += 1/(-8*Vc)
		dy[2:] += N11[2:]/(4*Ve)
		ey[2:] += n11[2:]/(4*Vc)

	MuY = ( np.sum(cy*ey/(2*by),1)-dy )/( 2*ay-np.sum(cy*cy/( 2*by ),1) )
	VY = 1/( np.sum(cy*cy/(2*by),1)-2*ay )
	Vy = 1/(-2*by)
	return MuY, VY, Vy, ay, by, cy, dy, ey #End GetMarginalY()

def GetMarginalPhi( Kc, Kd, Psi, Psi1, Psi11, PsiE, Model, Gaussian ):
	Kc = Kc/(Kc/Kmax+1)
	Kd = Kd/(Kd/Kmax+1)
	#Common terms for t=0...T-1
	Theta = deepcopy(PsiE)
	K = np.tile(Kd,(T,Nc))
	#Adding M terms for t=1...T-2
	Theta[1:-1], K[1:-1] = GetVM(Theta[1:-1], K[1:-1], Psi[1:-1], Kc, Gaussian)
	if Model=='1stOrder':
		#Adding M terms for t=1...T-2
		Theta[[0,-1]], K[[0,-1]] = GetVM(Theta[[0,-1]], K[[0,-1]], Psi[[0,-1]], Kc/2., Gaussian)
	else:
		#Adding MPrime terms for t=0...T-3
		Theta[:-2], K[:-2] = GetVM(Theta[:-2], K[:-2], Psi1[:-2], Kc/4., Gaussian)
		#Adding MPrimePrime terms for t=2...T-1
		Theta[2:], K[2:] = GetVM(Theta[2:], K[2:], Psi11[2:], Kc/4., Gaussian)
	#return Theta1%tao
	return K, Theta%tao #End GetMarginalPhi()

def Fvalue( X0, *Args ):
	'''
	Input unknows are standard deviations!
	Convert them into variances!
	'''
	EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1, Alphax, Alphay, Betax, Betay, M, N, m, n, M1, N1, m1, n1, M11, N11, m11, n11, Psi, Psi1, Psi11, PsiE,  x_, y_, Phi_, Model, VasVariable, Gaussian = Args

	Kh0 = X0[-1]
	Vc, Vd, Ve, Vo, Kc, Kd, Kh1 = X0[:-1]**2
	Kh = Kh1#*np.exp( -H42Sq/Kh0 )
	Vc +=Vmin
	Vd +=Vmin
	Ve +=Vmin
	Vo +=Vmin
	Kc = Kc/(Kc/Kmax+1)
	Kd = Kd/(Kd/Kmax+1)
	Kh = Kh/(Kh/Kmax+1)

	MuX, VX, Vx, ax, bx, cx, dx, ex = GetMarginalX( Vc, Vd, Ve, Alphax, Betax, M, m, M1, m1, M11, m11, Model )
	MuY, VY, Vy, ay, by, cy, dy, ey  = GetMarginalY( Vc, Vd, Ve, Alphay, Betay, N, n, N1, n1, N11, n11, Model )
	K, Theta = GetMarginalPhi( Kc, Kd, Psi, Psi1, Psi11, PsiE, Model, Gaussian )

	ExMux = ( cx*EXx + ex*Ex )/(-2*bx)
	EMuxMux = (cx**2 * V2M(EXX) + 2*cx*ex*V2M(EX)+ex**2)/(4*bx**2)
	EyMuy = ( cy*EYy + ey*Ey )/(-2*by)
	EMuyMuy = (cy**2 * V2M(EYY) + 2*cy*ey*V2M(EY)+ey**2)/(4*by**2)

	if Gaussian:
		ObsCost = np.sum( -.5*Kh*(Theta1**2+K1-2*Theta1*Phi_+Phi_**2) + .5*np.log(Kh) )
	else:
		ObsCost = np.sum( Kh*np.cos(Theta1-Phi_)*A(K1) - LogI(Kh,0) )
	
	if VasVariable:
		ObsCost += np.sum( ( Exx - 2*Ex*x_ + x_**2 )/(-2*Vo) + ( Eyy - 2*Ey*y_ + y_**2 )/(-2*Vo) - 1.0*np.log(Vo) )
	
	EvCost = np.sum( (EXX-2*MuX*EX+MuX**2)/(-2*VX) + (EYY-2*MuY*EY+MuY**2)/(-2*VY) - .5*np.log(VX*VY) )
	
	CamCost = np.sum( (Exx-2*ExMux+EMuxMux)/(-2*Vx) + (Eyy-2*EyMuy+EMuyMuy)/(-2*Vy) - .5*np.log(Vx*Vy) )
	if Gaussian:
		CamCost += np.sum( -.5*K*(Theta1**2+K1-2*Theta1*Theta+Theta**2) + .5*np.log(K) )
	else:
		CamCost += np.sum( K*np.cos(Theta1-Theta)*A(K1) - LogI(K,0) )
		
	fvalue = -ObsCost-EvCost-CamCost
	#print 'Fvalue:'+str(fvalue)
	return fvalue #end of Fvalue()

def FvalueXY( X0, *Args ):
	'''
	Input unknows are standard deviations!
	Convert them into variances!
	'''
	global OptSolution
	global OptFvalue
	
	EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1, Alphax, Alphay, Betax, Betay, M, N, m, n, M1, N1, m1, n1, M11, N11, m11, n11, Psi, Psi1, Psi11, PsiE,  x_, y_, Phi_, Model, VasVariable, Gaussian = Args

	Vc, Vd, Ve, Vo = X0**2
	Vc +=Vmin
	Vd +=Vmin
	Ve +=Vmin
	Vo +=Vmin

	MuX, VX, Vx, ax, bx, cx, dx, ex = GetMarginalX( Vc, Vd, Ve, Alphax, Betax, M, m, M1, m1, M11, m11, Model )
	MuY, VY, Vy, ay, by, cy, dy, ey  = GetMarginalY( Vc, Vd, Ve, Alphay, Betay, N, n, N1, n1, N11, n11, Model )

	ExMux = ( cx*EXx + ex*Ex )/(-2*bx)
	EMuxMux = (cx**2 * V2M(EXX) + 2*cx*ex*V2M(EX)+ex**2)/(4*bx**2)
	EyMuy = ( cy*EYy + ey*Ey )/(-2*by)
	EMuyMuy = (cy**2 * V2M(EYY) + 2*cy*ey*V2M(EY)+ey**2)/(4*by**2)

	ObsCost = 0.0
	if VasVariable:
		ObsCost += np.sum( ( Exx - 2*Ex*x_ + x_**2 )/(-2*Vo) + ( Eyy - 2*Ey*y_ + y_**2 )/(-2*Vo) - 1.0*np.log(Vo) )
	EvCost = np.sum( (EXX-2*MuX*EX+MuX**2)/(-2*VX) + (EYY-2*MuY*EY+MuY**2)/(-2*VY) - .5*np.log(VX*VY) )
	CamCost = np.sum( (Exx-2*ExMux+EMuxMux)/(-2*Vx) + (Eyy-2*EyMuy+EMuyMuy)/(-2*Vy) - .5*np.log(Vx*Vy) )
	fvalue = -ObsCost-EvCost-CamCost

	if fvalue<OptFvalue:
		OptSolution = deepcopy(X0)
		OptFvalue = fvalue

	#print 'FvalueXY:'+str(fvalue)
	#print 'X0:'+str(X0)
	#print 'OptFvalue:',OptFvalue
	#print 'OptSolution:',OptSolution

	return fvalue #end of Fvalue()

def FvaluePhi( X0, *Args ):
	'''
	Input unknows are standard deviations!
	Convert them into variances!
	'''
	global OptSolution
	global OptFvalue
	
	Kh0 = X0[-1]
	Kc, Kd, Kh1 = X0[:-1]**2
	Kh = Kh1#*np.exp( -H42Sq/Kh0 )
	Kc = Kc/(Kc/Kmax+1)
	Kd = Kd/(Kd/Kmax+1)
	Kh = Kh/(Kh/Kmax+1)

	EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1, Alphax, Alphay, Betax, Betay, M, N, m, n, M1, N1, m1, n1, M11, N11, m11, n11, Psi, Psi1, Psi11, PsiE,  x_, y_, Phi_, Model, VasVariable, Gaussian = Args
	K, Theta = GetMarginalPhi( Kc, Kd, Psi, Psi1, Psi11, PsiE, Model, Gaussian )

	if Gaussian:
		ObsCost = np.sum( -.5*Kh*(Theta1**2+K1-2*Theta1*Phi_+Phi_**2) + .5*np.log(Kh) )
		CamCost = np.sum( -.5*K*(Theta1**2+K1-2*Theta1*Theta+Theta**2) + .5*np.log(K) )
	else:
		ObsCost = np.sum( Kh*np.cos(Theta1-Phi_)*A(K1) - LogI(Kh,0) )
		CamCost = np.sum( K*np.cos(Theta1-Theta)*A(K1) - LogI(K,0) )
	fvalue = -ObsCost-CamCost
	
	if fvalue<OptFvalue:
		OptSolution = deepcopy(X0)
		OptFvalue = fvalue

	#print 'FvaluePhi:'+str(fvalue)
	#print 'OptFvalue:',OptFvalue
	#print 'X0:'+str(X0)
	return fvalue #end of Fvalue()

def Optimize( X0, Args, VasVariable=False ):
	method = 8
	#Options = {'maxiter':25}
	Options = {}
	#if VasVariable:
	#	Options = {'maxiter':1000, 'maxfev':100}
	if method == 0:
		MethodName = 'Nelder-Mead'#
	elif method == 1:
		MethodName = 'Powell'#
	elif method == 2:
		MethodName = 'CG'
	elif method == 3:
		MethodName = 'BFGS'
	elif method == 4:
		MethodName = 'Newton-CG' #Jacobian Must
	elif method == 5:
		MethodName = 'L-BFGS-B'
	elif method == 6:
		MethodName = 'TNC'#
	elif method == 7:
		MethodName = 'COBYLA'
	elif method == 8:
		MethodName = 'SLSQP'#*
	elif method == 9:
		MethodName = 'dogleg'#Jacobian Must
	else:
		MethodName = 'trust-ncg'#Jacobian Must

	#'''Optimizing all the unknows together!'''
	#res = minimize(fun=Fvalue, x0=X0**0.5, args = Args, method = MethodName, options = Options)
	#X0 = res.x**2

	global OptSolution
	global OptFvalue
	
	'''Optimizing Gaussian parameters!'''
	#minimizer_kwargs = {'method': 'Nelder-Mead', 'args':Args}
	#res = basinhopping(FvalueXY, X0[:4]**0.5, minimizer_kwargs=minimizer_kwargs, niter=200)
	OptFvalue = float('inf')
	res = minimize(fun=FvalueXY, x0=X0[:4]**0.5, args = Args, method = MethodName, options = Options)
	#X0[:4] = res.x**2
	X0[:4] = OptSolution**2
	print 'XY Iterations res'
	print res
	print 'OptFvalue:',OptFvalue
	

	'''Optimizing vonMises parameters!'''
	OptFvalue = float('inf')
	X0[4:-1] = X0[4:-1]**0.5
	res = minimize(fun=FvaluePhi, x0=X0[4:], args = Args, method = MethodName, options = Options)
	#res = basinhopping(FvaluePhi, X0[4:], minimizer_kwargs=minimizer_kwargs, niter=200)
	#X0[4:-1] = res.x[:-1]**2
	#X0[-1] = res.x[-1]
	X0[4:-1] = OptSolution[:-1]**2
	X0[-1] = OptSolution[-1]
	print 'Phi Iterations res'
	print res
	print 'OptFvalue:',OptFvalue
	return X0 #end of Optimize()

def pgm(sensor_data, cvx_sol, VasVariable = False, Gaussian = False):
	solution_list = {}
	global Nc
	global T
	x_, y_, Phi_, Nc, T, V, H = sensor_data
	[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, X, Y, x, y, Phi] = GetInitialSolution(cvx_sol,Phi_, Nc, T,x_,y_,H)
	[X_, Y_] = deepcopy([X, Y])
	InitialParameters = [Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, X, Y, x, y, Phi]
	OptimalParameters = ProbabilisticOptimization( x_, y_, Phi_, H, V, Nc, T, InitialParameters, '1stOrder', VasVariable, Gaussian )
	return OptimalParameters 
	#end of PGM()
	
def ProbabilisticOptimization( x_, y_, Phi_, H, V, Nc, T, InitialParameters, Model='1stOrder' , VasVariable=False, Gaussian=False):
	global tao
	global H42Sq
	global Vmin
	global Kmax
	global FiltLen
	global ThV
	global ThE
	ThE = 1.0
	ThV = 0.01
	Vmin = 0.0
	Kmax = float('inf')
	tao = np.pi*2
	H42Sq = (H-42)**2
	FiltLen = 3
	[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, X, Y, x, y, Phi] = InitialParameters
	X0 = np.asarray([Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0])

	print 'Initial Parameters:'
	print '[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0]'
	print str([Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0])
	PrevCost = float('inf')

	MaxIterations = 250
	#OptimalParameters = Algo1( InitialParameters, x_, y_, Phi_, V, H, Nc, T, MaxIterations, Model, VasVariable )
	OptimalParameters = Algo1_( InitialParameters, x_, y_, Phi_, V, H, Nc, T, MaxIterations, Model, VasVariable, Gaussian )
	#OptimalParameters2 = Algo2( InitialParameters, x_, y_, Phi_, V, H, Nc, T, MaxIterations, Model, VasVariable )
	#OptimalParameters3 = Algo3( InitialParameters, x_, y_, Phi_, V, H, Nc, T, MaxIterations, Model, VasVariable )
	#OptimalParameters4 = Algo4( InitialParameters, x_, y_, Phi_, V, H, Nc, T, MaxIterations, Model, VasVariable )
	#return OptimalParameters, OptimalParameters2, OptimalParameters3, OptimalParameters4 #End of ProbabilisticOptimization()
	return OptimalParameters
	#End of ProbabilisticOptimization()

def GetAllMeans( X, Y, x, y, Phi, Nc, T, FiltLen ):
	M, m = GetLocationMeans( X, x, Nc, T, FiltLen )
	M1, m1 = GetLocationMeansPrime( X, x, Nc, T, FiltLen )
	M11, m11 = GetLocationMeansPrimePrime( X, x, Nc, T, FiltLen )
	N, n = GetLocationMeans( Y, y, Nc, T, FiltLen )
	N1, n1 = GetLocationMeansPrime( Y, y, Nc, T, FiltLen )
	N11, n11 = GetLocationMeansPrimePrime( Y, y, Nc, T, FiltLen )
	Psi = GetOrientationMeans( X, Y, x, y, Phi, Nc, T, FiltLen )
	Psi1 = GetOrientationMeansPrime( X, Y, x, y, Phi, Nc, T, FiltLen )
	Psi11 = GetOrientationMeansPrimePrime( X, Y, x, y, Phi, Nc, T, FiltLen )
	return M, m, M1, m1, M11, m11, N, n, N1, n1, N11, n11, Psi, Psi1, Psi11
	
def GetAlphaBeta( X, Y, x, y, Phi, Nc, T, FiltLen ):
	AlphaPhi = V2M(X)-x
	BetaPhi = V2M(Y)-y
	Alphax = np.cos(Phi)
	Alphay = np.sin(Phi)
	Betax = BetaPhi*Alphay
	Betay = -AlphaPhi*Alphax
	PsiE = arctan2( AlphaPhi, BetaPhi )%(2*np.pi)
	return AlphaPhi, BetaPhi, Alphax, Alphay, Betax, Betay, PsiE
	
def GetPosterierXYPhi( Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, Alphax, Betax, Alphay, Betay, M, m, M1, m1, M11, m11, N, n, N1, n1, N11, n11, Psi, Psi1, Psi11, PsiE, x_,  y_, Phi_ , H, V, Model, VasVariable, Gaussian ):

	if VasVariable:
		EXX, EX, Exx, Ex, EXx = GetPosterierX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
		EYY, EY, Eyy, Ey, EYy = GetPosterierY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
	else:
		EXX, EX, Exx, Ex, EXx = GetPosterierX( Vc, Vd, Ve, V, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
		EYY, EY, Eyy, Ey, EYy = GetPosterierY( Vc, Vd, Ve, V, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
		
	K1, Theta1 = GetPosterierPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model, Gaussian )
	return EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1
	
def Algo1_( InitialParameters, x_, y_, Phi_, V, H, Nc, T, MaxIterations, Model, VasVariable, Gaussian ):
	'''
	Alternate iterations for Variances and expectations!
	'''
	global OptFvalue
	OptFvalue = float('inf')
	PrevCost = float('inf')
	PrevCostPhi = float('inf')
	PrevCostXY = float('inf')
	DecreaseInCost = float('inf')
	DecreaseInCostPhi = float('inf')
	DecreaseInCostXY = float('inf')
	DecreaseInCostE = float('inf')
	DecreaseInCostV = float('inf')
	print 'Executing Algorithm1!'

	[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, X, Y, x, y, Phi] = InitialParameters
	X0 = np.asarray([Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0])
	M, m, M1, m1, M11, m11, N, n, N1, n1, N11, n11, Psi, Psi1, Psi11 = GetAllMeans( X, Y, x, y, Phi, Nc, T, FiltLen )
	AlphaPhi, BetaPhi, Alphax, Alphay, Betax, Betay, PsiE = GetAlphaBeta( X, Y, x, y, Phi, Nc, T, FiltLen )
	EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1 = GetPosterierXYPhi( Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, Alphax, Betax, Alphay, Betay, M, m, M1, m1, M11, m11, N, n, N1, n1, N11, n11, Psi, Psi1, Psi11, PsiE, x_,  y_, Phi_, H, V, Model, VasVariable, Gaussian )

	Args = (EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1, Alphax, Alphay, Betax, Betay, M, N, m, n, M1, N1, m1, n1, M11, N11, m11, n11, Psi, Psi1, Psi11, PsiE,  x_, y_, Phi_, Model, VasVariable, Gaussian)

	Y0 = deepcopy(X0)
	Y0[:-1] = Y0[:-1]**.5
	CurrentCost = Fvalue(Y0, *Args)
	CurrentCostXY = FvalueXY(Y0[:4], *Args)
	CurrentCostPhi = FvaluePhi(Y0[4:], *Args)
	DecreaseInCost = PrevCost - CurrentCost
	DecreaseInCostXY = PrevCostXY - CurrentCostXY
	DecreaseInCostPhi = PrevCostPhi - CurrentCostPhi
	PrevCost = CurrentCost
	PrevCostXY = CurrentCostXY
	PrevCostPhi = CurrentCostPhi
	DecreaseInCostE = DecreaseInCost
	print 'FvalueAfterPost:'+str(CurrentCost)
	print 'FvalueXY:'+str(CurrentCostXY)
	print 'FvaluePhi:'+str(CurrentCostPhi)
	print 'DecreaseInCost:'+str(DecreaseInCost)
	print 'DecreaseInCostXY:'+str(DecreaseInCostXY)
	print 'DecreaseInCostPhi:'+str(DecreaseInCostPhi)

	for iteration in range(MaxIterations):
		print 
		print 'Iteration:'+str(iteration)
		print X0
		#Optimize the Parameters!
		print 'iteration:'+str(iteration)
		X0 = Optimize( X0, Args, VasVariable )
		[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0] = X0	
		print 'Optimized Parameters'
		print '[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0]:'
		print X0
		Y0 = deepcopy(X0)
		Y0[:-1] = Y0[:-1]**.5

		CurrentCost = Fvalue(Y0, *Args)
		CurrentCostXY = FvalueXY(Y0[:4], *Args)
		CurrentCostPhi = FvaluePhi(Y0[4:], *Args)
		DecreaseInCost = PrevCost - CurrentCost
		DecreaseInCostXY = PrevCostXY - CurrentCostXY
		DecreaseInCostPhi = PrevCostPhi - CurrentCostPhi
		PrevCost = CurrentCost
		PrevCostXY = CurrentCostXY
		PrevCostPhi = CurrentCostPhi
		DecreaseInCostV = DecreaseInCost
		print 'FvalueAfterOptmzn:'+str(CurrentCost)
		print 'FvalueXY:'+str(CurrentCostXY)
		print 'FvaluePhi:'+str(CurrentCostPhi)
		print 'DecreaseInCost:'+str(DecreaseInCost)
		print 'DecreaseInCostXY:'+str(DecreaseInCostXY)
		print 'DecreaseInCostPhi:'+str(DecreaseInCostPhi)

		#Compute Posterior Expectations!
		def ComputePosterior( MeanValues, AlphaBetas ):
			M, m, M1, m1, M11, m11, N, n, N1, n1, N11, n11, Psi, Psi1, Psi11 = MeanValues
			Alphax, Betax, Alphay, Betay, AlphaPhi, BetaPhi = AlphaBetas
	
			#Update X,x
			if VasVariable:
				X, x = GetMapX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
			else:
				X, x = GetMapX( Vc, Vd, Ve, V, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
			AlphaPhi = V2M(X)-x
			Betay = -AlphaPhi*Alphax

			#Update Y,y
			if VasVariable:
				Y, y = GetMapY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
			else:
				Y, y = GetMapY( Vc, Vd, Ve, V, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
			BetaPhi = V2M(Y)-y
			Betax = BetaPhi*Alphay

			#Update Phi
			PsiE = arctan2( AlphaPhi,BetaPhi )%tao
			Phi = GetMapPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model, Gaussian )
			Alphax = np.cos(Phi)
			Alphay = np.sin(Phi)
			Betax = BetaPhi*Alphay
			Betay = -AlphaPhi*Alphax

			M, m, M1, m1, M11, m11, N, n, N1, n1, N11, n11, Psi, Psi1, Psi11 = GetAllMeans( X, Y, x, y, Phi, Nc, T, FiltLen )
			EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1 = GetPosterierXYPhi( Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, Alphax, Betax, Alphay, Betay, M, m, M1, m1, M11, m11, N, n, N1, n1, N11, n11, Psi, Psi1, Psi11, PsiE, x_,  y_, Phi_ , H, V, Model, VasVariable, Gaussian )
		
			Args = (EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1, Alphax, Alphay, Betax, Betay, M, N, m, n, M1, N1, m1, n1, M11, N11, m11, n11, Psi, Psi1, Psi11, PsiE,  x_, y_, Phi_, Model, VasVariable, Gaussian)
			AlphaBetas = Alphax, Betax, Alphay, Betay, AlphaPhi, BetaPhi
			NewMeanValues = M, m, M1, m1, M11, m11, N, n, N1, n1, N11, n11, Psi, Psi1, Psi11
			return Args, AlphaBetas, NewMeanValues, X, Y, x, y, Phi

		AlphaBetas =  Alphax, Betax, Alphay, Betay, AlphaPhi, BetaPhi
		DecreaseInCost = float('inf')
		DecreaseInCostE = 0.0
		SubIteration = 0
		while (DecreaseInCost>5. and SubIteration<1):
			SubIteration += 1
			MeanValues = M, m, M1, m1, M11, m11, N, n, N1, n1, N11, n11, Psi, Psi1, Psi11
			Args, AlphaBetas, NewMeanValues, X, Y, x, y, Phi = ComputePosterior( MeanValues, AlphaBetas )
			Y0 = deepcopy(X0)
			Y0[:-1] = Y0[:-1]**.5
			CurrentCost = Fvalue(Y0, *Args)
			CurrentCostXY = FvalueXY(Y0[:4], *Args)
			CurrentCostPhi = FvaluePhi(Y0[4:], *Args)
			DecreaseInCost = PrevCost - CurrentCost
			DecreaseInCostXY = PrevCostXY - CurrentCostXY
			DecreaseInCostPhi = PrevCostPhi - CurrentCostPhi
			PrevCost = CurrentCost
			PrevCostXY = CurrentCostXY
			PrevCostPhi = CurrentCostPhi
			DecreaseInCostE += DecreaseInCost
			print 'FvalueAfterPost:'+str(CurrentCost)
			print 'FvalueXY:'+str(CurrentCostXY)
			print 'FvaluePhi:'+str(CurrentCostPhi)
			print 'DecreaseInCost:'+str(DecreaseInCost)
			print 'DecreaseInCostXY:'+str(DecreaseInCostXY)
			print 'DecreaseInCostPhi:'+str(DecreaseInCostPhi)
		M, m, M1, m1, M11, m11, N, n, N1, n1, N11, n11, Psi, Psi1, Psi11 = NewMeanValues
		Alphax, Betax, Alphay, Betay, AlphaPhi, BetaPhi = AlphaBetas
		if DecreaseInCostE<3.0 and DecreaseInCostV<0.01:
		#if DecreaseInCostE<5.0 and DecreaseInCostV<0.01:
		#if DecreaseInCostE<0.50 and DecreaseInCostV<0.001:
		#if DecreaseInCostE<1.0 and DecreaseInCostV<0.01:
			break
	print 'Completed Algorithm1!'
	return [X, Y, x, y, Phi, Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, CurrentCost, iteration]	#end of Algo1_()***

def Algo1( InitialParameters, x_, y_, Phi_, V, H, Nc, T, MaxIterations, Model, VasVariable ):
	'''
	Alternate iterations for Variances and expectations!
	'''
	print 'Executing Algorithm1!'
	[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, X, Y, x, y, Phi] = InitialParameters
	X0 = np.asarray([Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0])
	M, m = GetLocationMeans( X, x, Nc, T, FiltLen )
	M1, m1 = GetLocationMeansPrime( X, x, Nc, T, FiltLen )
	M11, m11 = GetLocationMeansPrimePrime( X, x, Nc, T, FiltLen )
	N, n = GetLocationMeans( Y, y, Nc, T, FiltLen )
	N1, n1 = GetLocationMeansPrime( Y, y, Nc, T, FiltLen )
	N11, n11 = GetLocationMeansPrimePrime( Y, y, Nc, T, FiltLen )
	Psi = GetOrientationMeans( X, Y, x, y, Phi, Nc, T, FiltLen )
	Psi1 = GetOrientationMeansPrime( X, Y, x, y, Phi, Nc, T, FiltLen )
	Psi11 = GetOrientationMeansPrimePrime( X, Y, x, y, Phi, Nc, T, FiltLen )

	AlphaPhi = V2M(X)-x
	BetaPhi = V2M(Y)-y
	Alphax = np.cos(Phi)
	Alphay = np.sin(Phi)
	Betax = BetaPhi*Alphay
	Betay = -AlphaPhi*Alphax
	PsiE = arctan2( AlphaPhi, BetaPhi )%tao
	PrevCost = float('inf')
	PrevCostPhi = float('inf')
	PrevCostXY = float('inf')
	DecreaseInCost = float('inf')
	DecreaseInCostPhi = float('inf')
	DecreaseInCostXY = float('inf')
	DecreaseInCostE = float('inf')
	DecreaseInCostV = float('inf')
	for iteration in range(MaxIterations):
		print 
		print 'Iteration:'+str(iteration)
		print X0

		#Compute the expectations for likelihood!
		if VasVariable:
			EXX, EX, Exx, Ex, EXx = GetPosterierX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
			EYY, EY, Eyy, Ey, EYy = GetPosterierY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
		else:
			EXX, EX, Exx, Ex, EXx = GetPosterierX( Vc, Vd, Ve, V, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
			EYY, EY, Eyy, Ey, EYy = GetPosterierY( Vc, Vd, Ve, V, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
		K1, Theta1 = GetPosterierPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model )
		Args = (EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1, Alphax, Alphay, Betax, Betay, M, N, m, n, M1, N1, m1, n1, M11, N11, m11, n11, Psi, Psi1, Psi11, PsiE,  x_, y_, Phi_, Model, VasVariable)
		Y0 = deepcopy(X0)
		Y0[:-1] = Y0[:-1]**.5
		CurrentCost = Fvalue(Y0, *Args)
		CurrentCostXY = FvalueXY(Y0[:4], *Args)
		CurrentCostPhi = FvaluePhi(Y0[4:], *Args)
		DecreaseInCost = PrevCost - CurrentCost
		DecreaseInCostXY = PrevCostXY - CurrentCostXY
		DecreaseInCostPhi = PrevCostPhi - CurrentCostPhi
		PrevCost = CurrentCost
		PrevCostXY = CurrentCostXY
		PrevCostPhi = CurrentCostPhi
		DecreaseInCostE = DecreaseInCost
		print 'FvalueAfterPost:'+str(CurrentCost)
		print 'FvalueXY:'+str(CurrentCostXY)
		print 'FvaluePhi:'+str(CurrentCostPhi)
		print 'DecreaseInCost:'+str(DecreaseInCost)
		print 'DecreaseInCostXY:'+str(DecreaseInCostXY)
		print 'DecreaseInCostPhi:'+str(DecreaseInCostPhi)

		if DecreaseInCostE+DecreaseInCostV<2.0 and not iteration<=5:
			break

		#Optimize the Parameters!
		print
		print 'iteration:'+str(iteration)
		X0 = Optimize( X0, Args, VasVariable )
		[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0] = X0
		print 'Optimized Parameters'
		print '[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0]:'
		print X0
		Y0 = deepcopy(X0)
		Y0[:-1] = Y0[:-1]**.5

		CurrentCost = Fvalue(Y0, *Args)
		CurrentCostXY = FvalueXY(Y0[:4], *Args)
		CurrentCostPhi = FvaluePhi(Y0[4:], *Args)
		DecreaseInCost = PrevCost - CurrentCost
		DecreaseInCostXY = PrevCostXY - CurrentCostXY
		DecreaseInCostPhi = PrevCostPhi - CurrentCostPhi
		PrevCost = CurrentCost
		PrevCostXY = CurrentCostXY
		PrevCostPhi = CurrentCostPhi
		DecreaseInCostV = DecreaseInCost
		print 'FvalueAfterOptmzn:'+str(CurrentCost)
		print 'FvalueXY:'+str(CurrentCostXY)
		print 'FvaluePhi:'+str(CurrentCostPhi)
		print 'DecreaseInCost:'+str(DecreaseInCost)
		print 'DecreaseInCostXY:'+str(DecreaseInCostXY)
		print 'DecreaseInCostPhi:'+str(DecreaseInCostPhi)

		#Compute Posterior Expectations!
		if VasVariable:
			X, x = GetMapX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
		else:
			X, x = GetMapX( Vc, Vd, Ve, V, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
		AlphaPhi = V2M(X)-x
		Betay = -AlphaPhi*Alphax

		if VasVariable:
			Y, y = GetMapY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
		else:
			Y, y = GetMapY( Vc, Vd, Ve, V, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
		BetaPhi = V2M(Y)-y
		Betax = BetaPhi*Alphay

		PsiE = arctan2( AlphaPhi,BetaPhi )%tao
		Phi = GetMapPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model )
		Alphax = np.cos(Phi)
		Alphay = np.sin(Phi)
		Betax = BetaPhi*Alphay
		Betay = -AlphaPhi*Alphax

		M, m = GetLocationMeans( X, x, Nc, T, FiltLen )
		M1, m1 = GetLocationMeansPrime( X, x, Nc, T, FiltLen )
		M11, m11 = GetLocationMeansPrimePrime( X, x, Nc, T, FiltLen )
		N, n = GetLocationMeans( Y, y, Nc, T, FiltLen )
		N1, n1 = GetLocationMeansPrime( Y, y, Nc, T, FiltLen )
		N11, n11 = GetLocationMeansPrimePrime( Y, y, Nc, T, FiltLen )
		Psi = GetOrientationMeans( X, Y, x, y, Phi, Nc, T, FiltLen )
		Psi1 = GetOrientationMeansPrime( X, Y, x, y, Phi, Nc, T, FiltLen )
		Psi11 = GetOrientationMeansPrimePrime( X, Y, x, y, Phi, Nc, T, FiltLen )
	print 'Completed Algorithm1!'
	return X, Y, x, y, Phi, Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, CurrentCost, iteration	#end of Algo1()

def Algo2( InitialParameters, x_, y_, Phi_, V, H, Nc, T, MaxIterations, Model, VasVariable ):
	'''
	Multiple iterations for expectations!
	'''
	print 'Executing Algorithm2!'
	[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, X, Y, x, y, Phi] = InitialParameters
	X0 = np.asarray([Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0])
	M, m = GetLocationMeans( X, x, Nc, T, FiltLen )
	M1, m1 = GetLocationMeansPrime( X, x, Nc, T, FiltLen )
	M11, m11 = GetLocationMeansPrimePrime( X, x, Nc, T, FiltLen )
	N, n = GetLocationMeans( Y, y, Nc, T, FiltLen )
	N1, n1 = GetLocationMeansPrime( Y, y, Nc, T, FiltLen )
	N11, n11 = GetLocationMeansPrimePrime( Y, y, Nc, T, FiltLen )
	Psi = GetOrientationMeans( X, Y, x, y, Phi, Nc, T, FiltLen )
	Psi1 = GetOrientationMeansPrime( X, Y, x, y, Phi, Nc, T, FiltLen )
	Psi11 = GetOrientationMeansPrimePrime( X, Y, x, y, Phi, Nc, T, FiltLen )

	AlphaPhi = V2M(X)-x
	BetaPhi = V2M(Y)-y
	Alphax = np.cos(Phi)
	Alphay = np.sin(Phi)
	Betax = BetaPhi*Alphay
	Betay = -AlphaPhi*Alphax
	PsiE = arctan2( AlphaPhi, BetaPhi )%tao
	PrevCost = float('inf')
	PrevCostPhi = float('inf')
	PrevCostXY = float('inf')
	DecreaseInCost = float('inf')
	DecreaseInCostE = float('inf')
	DecreaseInCostV = float('inf')
	DecreaseInCostPhi = float('inf')
	DecreaseInCostXY = float('inf')

	for iteration in range(MaxIterations):
		print 
		print 'Iteration:'+str(iteration)
		#Computing expectations for likelihood
		if VasVariable:
			EXX, EX, Exx, Ex, EXx = GetPosterierX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
			EYY, EY, Eyy, Ey, EYy = GetPosterierY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
		else:
			EXX, EX, Exx, Ex, EXx = GetPosterierX( Vc, Vd, Ve, V, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
			EYY, EY, Eyy, Ey, EYy = GetPosterierY( Vc, Vd, Ve, V, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
		K1, Theta1 = GetPosterierPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model )
		Args = (EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1, Alphax, Alphay, Betax, Betay, M, N, m, n, M1, N1, m1, n1, M11, N11, m11, n11, Psi, Psi1, Psi11, PsiE,  x_, y_, Phi_, Model, VasVariable)
		print X0
		Y0 = deepcopy(X0)
		Y0[:-1] = Y0[:-1]**.5
		CurrentCost = Fvalue(Y0, *Args)
		CurrentCostXY = FvalueXY(Y0[:4], *Args)
		CurrentCostPhi = FvaluePhi(Y0[4:], *Args)
		DecreaseInCost = PrevCost - CurrentCost
		DecreaseInCostXY = PrevCostXY - CurrentCostXY
		DecreaseInCostPhi = PrevCostPhi - CurrentCostPhi
		PrevCost = CurrentCost
		PrevCostXY = CurrentCostXY
		PrevCostPhi = CurrentCostPhi
		DecreaseInCostE += DecreaseInCost
		
		print 'FvalueAfterPost:'+str(CurrentCost)
		print 'FvalueXY:'+str(CurrentCostXY)
		print 'FvaluePhi:'+str(CurrentCostPhi)
		print 'DecreaseInCost:'+str(DecreaseInCost)
		print 'DecreaseInCostXY:'+str(DecreaseInCostXY)
		print 'DecreaseInCostPhi:'+str(DecreaseInCostPhi)
				
		if DecreaseInCostE+DecreaseInCostV<2.0:
			break

		#Optimizing the Parameters
		if DecreaseInCost<ThE or iteration == 0:
			DecreaseInCostE = 0.0
			print 'iteration:'+str(iteration)
			X0 = Optimize( X0, Args )
			[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0] = X0
			print 'Optimized Parameters'
			print '[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0]:'
			print X0
			Y0 = deepcopy(X0)
			Y0[:-1] = Y0[:-1]**.5
			CurrentCost = Fvalue(Y0, *Args)
			CurrentCostXY = FvalueXY(Y0[:4], *Args)
			CurrentCostPhi = FvaluePhi(Y0[4:], *Args)
			DecreaseInCost = PrevCost - CurrentCost
			DecreaseInCostXY = PrevCostXY - CurrentCostXY
			DecreaseInCostPhi = PrevCostPhi - CurrentCostPhi
			PrevCost = CurrentCost
			PrevCostXY = CurrentCostXY
			PrevCostPhi = CurrentCostPhi
			DecreaseInCostV = DecreaseInCost
			print 'FvalueAfterOptmzn:'+str(CurrentCost)
			print 'FvalueXY:'+str(CurrentCostXY)
			print 'FvaluePhi:'+str(CurrentCostPhi)
			print 'DecreaseInCost:'+str(DecreaseInCost)
			print 'DecreaseInCostXY:'+str(DecreaseInCostXY)
			print 'DecreaseInCostPhi:'+str(DecreaseInCostPhi)
		
		#Computing the posteriors!
		if VasVariable:
			X, x = GetMapX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
		else:
			X, x = GetMapX( Vc, Vd, Ve, V, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
		AlphaPhi = V2M(X)-x
		Betay = -AlphaPhi*Alphax

		if VasVariable:
			Y, y = GetMapY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
		else:
			Y, y = GetMapY( Vc, Vd, Ve, V, Alphay, Betay, N, n, N1, n1, N11, n11, y_,  Model )
		BetaPhi = V2M(Y)-y
		Betax = BetaPhi*Alphay

		PsiE = arctan2( AlphaPhi,BetaPhi )%tao
		Phi = GetMapPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model )
		Alphax = np.cos(Phi)
		Alphay = np.sin(Phi)
		Betax = BetaPhi*Alphay
		Betay = -AlphaPhi*Alphax

		M, m = GetLocationMeans( X, x, Nc, T, FiltLen )
		M1, m1 = GetLocationMeansPrime( X, x, Nc, T, FiltLen )
		M11, m11 = GetLocationMeansPrimePrime( X, x, Nc, T, FiltLen )
		N, n = GetLocationMeans( Y, y, Nc, T, FiltLen )
		N1, n1 = GetLocationMeansPrime( Y, y, Nc, T, FiltLen )
		N11, n11 = GetLocationMeansPrimePrime( Y, y, Nc, T, FiltLen )
		Psi = GetOrientationMeans( X, Y, x, y, Phi, Nc, T, FiltLen )
		Psi1 = GetOrientationMeansPrime( X, Y, x, y, Phi, Nc, T, FiltLen )
		Psi11 = GetOrientationMeansPrimePrime( X, Y, x, y, Phi, Nc, T, FiltLen )
	return X, Y, x, y, Phi, Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, CurrentCost, iteration #end of Algo2()

def Algo3( InitialParameters, x_, y_, Phi_, V, H, Nc, T, MaxIterations, Model, VasVariable ):
	'''
	Multiple iterations for variances!
	'''
	print 'Executing Algorithm3!'
	[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, X, Y, x, y, Phi] = InitialParameters
	X0 = np.asarray([Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0])
	M, m = GetLocationMeans( X, x, Nc, T, FiltLen )
	M1, m1 = GetLocationMeansPrime( X, x, Nc, T, FiltLen )
	M11, m11 = GetLocationMeansPrimePrime( X, x, Nc, T, FiltLen )
	N, n = GetLocationMeans( Y, y, Nc, T, FiltLen )
	N1, n1 = GetLocationMeansPrime( Y, y, Nc, T, FiltLen )
	N11, n11 = GetLocationMeansPrimePrime( Y, y, Nc, T, FiltLen )
	Psi = GetOrientationMeans( X, Y, x, y, Phi, Nc, T, FiltLen )
	Psi1 = GetOrientationMeansPrime( X, Y, x, y, Phi, Nc, T, FiltLen )
	Psi11 = GetOrientationMeansPrimePrime( X, Y, x, y, Phi, Nc, T, FiltLen )

	AlphaPhi = V2M(X)-x
	BetaPhi = V2M(Y)-y
	Alphax = np.cos(Phi)
	Alphay = np.sin(Phi)
	Betax = BetaPhi*Alphay
	Betay = -AlphaPhi*Alphax
	PsiE = arctan2( AlphaPhi, BetaPhi )%tao
	PrevIterationForExpectation = False
	PrevCost = float('inf')
	PrevCostPhi = float('inf')
	PrevCostXY = float('inf')
	DecreaseInCost = float('inf')
	DecreaseInCostPhi = float('inf')
	DecreaseInCostXY = float('inf')
	DecreaseInCostE = float('inf')
	DecreaseInCostV = float('inf')
	for iteration in range(MaxIterations):
		print 
		print 'Iteration:'+str(iteration)
		#Computing expectations for likelihood
		if VasVariable:
			EXX, EX, Exx, Ex, EXx = GetPosterierX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
			EYY, EY, Eyy, Ey, EYy = GetPosterierY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
		else:
			EXX, EX, Exx, Ex, EXx = GetPosterierX( Vc, Vd, Ve, V, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
			EYY, EY, Eyy, Ey, EYy = GetPosterierY( Vc, Vd, Ve, V, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
		K1, Theta1 = GetPosterierPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model )
		Args = (EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1, Alphax, Alphay, Betax, Betay, M, N, m, n, M1, N1, m1, n1, M11, N11, m11, n11, Psi, Psi1, Psi11, PsiE,  x_, y_, Phi_, Model, VasVariable)
		print X0
		Y0 = deepcopy(X0)
		Y0[:-1] = Y0[:-1]**.5
		CurrentCost = Fvalue(Y0, *Args)
		CurrentCostXY = FvalueXY(Y0[:4], *Args)
		CurrentCostPhi = FvaluePhi(Y0[4:], *Args)
		DecreaseInCost = PrevCost - CurrentCost
		DecreaseInCostXY = PrevCostXY - CurrentCostXY
		DecreaseInCostPhi = PrevCostPhi - CurrentCostPhi
		PrevCost = CurrentCost
		PrevCostXY = CurrentCostXY
		PrevCostPhi = CurrentCostPhi
		
		print 'FvalueAfterPost:'+str(CurrentCost)
		print 'FvalueXY:'+str(CurrentCostXY)
		print 'FvaluePhi:'+str(CurrentCostPhi)
		print 'DecreaseInCost:'+str(DecreaseInCost)
		print 'DecreaseInCostXY:'+str(DecreaseInCostXY)
		print 'DecreaseInCostPhi:'+str(DecreaseInCostPhi)

		if PrevIterationForExpectation:
			if DecreaseInCost+DecreaseInCostV<2.0:
				break
			else:
				DecreaseInCostV = 0.0
		
		DecreaseInCostV += DecreaseInCost
		print
		print 'iteration:'+str(iteration)
		#Optimizing the Parameters
		X0 = Optimize( X0, Args )
		[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0] = X0
		print 'Optimized Parameters'
		print '[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0]:'
		print X0
		Y0 = deepcopy(X0)
		Y0[:-1] = Y0[:-1]**.5
		CurrentCost = Fvalue(Y0, *Args)
		CurrentCostXY = FvalueXY(Y0[:4], *Args)
		CurrentCostPhi = FvaluePhi(Y0[4:], *Args)
		DecreaseInCost = PrevCost - CurrentCost
		DecreaseInCostXY = PrevCostXY - CurrentCostXY
		DecreaseInCostPhi = PrevCostPhi - CurrentCostPhi
		PrevCost = CurrentCost
		PrevCostXY = CurrentCostXY
		PrevCostPhi = CurrentCostPhi

		print 'FvalueAfterOptmzn:'+str(CurrentCost)
		print 'FvalueXY:'+str(CurrentCostXY)
		print 'FvaluePhi:'+str(CurrentCostPhi)
		print 'DecreaseInCost:'+str(DecreaseInCost)
		print 'DecreaseInCostXY:'+str(DecreaseInCostXY)
		print 'DecreaseInCostPhi:'+str(DecreaseInCostPhi)
		
		DecreaseInCostV +=DecreaseInCost

		if DecreaseInCost<ThV:
			if VasVariable:
				X, x = GetMapX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
			else:
				X, x = GetMapX( Vc, Vd, Ve, V, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
			AlphaPhi = V2M(X)-x
			Betay = -AlphaPhi*Alphax

			if VasVariable:
				Y, y = GetMapY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
			else:
				Y, y = GetMapY( Vc, Vd, Ve, V, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
			BetaPhi = V2M(Y)-y
			Betax = BetaPhi*Alphay

			PsiE = arctan2( AlphaPhi,BetaPhi )%tao
			Phi = GetMapPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model )
			Alphax = np.cos(Phi)
			Alphay = np.sin(Phi)
			Betax = BetaPhi*Alphay
			Betay = -AlphaPhi*Alphax

			M, m = GetLocationMeans( X, x, Nc, T, FiltLen )
			M1, m1 = GetLocationMeansPrime( X, x, Nc, T, FiltLen )
			M11, m11 = GetLocationMeansPrimePrime( X, x, Nc, T, FiltLen )
			N, n = GetLocationMeans( Y, y, Nc, T, FiltLen )
			N1, n1 = GetLocationMeansPrime( Y, y, Nc, T, FiltLen )
			N11, n11 = GetLocationMeansPrimePrime( Y, y, Nc, T, FiltLen )
			Psi = GetOrientationMeans( X, Y, x, y, Phi, Nc, T, FiltLen )
			Psi1 = GetOrientationMeansPrime( X, Y, x, y, Phi, Nc, T, FiltLen )
			Psi11 = GetOrientationMeansPrimePrime( X, Y, x, y, Phi, Nc, T, FiltLen )
			PrevIterationForExpectation = True
	return X, Y, x, y, Phi, Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, CurrentCost, iteration #end of Algo3()

def Algo4( InitialParameters, x_, y_, Phi_, V, H, Nc, T, MaxIterations, Model, VasVariable ):
	'''
	Multiple iterations for variances and expectations!
	'''
	print 'Executing Algorithm4!'
	[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, X, Y, x, y, Phi] = InitialParameters
	X0 = np.asarray([Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0])
	M, m = GetLocationMeans( X, x, Nc, T, FiltLen )
	M1, m1 = GetLocationMeansPrime( X, x, Nc, T, FiltLen )
	M11, m11 = GetLocationMeansPrimePrime( X, x, Nc, T, FiltLen )
	N, n = GetLocationMeans( Y, y, Nc, T, FiltLen )
	N1, n1 = GetLocationMeansPrime( Y, y, Nc, T, FiltLen )
	N11, n11 = GetLocationMeansPrimePrime( Y, y, Nc, T, FiltLen )
	Psi = GetOrientationMeans( X, Y, x, y, Phi, Nc, T, FiltLen )
	Psi1 = GetOrientationMeansPrime( X, Y, x, y, Phi, Nc, T, FiltLen )
	Psi11 = GetOrientationMeansPrimePrime( X, Y, x, y, Phi, Nc, T, FiltLen )

	AlphaPhi = V2M(X)-x
	BetaPhi = V2M(Y)-y
	Alphax = np.cos(Phi)
	Alphay = np.sin(Phi)
	Betax = BetaPhi*Alphay
	Betay = -AlphaPhi*Alphax
	PsiE = arctan2( AlphaPhi, BetaPhi )%tao
	PrevIterationForExpectation = False
	PrevCost = float('inf')
	PrevCostPhi = float('inf')
	PrevCostXY = float('inf')
	DecreaseInCost = float('inf')
	DecreaseInCostPhi = float('inf')
	DecreaseInCostXY = float('inf')
	DecreaseInCostE = float('inf')
	DecreaseInCostV = float('inf')
	for iteration in range(MaxIterations):
		print 
		print 'Iteration:'+str(iteration)
		#Computing expectations for likelihood
		if VasVariable:
			EXX, EX, Exx, Ex, EXx = GetPosterierX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
			EYY, EY, Eyy, Ey, EYy = GetPosterierY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
		else:
			EXX, EX, Exx, Ex, EXx = GetPosterierX( Vc, Vd, Ve, V, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
			EYY, EY, Eyy, Ey, EYy = GetPosterierY( Vc, Vd, Ve, V, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
		K1, Theta1 = GetPosterierPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model )
		Args = (EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1, Alphax, Alphay, Betax, Betay, M, N, m, n, M1, N1, m1, n1, M11, N11, m11, n11, Psi, Psi1, Psi11, PsiE,  x_, y_, Phi_, Model, VasVariable)
		print X0
		Y0 = deepcopy(X0)
		Y0[:-1] = Y0[:-1]**.5
	
		CurrentCost = Fvalue(Y0, *Args)
		CurrentCostXY = FvalueXY(Y0[:4], *Args)
		CurrentCostPhi = FvaluePhi(Y0[4:], *Args)
		DecreaseInCost = PrevCost - CurrentCost
		DecreaseInCostXY = PrevCostXY - CurrentCostXY
		DecreaseInCostPhi = PrevCostPhi - CurrentCostPhi
		PrevCost = CurrentCost
		PrevCostXY = CurrentCostXY
		PrevCostPhi = CurrentCostPhi

		print 'FvalueAfterPost:'+str(CurrentCost)
		print 'FvalueXY:'+str(CurrentCostXY)
		print 'FvaluePhi:'+str(CurrentCostPhi)
		print 'DecreaseInCost:'+str(DecreaseInCost)
		print 'DecreaseInCostXY:'+str(DecreaseInCostXY)
		print 'DecreaseInCostPhi:'+str(DecreaseInCostPhi)
		
		DecreaseInCostV += DecreaseInCost
		#Optimizing the parameters
		print
		print 'iteration:'+str(iteration)
		X0 = Optimize( X0, Args )
		[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0] = X0
		print 'Optimized Parameters'
		print '[Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0]:'
		print X0
		Y0 = deepcopy(X0)
		Y0[:-1] = Y0[:-1]**.5
		CurrentCost = Fvalue(Y0, *Args)
		CurrentCostXY = FvalueXY(Y0[:4], *Args)
		CurrentCostPhi = FvaluePhi(Y0[4:], *Args)
		DecreaseInCost = PrevCost - CurrentCost
		DecreaseInCostXY = PrevCostXY - CurrentCostXY
		DecreaseInCostPhi = PrevCostPhi - CurrentCostPhi
		PrevCost = CurrentCost
		PrevCostXY = CurrentCostXY
		PrevCostPhi = CurrentCostPhi

		print 'FvalueAfterOptmzn:'+str(CurrentCost)
		print 'FvalueXY:'+str(CurrentCostXY)
		print 'FvaluePhi:'+str(CurrentCostPhi)
		print 'DecreaseInCost:'+str(DecreaseInCost)
		print 'DecreaseInCostXY:'+str(DecreaseInCostXY)
		print 'DecreaseInCostPhi:'+str(DecreaseInCostPhi)
		
		DecreaseInCostV += DecreaseInCost

		if DecreaseInCost<ThV:
			ExpectationConvergence = False
			while not ExpectationConvergence:
				if VasVariable:
					X, x = GetMapX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
				else:
					X, x = GetMapX( Vc, Vd, Ve, V, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
				AlphaPhi = V2M(X)-x
				Betay = -AlphaPhi*Alphax

				if VasVariable:
					Y, y = GetMapY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
				else:
					Y, y = GetMapY( Vc, Vd, Ve, V, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
				BetaPhi = V2M(Y)-y
				Betax = BetaPhi*Alphay

				PsiE = arctan2( AlphaPhi,BetaPhi )%tao
				Phi = GetMapPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model )
				Alphax = np.cos(Phi)
				Alphay = np.sin(Phi)
				Betax = BetaPhi*Alphay
				Betay = -AlphaPhi*Alphax

				M, m = GetLocationMeans( X, x, Nc, T, FiltLen )
				M1, m1 = GetLocationMeansPrime( X, x, Nc, T, FiltLen )
				M11, m11 = GetLocationMeansPrimePrime( X, x, Nc, T, FiltLen )
				N, n = GetLocationMeans( Y, y, Nc, T, FiltLen )
				N1, n1 = GetLocationMeansPrime( Y, y, Nc, T, FiltLen )
				N11, n11 = GetLocationMeansPrimePrime( Y, y, Nc, T, FiltLen )
				Psi = GetOrientationMeans( X, Y, x, y, Phi, Nc, T, FiltLen )
				Psi1 = GetOrientationMeansPrime( X, Y, x, y, Phi, Nc, T, FiltLen )
				Psi11 = GetOrientationMeansPrimePrime( X, Y, x, y, Phi, Nc, T, FiltLen )
				if VasVariable:
					EXX, EX, Exx, Ex, EXx = GetPosterierX( Vc, Vd, Ve, Vo, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
					EYY, EY, Eyy, Ey, EYy = GetPosterierY( Vc, Vd, Ve, Vo, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
				else:
					EXX, EX, Exx, Ex, EXx = GetPosterierX( Vc, Vd, Ve, V, Alphax, Betax, M, m, M1, m1, M11, m11, x_, Model )
					EYY, EY, Eyy, Ey, EYy = GetPosterierY( Vc, Vd, Ve, V, Alphay, Betay, N, n, N1, n1, N11, n11, y_, Model )
				K1, Theta1 = GetPosterierPhi( Kc, Kd, Kh1, Kh0, Psi, Psi1, Psi11, PsiE, Phi_, H, Model )
				Args = (EXX, EX, Exx, Ex, EXx, EYY, EY, Eyy, Ey, EYy, K1, Theta1, Alphax, Alphay, Betax, Betay, M, N, m, n, M1, N1, m1, n1, M11, N11, m11, n11, Psi, Psi1, Psi11, PsiE,  x_, y_, Phi_, Model, VasVariable)
				print X0
				Y0 = deepcopy(X0)
				Y0[:-1] = Y0[:-1]**.5
				CurrentCost = Fvalue(Y0, *Args)
				CurrentCostXY = FvalueXY(Y0[:4], *Args)
				CurrentCostPhi = FvaluePhi(Y0[4:], *Args)
				DecreaseInCost = PrevCost - CurrentCost
				DecreaseInCostXY = PrevCostXY - CurrentCostXY
				DecreaseInCostPhi = PrevCostPhi - CurrentCostPhi
				PrevCost = CurrentCost
				PrevCostXY = CurrentCostXY
				PrevCostPhi = CurrentCostPhi

				print 'FvalueAfterPost:'+str(CurrentCost)
				print 'FvalueXY:'+str(CurrentCostXY)
				print 'FvaluePhi:'+str(CurrentCostPhi)
				print 'DecreaseInCost:'+str(DecreaseInCost)
				print 'DecreaseInCostXY:'+str(DecreaseInCostXY)
				print 'DecreaseInCostPhi:'+str(DecreaseInCostPhi)
		
				DecreaseInCostE += DecreaseInCost
				if DecreaseInCost<ThE:
					ExpectationConvergence = True
				if DecreaseInCostE+DecreaseInCostV<2.0:
					break
			DecreaseInCostV = 0.0
			DecreaseInCostE = 0.0
	return X, Y, x, y, Phi, Vc, Vd, Ve, Vo, Kc, Kd, Kh1, Kh0, CurrentCost, iteration #end of Algo4()

