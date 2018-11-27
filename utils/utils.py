
'''
This file includes all the libraries to be imported into respective modules.
Also this file includes all the parameters to be initialized, which are in turn used in the subsequent modules!
'''

#Import following into every file!
from __future__ import division
#156543.03392*f.np.cos(f.np.radians(lat))/2**Zoom
#(f.np.cos(f.np.radians(lat))*2*f.np.pi*6378137)/(256*2**Zoom)
#http://abel.ee.ucla.edu/cvxopt/documentation/: reference for cvxopt interior point method
'''
All calculations are done with x axis being horizontal axis 
and y axis being vertical axis
While presenting results as an image or video we must consider this setup
as Images and videos will have x axis as vertical axis
Yaw is considered wrt North +ve in clockwise direction
'''
import re
import json
from copy import deepcopy
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import *
import os
import time
import datetime
import subprocess
import math
import random
#import SimpleITK
import os
import shutil
import Queue
import requests
from skimage import draw as SkimageDraw
import cv2

from cvxopt import matrix as cvxmatrix
from cvxopt import spdiag
from cvxopt import sparse as cvxsparse
from cvxopt import solvers
from cvxopt.solvers import cp
from cvxopt import mul
from cvxopt import div
from cvxopt import sin as Sin
from cvxopt import cos as Cos
from cvxopt import normal as Normal
from cvxopt import uniform as Uniform
from cvxopt import sqrt as Sqrt

import warnings
from operator import itemgetter
from collections import OrderedDict
import pickle
from PIL import Image
from StringIO import StringIO

np.set_printoptions(precision = 12)
#np.set_printoptions(threshold = np.nan)
np.set_printoptions(threshold = np.inf)

#Parameter definitions!

Resolution = 'Meter'

BlackBackGround = True
BlackBackGround = False

DummySolution = True
DummySolution = False

CamParaEstimation = False
CamParaEstimation = True

#Resolution Improvement Factor 
if Resolution == 'Meter':
	ImprovementFactor = 1.
if Resolution == 'Quartermeter':
	#Unit for new score is 1 pixel = 25 centimetes
	ImprovementFactor = 4.
if Resolution == 'Decimeter':
	#Unit for new score is 1 pixel = 10 centimetes
	ImprovementFactor = 10.
if Resolution == 'Centimeter':
	#Unit for new score is 1 pixel = 1 centimetes
	ImprovementFactor = 100.

#MaxGpsError = 50					#Meters
#MaxGpsError = 25					#meters
MaxGpsError = 15					#meters
MaxVisibleDistance = 100 				#meters
#MaxVisibleDistance = 50 				#meters
#MaxVisibleDistance = 150 				#meters
MinVisibleDistance = 5				#meters
ScoreSize = 400 						#meters
ScoreSizeBy2 = 200 					#meters
GridSize = 10						#meters
ActivityScoreRadius = 5				#10 meters wide spot
ActivityScoreRadiusSmall = 2			#4 meters wide spot
EarthRadius = 6371000 				#Mean earth radius in meters
ArrowLength = int(MaxVisibleDistance/8)

MaxGpsError *= 1							#ie 25 meters
MaxDistance = MaxVisibleDistance*ImprovementFactor 		#

#############
ScoreSize *= int(ImprovementFactor) 				#ie 400 meters
ScoreSizeBy2 *= int(ImprovementFactor) 			#ie 200 meters
GridSize *= int(ImprovementFactor)				#ie 10 meters size boxes
ActivityScoreRadius *= 1						#ie 10 pixel wide spot
ActivityScoreRadiusSmall *= 1				#ie 4 pixel wide spot
ArrowLength *= int(ImprovementFactor)
#######################
ArrowHeadAngle = 30
ActivityIntensity = 250
ScoreIntensity = 250
CompassErrorIntensity = 100
GpsErrorIntensity = 250
CropBlocks = 10
CropPixels = GridSize*CropBlocks

if not ScoreSizeBy2 == round(ScoreSize/2):
	raise ValueError('ScoreSize and ScoreSizeBy2 are not consistent')

CompleteVideoProcessing = False
#CompleteVideoProcessing = True

#RawScore = True
Path = False
CompassFreq1Hz = True
#AddConstraint = True

#LineScore = True
#LineScore = False

ProbOptimization = True
#Parameters definition ends here
session = requests.Session()
session.trust_env=False
session.mount("http://", requests.adapters.HTTPAdapter(max_retries=100))
#session.get(url="https://www.service-that-drops-every-odd-request.com/")
import threading

#global variables declaration!
#global GroundTruthAvailable
#GroundTruthAvailable = False
global QueryName
global ScoreTemplate
global ActivityScore
global ActivityScoreSmall
global CompassErrorTemplate
'''
This module contains all the basic funcitons definitions as well as some standard variables used throughout!
'''
def MyNorm(x,y):
	return math.sqrt(x**2+y**2)

def MyDegreeATan2(y,x,Boolean):
	'''
	Returns angles in the range of 0 - 360
	'''
	if Boolean:
		#return math.degrees(math.atan2(y,x))
		Angle = math.degrees(math.atan2(y,x))
		if Angle < 0 :
			Angle = Angle + 360
		return Angle
	else:
		return 0

def AngleDiff(DiffAngle, Boolean):
	if Boolean:
		#return min(abs(DiffAngle + 180 + 360), abs(DiffAngle + 180), abs(DiffAngle + 180 - 360))
		return min(abs(DiffAngle),abs(abs(DiffAngle) - 360))
	else:
		return 0
	
	#In our case, camera_angle is Yaw & point_angle is angle of given point ie CamTempletAngle

def GetCameraScore(Distance, DiffAngle,DiffAngleMax=None):
	if DiffAngleMax == None:
		DiffAngleMax = 2
	if Distance<=MaxDistance and Distance >= MinVisibleDistance and DiffAngle <= DiffAngleMax:
		#return GetBetaDistributionSample(Distance/MaxDistance,2.5,4)*GetBetaDistributionSample(0.5+DiffAngle/60,2,2)
		#return GetBetaDistributionSample(Distance/MaxDistance,2.5,4)*GetBetaDistributionSample(0.5+DiffAngle/60,1.5,1.5)
		return 1
		#return 5*math.exp(-math.log(4)/MaxDistance*Distance)*(math.cos(math.radians(DiffAngle)) - 0.8213672050459183)/0.17863279495408171
	else:
		return 0.0

#Following are array vectorized functions
MyMatrixNorm = np.vectorize(MyNorm)
MyMatrixDegreeATan2 = np.vectorize(MyDegreeATan2)
MyMatrixAngleDiff = np.vectorize(AngleDiff)
MyMatrixCameraScore = np.vectorize(GetCameraScore)

[CamTemplX,CamTemplY] = np.mgrid[-ScoreSizeBy2:ScoreSizeBy2,-ScoreSizeBy2:ScoreSizeBy2]
[GpsTempletX,GpsTempletY] = np.mgrid[-ScoreSizeBy2:ScoreSizeBy2,-ScoreSizeBy2:ScoreSizeBy2]
	
CamTempletDistance = MyMatrixNorm(CamTemplX,CamTemplY)
CamTempletBoolean = MyMatrixNorm(CamTemplX,CamTemplY)
CamTempletBoolean[CamTempletBoolean<=MaxDistance] = 1
CamTempletBoolean[CamTempletBoolean>MaxDistance] = 0	
CamTempletAngle = np.roll(np.flipud(MyMatrixDegreeATan2(CamTemplY,CamTemplX,CamTempletBoolean)),1,axis=0)
CamTempletAngle2 = np.roll(np.flipud(MyMatrixDegreeATan2(CamTemplY,CamTemplX,1)),1,axis=0)

GpsTempletScoreTemp = MyMatrixNorm(GpsTempletX,GpsTempletY)
ScoreGrid = np.zeros([ScoreSize,ScoreSize])
ScoreGrid[GridSize:-1:GridSize,:]=1
ScoreGrid[:,GridSize:-1:GridSize]=1
ScoreGrid = 50*ScoreGrid

def GetNormalTime(EpochTime):
	#return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(EpochTime/1000))
	return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(EpochTime/1000))

def GetEpochTime(YMD_HMS):
	return 1000*time.mktime(time.strptime(YMD_HMS,"%Y-%m-%d %H:%M:%S"))

def Array3D2Video(Array3D,OutName=None,Length=None):
	print 'Creating video from 3D array'
	OutFrameRate = int(round(1000/200))
	fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
	if OutName==None:
		OutName = '3DArray.avi'
	else:
		OutName+='.avi'
	print 'Output Video File Name: '+str(OutName)
	[Height,Width] = np.shape(Array3D[0])
	TempScore = np.empty([Height,Width,3],'uint8')
	vidput = cv2.VideoWriter(OutName,fourcc,OutFrameRate,(Width,Height))
	if Length==None:
		Length = len(Array3D)
	for i in range(Length):
		print 'Creating frame '+str(i)
		TempScore[:,:,0] = Array3D[i]
		TempScore[:,:,1] = Array3D[i]
		TempScore[:,:,2] = Array3D[i]
		#if i%5==0:
		#	p(Array3D[i])
		#MyCurrentFrame = TempScore.astype('u1')
		MyCurrentFrame = TempScore
		vidput.write(MyCurrentFrame)
	vidput.release()
	return

def MedFilt(Array,Len):
	return signal.medfilt(Array,Len)

def AvgFilt(Array,Len):
	Indx = int((Len-1)/2)
	ArrayOut = deepcopy(Array)
	ArrayOut[Indx:-Indx] = (signal.convolve(Array,np.ones(Len)))[(Len-1):-(Len-1)]/(1.0*Len)
	return ArrayOut

def FilterAccelData(AccelData):
	Len = 9
	AccelDataOut = np.zeros(np.shape(AccelData))

	#Median Filtering
	for col in range(3):
		#AccelDataOut[:,col] = signal.medfilt(AccelData[:,col],Len)
		AccelDataOut[:,col] = MedFilt(AccelData[:,col],Len)
	#Indx = (Len-1)/2
	#LowPassFiltering
	for col in range(3):
		#AccelDataOut[:,col] = (signal.convolve(AccelDataOut[:,col],np.ones(Len)))[(Len-1)/2:-(Len-1)/2]/(1.0*Len)
		#AccelDataOut[Indx:-Indx,col] = (signal.convolve(AccelDataOut[:,col],np.ones(Len)))[(Len-1):-(Len-1)]/(1.0*Len)
		#AccelDataOut[:,col] = (signal.convolve(AccelData[:,col],np.ones(Len)))[(Len-1)/2:-(Len-1)/2]/(1.0*Len)
		AccelDataOut[:,col] = AvgFilt(AccelDataOut[:,col],Len)
	#Recomputing Magnitude
	AccelDataOut[:,3] = np.asarray(map(lambda x:np.linalg.norm(x[0:3]),AccelDataOut))
	return AccelDataOut

def FilterMagneticData(MagneticData):
	Len = 9
	#Indx = (Len-1)/2
	MagneticDataOut = np.zeros(np.shape(MagneticData))
	#Median Filtering
	for col in range(3):
		#MagneticDataOut[Indx:-Indx,col] = (signal.convolve(MagneticData[:,col],np.ones(Len)))[(Len-1):-(Len-1)]/(1.0*Len)
		MagneticDataOut[:,col] = MedFilt(MagneticData[:,col],Len)
	#LowPassFiltering
	for col in range(3):
		#MagneticDataOut[Indx:-Indx,col] = (signal.convolve(MagneticData[:,col],np.ones(Len)))[(Len-1):-(Len-1)]/(1.0*Len)
		#MagneticDataOut[:,col] = AvgFilt(MagneticData[:,col],Len)
		MagneticDataOut[:,col] = AvgFilt(MagneticDataOut[:,col],Len)
	#Recomputing Magnitude
	MagneticDataOut[:,3] = np.asarray(map(lambda x:np.linalg.norm(x[0:3]),MagneticDataOut))
	return MagneticDataOut

def GetMagneticData(CompassData):
	MagneticData = np.asarray(map(lambda x: [x[8],x[9],x[10],np.linalg.norm(x[8:11])],CompassData))
	return MagneticData

def GetAccelerometerData(CompassData):
	AccelData = np.asarray(map(lambda x:[x[5],x[6],x[7],np.linalg.norm(x[5:8])],CompassData))
	return AccelData

def GetLayout(k):
	'''
	There are k videos available
	'''
	if k>=10:
		raise ValueError('Too many videos to display!')
	LayOut = [[2,1],[3,1],[2,2],[3,2],[3,2],[4,2],[4,2],[5,2],[5,2]]
	return LayOut[k]

def GetTimelineChart(videoList):
	#MinTime = videoList[0].sensor.TYXYE[0][0]
	#MaxTime = -1
	MinTime = QueryStartTime
	MaxTime = QueryStopTime
	for video in videoList:
		MinTime = min(MinTime,video.sensor.TYXYE[0][0])
		MaxTime = max(MaxTime,video.sensor.TYXYE[len(video.sensor.TYXYE)-1][0])
		#MinTime = min(MinTime,video.sensor.gps[0][0]+start_timestamp)
		#MaxTime = max(MaxTime,video.sensor.gps[len(video.sensor.gps)-1][0]+start_timestamp)
	#Time = range(MinTime,MaxTime,int(Period))

	QueryX = [QueryStartTime,QueryStartTime, QueryStopTime,QueryStopTime]
	#QueryY = [QueryStartTime,1,1,QueryStopTime]
	QueryY = [0,1,1,0]
	plt.close('all')
	plt.plot(QueryX,QueryY)
	for i in range(len(videoList)):
		#print i
		#print len(videoList[i].sensor.TYXYE)
		VidX = [videoList[i].sensor.TYXYE[0][0],videoList[i].sensor.TYXYE[0][0],videoList[i].sensor.TYXYE[len(videoList[i].sensor.TYXYE)-1][0],videoList[i].sensor.TYXYE[len(videoList[i].sensor.TYXYE)-1][0]]
		start_timestamp = int(video.info['start_timestamp'])
		#VidX = [videoList[i].sensor.gps[0][0]+start_timestamp,videoList[i].sensor.gps[0][0]+start_timestamp,videoList[i].sensor.gps[len(videoList[i].sensor.gps)-1][0]+start_timestamp,videoList[i].sensor.gps[len(videoList[i].sensor.gps)-1][0]+start_timestamp]
		#VidY = [videoList[i].sensor.TYXYE[0][0],i+2, i+2,videoList[i].sensor.TYXYE[len(videoList[i].sensor.TYXYE)-1][0]]
		VidY = [0,i+2, i+2,0]
		plt.plot(VidX,VidY)
		plt.plot(VidX,VidY)
	plt.axis([MinTime-200,MaxTime+200,0,i+3])
	plt.grid()
	plt.show()
	return #End timelinechart

def GetTimelineChart2(videoList,QueryStartTime=None,QueryStopTime=None):

	if not QueryStartTime == None:
		MinTime = QueryStartTime
	else:
		MinTime = int(videoList[0].info['start_timestamp'])

	if not QueryStopTime == None:
		MaxTime = QueryStopTime
	else:
		MaxTime = int(videoList[0].info['start_timestamp'])+int(videoList[0].info['duration'])

	for video in videoList:
		MinTime = min(MinTime,int(video.info['start_timestamp']))
		MaxTime = max(MaxTime,int(video.info['start_timestamp'])+int(video.info['duration']))

	QueryX = [QueryStartTime,QueryStartTime, QueryStopTime,QueryStopTime]
	QueryY = [0,1,1,0]
	plt.close('all')
	plt.plot(QueryX,QueryY)
	for i in range(len(videoList)):
		StartTime = int(videoList[i].info['start_timestamp'])
		StopTime = int(videoList[i].info['start_timestamp'])+int(videoList[i].info['duration'])
		VidX = [StartTime,StartTime,StopTime,StopTime]
		start_timestamp = int(video.info['start_timestamp'])
		VidY = [0,i+2, i+2,0]
		plt.plot(VidX,VidY)
		plt.plot(VidX,VidY)
	plt.axis([MinTime-200,MaxTime+200,0,i+3])
	plt.grid()
	plt.show()
	return # End GetTimelineChart2

def GPSDegreeToMeters(gps):
	'''
	Note: Maximum error in distance is 1cm
	This function accepts list of Latitude Longitude and returns distance in
	meters as per haversine formulae.
	Radius of Earth is assumed to be 6371000 meters.
	Refer following link for haversine formulae
	http://www.movable-type.co.uk/scripts/latlong.html
	or look on wikipedia
	'''
	TXYE = map(HaversineDistance,gps)
	
	return TXYE

def HaversineDistance(GPSData):
	'''
	This function returns Great Circle Distance of given GPS 
	location with that of 0 Latitude, 0 Longitude
	In general, for points at (Lat1, Long1) & (Lat2, Long2)
	we have 
	X = Distance between (Lat1, Long1) & (Lat1, Long2)
	Y = Distance between (Lat1, Long1) & (Lat2, Long1)
	'''
	[Time, Latitude, Longitude, GPSError, Altitude] = GPSData
	Lat2 = Latitude
	Long2 = Longitude
	Lat1 = 0
	Long1 = 0
	
	#X = HaversineDistanceBetweenTwoPoints(Lat1, Long1, Lat1, Long2)
	#Y = HaversineDistanceBetweenTwoPoints(Lat1, Long1, Lat2, Long1)

	X = np.radians(Longitude)*EarthRadius
	Y = np.radians(Latitude)*EarthRadius
	
	return [Time, X, Y, GPSError]

def HaversineDistanceBetweenTwoPoints(Lat1, Long1, Lat2, Long2):
	'''
	Computes and returns Great Circle Distance between given two points
	'''
	#Degress to Radiance
	[Lat1, Long1, Lat2, Long2] = map(math.radians,[Lat1, Long1, Lat2, Long2])
	DelLat = Lat2-Lat1
	DelLong = Long2-Long1
	A = math.sin(DelLat/2)**2 + math.cos(Lat1) * math.cos(Lat2) * math.sin(DelLong/2)**2
	Distance = 2*EarthRadius*math.asin(math.sqrt(A)) 
	#OR
	#Distance = 2*EarthRadius* math.atan2(math.sqrt(A),math.sqrt(1-A)) 
	return Distance

def CheckForGpsError(TYXYE):
	#Lets check if ScoreSize & Maximum GPS error are consistent
	if (MaxGpsError)<=max(TYXYE[:,4]):
		print 'GPS Error beyond Allowed range!'
		Index = np.argmax(TYXYE[:,4])
		#MaxError = max(TYXYE[:,4])
		MaxError = TYXYE[Index,4]

		while MaxError>=MaxGpsError:
			print 'Maximum allowed GPS Error is '+str(MaxGpsError)
			print 'GPS Error at index '+str(Index)+' is '+str(MaxError)
			#print 'Replacing GPS Error value at index '+str(Index)+' by '+str(MaxGpsError-1)
			#TYXYE[Index][4] = MaxGpsError-1
			print 'Replacing X & Y Co-ordinates by previous or next values'
			print 'Old values: X='+str(TYXYE[Index][2])+' Y='+str(TYXYE[Index][3])
			OffSet = 1
			while TYXYE[Index][4]>=MaxGpsError:
				if (Index-OffSet)>=0:
					PrevIndex = Index-OffSet
				else:
					PrevIndex = 0
				if (Index+OffSet)<len(TYXYE):
					NextIndex = Index+OffSet
				else:
					NextIndex = len(TYXYE)-1
				if TYXYE[PrevIndex][4]<TYXYE[NextIndex][4]:
					NewIndex = PrevIndex
				else:
					NewIndex = NextIndex
				TYXYE[Index][2] = TYXYE[NewIndex][2]
				TYXYE[Index][3] = TYXYE[NewIndex][3]
				TYXYE[Index][4] = TYXYE[NewIndex][4]
				TYXYE[Index][5] = TYXYE[NewIndex][5]
				TYXYE[Index][6] = TYXYE[NewIndex][6]
				OffSet+=1
			
			print 'New values: X='+str(TYXYE[Index][2])+' Y='+str(TYXYE[Index][3])
			#MaxError = max(TYXYE[:,4])
			Index = np.argmax(TYXYE[:,4])
			MaxError = TYXYE[Index,4]
		#raise ValueError('GPS Error is Maximum')
	return TYXYE

def GetMedXandY(VideoList):
	MedX = []
	MedY = []
	for vid in VideoList:
		MedX.append(vid.sensor.TYXYE[:,2])
		MedY.append(vid.sensor.TYXYE[:,3])
	MedX=np.asarray(MedX).reshape(1,-1)
	MedY=np.asarray(MedY).reshape(1,-1)
	MedX = np.median(MedX[0,0])
	MedY = np.median(MedY[0,0])
	return [MedX,MedY]

def GetMeanXandY(VideoList):
	meanX = []
	meanY = []
	for vid in VideoList:
		meanX.append(np.mean(vid.sensor.TYXYE[:,2]))
		meanY.append(np.mean(vid.sensor.TYXYE[:,3]))

	meanX=np.array(meanX)
	meanY=np.array(meanY)
	return [np.mean(meanX),np.mean(meanY)]

def isEventTooFar( X, Y, x, y ):
	d = ((X-x)**2+(Y-y)**2)**.5
	if np.max(d)>MaxVisibleDistance:
		print 'Event is too far away from the camera!'
		return True
	else:
		return False
			
def TestDistance(gps):
	'''
	This function is written to test accuracy of the function GPSDegreeToMeters.
	It is observed that accuracy is no poor than 1cm. :)
	'''
	Len = len(gps)
	D = np.ndarray([Len,Len],list)
	D2 = np.ndarray([Len,Len],list)
	for i in range(Len):
		for j in range(Len):
			
			[x1,y1,x2,y2] = [gps[i][1],gps[i][2],gps[j][1],gps[j][2]]
			d1 = HaversineDistanceBetweenTwoPoints(x1,y1,x2,y2)
			x = HaversineDistanceBetweenTwoPoints(x1,y1,x2,y1)
			y = HaversineDistanceBetweenTwoPoints(x1,y1,x1,y2)
			d2 = math.sqrt(x*x+y*y)
			
			[x1,y1,x2,y2] = [0,0,gps[i][1],gps[i][2]]
			xx1 = HaversineDistanceBetweenTwoPoints(x1,y1,x2,y1)
			yy1 = HaversineDistanceBetweenTwoPoints(x1,y1,x1,y2)
			[x1,y1,x2,y2] = [0,0,gps[j][1],gps[j][2]]
			xx2 = HaversineDistanceBetweenTwoPoints(x1,y1,x2,y1)
			yy2 = HaversineDistanceBetweenTwoPoints(x1,y1,x1,y2)
			d3 = math.sqrt((xx1-xx2)*(xx1-xx2)+(yy1-yy2)*(yy1-yy2))

			#D[i][j].append(d1)
			#D[i][j].append(d2)
			#D[i][j].append(d3)
			[d1,d2,d3] = [int(100*d1),int(100*d2),int(100*d3)]
			D[i][j] = [d1,d2,d3]
			D2[i][j] = [d2-d1,d3-d1]
	return D2

def GetVideoTimelineIndicator(VideoList):
	VidLen = len(VideoList)
	VideoTimeLineIndicator = []
	VideoFrameList = []
	Len = len(VideoList)
	if GroundTruthAvailable:
		Len=Len-1
	for i in range(Len):
		print i
		video = VideoList[i]
		DirName = 'Video/VideoFrames/'+str(video.id)+'/'
		FrameList = os.listdir(DirName)
		#FrameList = map(lambda FrameName: video.start_time+int(video.info['start_timestamp'])+int(FrameName[4:-4]),FrameList)
		FrameList = map(lambda FrameName: int(FrameName[4:-4]),FrameList)
		VideoFrameList.append(FrameList)
	for Time in TimeLineIndicator[:,0]:
		CurrentTimeLine=[]
		CurrentTimeLine.append(Time)
		for i in range(Len):
			video = VideoList[i]
			MinIndx = np.argmin(abs(VideoFrameList[i]+(video.start_time+int(video.info['start_timestamp'])-Time)))
			CurrentTimeLine.append(VideoFrameList[i][MinIndx])
		VideoTimeLineIndicator.append(CurrentTimeLine)
	VideoTimeLineIndicator = np.asarray(VideoTimeLineIndicator)
	return VideoTimeLineIndicator

def DisplayVideoInfo(videoList):
	for vid in videoList:
		print 'video start time from query:'+str(GetNormalTime(vid.start_epoch_time))
		print 'video start time from info:'+str(GetNormalTime(int(vid.info['start_timestamp'])))
		print 'Video Seg Start:'+str(vid.start_time)
		print 'Video Seg Stop:'+str(vid.stop_time)
		print 'Video gps start:'+str(vid.sensor.gps[0,0])
		print 'Video gps end:'+str(vid.sensor.gps[-1,0])
		print 'Video Compass start:'+str(vid.sensor.Compass[0,0])
		print 'Video Compass end:'+str(vid.sensor.Compass[-1,0])
		print '\n'

def MyAnimation(Array3D, Title='NoTitle', FrameTitle=None,dpi=None, Vmax=None):
	'''
	This function accepts 3D array as input and stores it as a video animation.
	1st dimension is time, remaining are spatial dimensions.
	'''
	if Vmax==None:
		#Vmax = np.max(Array3D)	
		Vmax = 7
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#im = ax.imshow(self.score[0],cmap='jet', interpolation='nearest', origin='lower' )
	#im = ax.imshow(Array3D[0],cmap='jet', interpolation='nearest', origin='lower' )
	im = ax.imshow(Array3D[0].toarray(),cmap='jet', interpolation='nearest', origin='lower' )
	im.set_clim(vmin=0,vmax=Vmax)
	fig.colorbar(im, )
	ax.set_xlabel('Longitude')
	ax.set_ylabel('Latitude')
	tight_layout()
	event_video= None
	#print len(FrameTitle)
	#print len(Array3D)
	if FrameTitle == None:
		#FrameTitle = np.zeros([len(Array3D)])
		FrameTitle = np.arange(len(Array3D))
	else:
		FrameTitle =  map(GetNormalTime,FrameTitle)

	def GetFrame(n):
		#Flipping the array vertically
		#im.set_data(np.flipud(Array3D[n]))
		im.set_data(np.flipud(Array3D[n].toarray()))
		#plt.title(Title+' '+str(GetNormalTime(FrameTitle[n])))
		plt.title(Title+' '+str(FrameTitle[n]))
		#plt.close()
		return im
	FrameRate = int(1000/Period)
	#FrameRate = 25
	print 'Frame rate is '+str(FrameRate)
	#writer = animation.writers['avconv'](fps=1)
	writer = animation.writers['avconv'](fps=FrameRate)
	#writer = animation.writers['ffmpeg'](fps=FrameRate)
	#print 'len of array3D:'+str(shape(Array3D)[0])
	ani = animation.FuncAnimation(fig,GetFrame,shape(Array3D)[0],interval=100)
	#ani.save(Title+'.mp4', fps=FrameRate, extra_args=['-vcodec', 'libx264'])
	ani.save(Title+'.mp4',writer,dpi=dpi)
	plt.close('all')
	return

def GetBetaDistributionSample(x,a=2.,b=2.):
	if x == 0 or x == 1:
		return 0
	else:
		return (x**(a-1))*((1-x)**(b-1))*math.gamma(a+b)/(math.gamma(a)*math.gamma(b))

def GetBetaDistribution(X,a,b):
	def Beta(x):
		if x == 0 or x == 1:
			return 0
		else:
			return (x**(a-1))*((1-x)**(b-1))*math.gamma(a+b)/(math.gamma(a)*math.gamma(b))
	return map(Beta,X)

def fft(x,shape=None):
	return np.fft.rfft2(x,shape)

def ifft(x,shape=None):
	return np.fft.irfft2(x,shape)
	

##########################PLOTS#################################
##########################PLOTS#################################
##########################PLOTS#################################
def p(a,t=None):
	plt.close('all')
	plt.imshow(a)
	plt.show()
	return 0

def Plot(x,y):
	plt.close('all')
	plt.plot(x,y,'r')
	plt.grid()
	plt.show()
	plt.savefig('a.png')
	return plt

def Plot2(List, Save = False, Name = None, Label = None, Location=0, EqualAxes = False, AxesLim = None,xlabel=None, ylabel=None, fontsize = None, ylimit = None):
	plt.close('all')
	if not Label == None and not len(Label) == len(List):
		raise ValueError('Not enough Lables:',Label)
	if Label==None:
		Label = [None]
	fig, ax = plt.subplots()
	for ListEle,EleLabel in map(None,List,Label):
		ax.plot(ListEle[0],ListEle[1],ListEle[2], label=EleLabel)
	if fontsize == None:
		legend = ax.legend(loc=Location, shadow=False)
	else:
		legend = ax.legend(loc=Location, shadow=False, fontsize=fontsize)
	plt.grid()
	if EqualAxes:
		plt.axis('equal')
	
	if xlabel:
		ax.set_xlabel(xlabel)
	else:
		ax.set_xlabel('X Co-ordinates (m)')
	if ylabel:
		ax.set_ylabel(ylabel)
	else:
		ax.set_ylabel('Y Co-ordinates (m)')
	ax.labelsize = 'large'
	if not AxesLim == None:
		ax.set_xlim(AxesLim[0])
		ax.set_ylim(AxesLim[1])
	plt.tight_layout()
	
	if not ylimit==None:
		ax.set_ylim(ylimit[0],ylimit[1])
		
	if Save:
		plt.savefig(Name)
	else:
		plt.show()
	return

def Plot2_zoom(List, Save = False, Name = None, Label = None, Location=0, EqualAxes = False, AxesLim = None,xlabel=None, ylabel=None, fontsize = None, ylimit = None):
	plt.close('all')
	if not Label == None and not len(Label) == len(List):
		raise ValueError('Not enough Lables:',Label)
	if Label==None:
		Label = [None]
	fig, ax = plt.subplots()
	
	for ListEle,EleLabel in map(None,List,Label):
		ax.plot(ListEle[0],ListEle[1],ListEle[2], label=EleLabel)
	if fontsize == None:
		legend = ax.legend(loc=Location, shadow=False)
	else:
		legend = ax.legend(loc=Location, shadow=False, fontsize=fontsize)
	plt.grid()
	if EqualAxes:
		plt.axis('equal')
	
	if xlabel:
		ax.set_xlabel(xlabel)
	else:
		ax.set_xlabel('X Co-ordinates (m)')
	if ylabel:
		ax.set_ylabel(ylabel)
	else:
		ax.set_ylabel('Y Co-ordinates (m)')
	ax.labelsize = 'large'
	if not AxesLim == None:
		ax.set_xlim(AxesLim[0])
		ax.set_ylim(AxesLim[1])
	plt.tight_layout()
	
	if not ylimit==None:
		ax.set_ylim(ylimit[0],ylimit[1])
	
	ax2 = plt.axes([.55, .525, .4, .4])
	for ListEle in List[:4]:
		ax2.plot(ListEle[0][-4:],ListEle[1][-4:],ListEle[2])
	plt.xticks(fontsize=10)
	plt.yticks(fontsize=10)
	plt.grid()
	
	if Save:
		plt.savefig(Name)
	else:
		plt.show()
	return
	
def Plot2_(List, Save = False, Name = None, Label = None, Location=0, EqualAxes = False, AxesLim = None,xlabel=None, ylabel=None):
	plt.close('all')
	if not Label == None and not len(Label) == len(List):
		raise ValueError('Not enough Lables:',Label)
	if Label==None:
		Label = [None]
	fig, ax = plt.subplots()
	
	DelY = (np.arange(len(List))-len(List)/2.)/(1.*len(List))
	i=0
	for ListEle,EleLabel in map(None,List,Label):
		ax.errorbar(x=ListEle[0]+DelY[i],y=ListEle[1],yerr=ListEle[2], fmt=ListEle[3], label=EleLabel)
		i+=1
		#ax.fill_between(ListEle[0], ListEle[1]-ListEle[2], ListEle[1]+ListEle[2],alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=4, linestyle='dashdot', antialiased=True)
	#legend = ax.legend(loc=Location, shadow=False, fontsize='x-large')
	legend = ax.legend(loc=Location, shadow=False)
	plt.grid()
	if EqualAxes:
		plt.axis('equal')
	
	if xlabel:
		ax.set_xlabel(xlabel)
	else:
		ax.set_xlabel('X Co-ordinates (m)')
	if ylabel:
		ax.set_ylabel(ylabel)
	else:
		ax.set_ylabel('Y Co-ordinates (m)')

	if not AxesLim == None:
		ax.set_xlim(AxesLim[0])
		ax.set_ylim(AxesLim[1])
	plt.tight_layout()
	if Save:
		plt.savefig(Name)
	else:
		plt.show()
	return
			
def Plot2Return(List,Label = None):
	if not Label == None and not len(Label) == len(List):
		raise ValueError('Not enough Lables:',Label)
	if Label==None:
		Label = [None]
	for ListEle,EleLabel in map(None,List,Label):
		plt.plot(ListEle[0],ListEle[1],ListEle[2], label=EleLabel)
	legend = plt.legend(loc=0, shadow=False, fontsize='x-large')
	plt.grid()
	plt.tight_layout()

	
def plot2(List):
	plt.close('all')
	[ListEle0,ListEle1] = List
	plt.figure(1)
	plt.subplot(211)
	Plot2Return(ListEle0)
	plt.subplot(212)
	Plot2Return(ListEle1)
	plt.show()

def plot3(List, Save = False, Name = None, Label = None, xylabel = None):
	def plot2Return(List, Label = None, xlabel = None, ylabel = None):
		if not Label == None and not len(Label) == len(List):
			raise ValueError('Not enough Lables:',Label)
		if Label==None:
			Label = [None]
		for ListEle,EleLabel in map(None,List,Label):
			plt.plot(ListEle[0],ListEle[1],ListEle[2], label=EleLabel)
		legend = plt.legend(loc=0, shadow=False, fontsize='x-small')
		if xlabel:
			plt.xlabel(xlabel)
		if ylabel:
			plt.ylabel(ylabel)
		plt.grid()
		#plt.tight_layout()
		return
	
	plt.close('all')
	if Label == None:
		Label = [None,None,None]
	if xylabel == None:
		xylabel = [[None,None], [None,None], [None,None]]
	[ListEle0,ListEle1,ListEle2] = List
	
	#subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
	#for case 2, 3, 5, 6, 7
	left  = 0.16  # the left side of the subplots of the figure
	
	#for case 4, gaussian 7
	left  = 0.20  # the left side of the subplots of the figure
	
	#right = 0.9    # the right side of the subplots of the figure
	#bottom = 0.1   # the bottom of the subplots of the figure
	#top = 0.9      # the top of the subplots of the figure
	#wspace = 0.2   # the amount of width reserved for blank space between subplots
	hspace = 0.3   # the amount of height reserved for white space between subplots
	subplots_adjust(left=left, bottom=None, right=None, top=None, wspace=None, hspace=hspace)

	plt.figure(1)
	plt.subplot(311)
	plot2Return(ListEle0,Label[0], xylabel[0][0], xylabel[0][1])
	plt.subplot(312)
	plot2Return(ListEle1,Label[1], xylabel[1][0], xylabel[1][1])
	plt.subplot(313)
	plot2Return(ListEle2,Label[2], xylabel[2][0], xylabel[2][1])
	if Save:
		plt.savefig(Name)
	else:
		plt.show()
		

def PlotN(X,Y):
	plt.close('all')
	for x,y in zip(X,Y):
		plt.plot(x,y,'o')
		#plt.plot(np.asarray(x),np.asarray(y))
		#plt.plot(ListEle[0],ListEle[1])
	plt.grid()
	#plt.axis('equal')
	plt.tight_layout()
	plt.show()
##########################PLOTS#################################
##########################PLOTS#################################
##########################PLOTS#################################
def CreateScoreTemplates():
	global CompassErrorTemplate
	ActivityScore = np.zeros((ScoreSize, ScoreSize), dtype=np.uint8)
	rr, cc = SkimageDraw.circle(ScoreSizeBy2, ScoreSizeBy2, ActivityScoreRadius)
	ActivityScore[rr, cc] = ActivityIntensity
	rr, cc = SkimageDraw.circle(ScoreSizeBy2, ScoreSizeBy2, ActivityScoreRadius-1)
	ActivityScore[rr, cc] = ActivityIntensity
	rr, cc = SkimageDraw.circle(ScoreSizeBy2, ScoreSizeBy2, ActivityScoreRadius-2)
	ActivityScore[rr, cc] = ActivityIntensity

	ActivityScoreSmall = np.zeros((ScoreSize, ScoreSize), dtype=np.uint8)
	rr, cc = SkimageDraw.circle(ScoreSizeBy2, ScoreSizeBy2, ActivityScoreRadiusSmall)
	ActivityScoreSmall[rr, cc] = ActivityIntensity


	if Resolution == 'Meter':
		ScoreTemplateFileName = 'PickleFiles/ScoreTemplate'+'MVD='+str(int(MaxDistance/ImprovementFactor))+'m.txt'
		#GPSErrorTemplateFileName = 'PickleFiles/GPSErrorTemplate.txt'
		CompassErrorTemplateFileName = 'PickleFiles/CompassErrorTemplate5'+'MVD='+str(int(MaxDistance/ImprovementFactor))+'m.txt'
	if Resolution == 'Quartermeter':
		ScoreTemplateFileName = 'PickleFiles/ScoreTemplateQuartermeter'+'MVD='+str(int(MaxDistance/ImprovementFactor))+'m.txt'
		#GPSErrorTemplateFileName = 'PickleFiles/GPSErrorTemplateQuartermeter.txt'
		CompassErrorTemplateFileName = 'PickleFiles/CompassErrorTemplate5Quartermeter'+'MVD='+str(int(MaxDistance/ImprovementFactor))+'m.txt'
	if Resolution == 'Decimeter':
		ScoreTemplateFileName = 'PickleFiles/ScoreTemplateDecimeter'+'MVD='+str(int(MaxDistance/ImprovementFactor))+'m.txt'
		#GPSErrorTemplateFileName = 'PickleFiles/GPSErrorTemplateDecimeter.txt'
		CompassErrorTemplateFileName = 'PickleFiles/CompassErrorTemplate5Decimeter'+'MVD='+str(int(MaxDistance/ImprovementFactor))+'m.txt'
	if Resolution == 'Centimeter':
		ScoreTemplateFileName = 'PickleFiles/ScoreTemplateCentimeter'+'MVD='+str(int(MaxDistance/ImprovementFactor))+'m.txt'
		#GPSErrorTemplateFileName = 'PickleFiles/GPSErrorTemplateCentimeter.txt'
		CompassErrorTemplateFileName = 'PickleFiles/CompassErrorTemplate5Centimeter'+'MVD='+str(int(MaxDistance/ImprovementFactor))+'m.txt'

	if os.path.isfile(ScoreTemplateFileName):
		ScoreTemplateFile = open(ScoreTemplateFileName,'r')
		ScoreTemplate = pickle.load(ScoreTemplateFile)
		ScoreTemplateFile.close()
	else:
		ScoreTemplate = []
		for Yaw in range(360):
			print 'Creating Score templet for Yaw = '+str(Yaw)
			CamTempletScore = np.zeros((ScoreSize, ScoreSize), dtype=np.uint8)
			rr, cc, val = SkimageDraw.line_aa(ScoreSizeBy2, ScoreSizeBy2, int(ScoreSizeBy2-MaxDistance*np.cos(np.radians(Yaw))),int(ScoreSizeBy2+MaxDistance*np.sin(np.radians(Yaw))))
			#CamTempletScore[rr, cc] = val*ScoreIntensity
			CamTempletScore[rr, cc] = ScoreIntensity
			#Creating Arrow Heads
			rr, cc, val = SkimageDraw.line_aa(int(ScoreSizeBy2-MaxDistance*np.cos(np.radians(Yaw))),int(ScoreSizeBy2+MaxDistance*np.sin(np.radians(Yaw))),int(ScoreSizeBy2-MaxDistance*np.cos(np.radians(Yaw))-ArrowLength*np.cos(np.radians(Yaw+180+ArrowHeadAngle))),int(ScoreSizeBy2+MaxDistance*np.sin(np.radians(Yaw))+ArrowLength*np.sin(np.radians(Yaw+180+ArrowHeadAngle))))
			#CamTempletScore[rr, cc] = val*ScoreIntensity
			CamTempletScore[rr, cc] = ScoreIntensity
			rr, cc, val = SkimageDraw.line_aa(int(ScoreSizeBy2-MaxDistance*np.cos(np.radians(Yaw))),int(ScoreSizeBy2+MaxDistance*np.sin(np.radians(Yaw))),int(ScoreSizeBy2-MaxDistance*np.cos(np.radians(Yaw))-ArrowLength*np.cos(np.radians(Yaw+180-ArrowHeadAngle))),int(ScoreSizeBy2+MaxDistance*np.sin(np.radians(Yaw))+ArrowLength*np.sin(np.radians(Yaw+180-ArrowHeadAngle))))
			#CamTempletScore[rr, cc] = val*ScoreIntensity
			CamTempletScore[rr, cc] = ScoreIntensity
			CamTempletScore = CamTempletScore+ActivityScore
			ScoreTemplate.append(sparse.coo_matrix(CamTempletScore))
		ScoreTemplateFile = open(ScoreTemplateFileName,'w')
		pickle.dump(ScoreTemplate,ScoreTemplateFile)
		ScoreTemplateFile.close()
	'''
	if os.path.isfile(GPSErrorTemplateFileName):
		GPSErrorTemplateFile = open(GPSErrorTemplateFileName,'r')
		GPSErrorTemplate = pickle.load(GPSErrorTemplateFile)
		GPSErrorTemplateFile.close()
	else:
		GPSErrorTemplate = []
		for Error in range(1*ImprovementFactor,200*ImprovementFactor,ImprovementFactor):
			print 'Creating GPS Error templet for GPS Error = '+str(Error)
			ErrorTemplate = deepcopy(GpsTempletScoreTemp)
			ErrorTemplate[ErrorTemplate<=Error] = 1
			ErrorTemplate[ErrorTemplate>Error] = 0
			GPSErrorTemplate.append(sparse.coo_matrix(ErrorTemplate))
		GPSErrorTemplateFile = open(GPSErrorTemplateFileName,'w')
		pickle.dump(GPSErrorTemplate,GPSErrorTemplateFile)
		GPSErrorTemplateFile.close()
	'''
	#'''
	if os.path.isfile(CompassErrorTemplateFileName):
		print 'Reading Compass Error Templet from file...'
		CompassErrorTemplateFile = open(CompassErrorTemplateFileName,'r')
		CompassErrorTemplate = pickle.load(CompassErrorTemplateFile)
		CompassErrorTemplateFile.close()
	else:
		CompassErrorTemplate = []
		print 'Creating Compass Error Templet for 5 degree Error!'
		PhiErr = 5
		for Yaw in range(360):
			print 'Creating Compass Error Templet for Yaw = '+str(Yaw)+'	and Error = '+str(PhiErr)
			CompassErrorScore = CamTempletBoolean*np.minimum(abs(CamTempletAngle-Yaw),abs(abs(CamTempletAngle-Yaw)-360))
			CompassErrorScore[CompassErrorScore<=5] = 1
			CompassErrorScore[CompassErrorScore>5] = 0
			CompassErrorScore = CompassErrorScore*CompassErrorIntensity
			CompassErrorScore = CamTempletBoolean*CompassErrorScore
			CompassErrorTemplate.append(sparse.coo_matrix(CompassErrorScore))
			#CamTempletAngleDiff = MyMatrixAngleDiff(CamTempletAngle-Yaw,CamTempletBoolean)
			#CamTempletScore = MyMatrixCameraScore(CamTempletDistance,CamTempletAngleDiff,PhiErr)
		#	CompassErrorTemplate.append(CamTempletScore)
		#for ind in range(len(CompassErrorTemplate)):
		#	CompassErrorTemplate[ind]=sparse.coo_matrix(CompassErrorTemplate[ind])
		CompassErrorTemplateFile = open(CompassErrorTemplateFileName,'w')
		pickle.dump(CompassErrorTemplate,CompassErrorTemplateFile)
		CompassErrorTemplateFile.close()
	#'''
	ActivityScore = sparse.coo_matrix(ActivityScore)
	ActivityScoreSmall = sparse.coo_matrix(ActivityScoreSmall)
	return ScoreTemplate, ActivityScore, ActivityScoreSmall #end of CreateScoreTemplates()

def GetScore(sol, ScoreTemplate):
	'''
	If x co-ordinate is +ve, increase column co-ordinates by x
	If x co-ordinate is -ve, decrease column co-ordinates by x
	col = col + x

	If y co-ordinate is +ve, decrease row co-ordinates by y
	If y co-ordinate is -ve, increase row co-ordinates by y
	row = row - y
	'''
	#print 'Inside GetScore'
	#print sol
	xi = sol[0]
	yi = sol[1]
	phi = sol[2]
	di = sol[3]

	Score = deepcopy(ScoreTemplate[int(round(phi))%360])
	#Shifting score as per xi yi coordinates
	Score.row = Score.row - int(round(yi))
	Score.col = Score.col + int(round(xi))
	return sparse.csc_matrix(Score.toarray())

def CreateScore(Solution,VideoList,ScoreTemplate):
	'''	
	Scores is array of size similar to TimeLineIndicator
	If there is valid entry in TimeLineIndicator, Score is computed from Solution
	Else Score is array of zero
	'''
	Score = []
	Scores = []
	#ZeroScore = sparse.coo_matrix(np.zeros([ScoreSize,ScoreSize]))
	ZeroScore = sparse.csc_matrix(np.zeros([ScoreSize,ScoreSize]))
	for j in range(len(VideoList)):
		print 'Creating a Score for '+str(j)+'th video'
		for Sol in Solution:
			#print 'Solution is '+str(Sol)
			if not Sol[j+1]==[None]:
				Score.append(GetScore(Sol[j+1],ScoreTemplate))
			else:
				Score.append(ZeroScore)
		Scores.append(Score)
		Score = []
	return Scores

def CreateActivity(Solution,ActivityScoreSmall):
	Activity = []
	ActivityTrailed = []

	print 'Creating Activity'
	for Sol in Solution:
		X=Sol[0][1]
		Y=Sol[0][2]
		Activity.append(GetActivity(X,Y))
		ActivityTrailed.append(GetActivitySmall(X,Y,ActivityScoreSmall))

	print 'Creating Activity Trail'
	for i in range(1,len(ActivityTrailed)):
		CurrentActivityTrail = ActivityTrailed[i-1].toarray()
		CurrentActivityTrail = ActivityTrailed[i].toarray()+CurrentActivityTrail
		CurrentActivityTrail[CurrentActivityTrail>0]=ActivityIntensity
		ActivityTrailed[i] = sparse.csc_matrix(CurrentActivityTrail)
	return Activity,ActivityTrailed
	#return ActivityTrailed

def GetActivity(X,Y):
	#CurrentActivityScore = deepcopy(ActivityScore)
	CurrentActivityScore = GPSErrorTemplate(ActivityScoreRadius)
	#Shifting activity score as per XY co-ordinates
	CurrentActivityScore.row = CurrentActivityScore.row - int(round(Y))
	CurrentActivityScore.col = CurrentActivityScore.col + int(round(X))
	return sparse.csc_matrix(CurrentActivityScore)

def GetActivitySmall(X,Y,ActivityScoreSmall):
	CurrentActivityScore = deepcopy(ActivityScoreSmall)
	#Shifting activity score as per XY co-ordinates
	CurrentActivityScore.row = CurrentActivityScore.row - int(round(Y))
	CurrentActivityScore.col = CurrentActivityScore.col + int(round(X))
	return sparse.csc_matrix(CurrentActivityScore)

def GPSErrorTemplate(GpsError):
	GpsErrorPixel = int(GpsError*ImprovementFactor)
	ErrorTemplate = np.zeros((ScoreSize, ScoreSize), dtype=np.uint8)
	rr, cc, val = SkimageDraw.circle_perimeter_aa(ScoreSizeBy2, ScoreSizeBy2, GpsErrorPixel)
	#ErrorTemplate[rr, cc] = GpsErrorIntensity*val
	ErrorTemplate[rr, cc] = GpsErrorIntensity*1
	rr, cc, val = SkimageDraw.circle_perimeter_aa(ScoreSizeBy2, ScoreSizeBy2, GpsErrorPixel-1)
	#ErrorTemplate[rr, cc] = GpsErrorIntensity*val
	ErrorTemplate[rr, cc] = GpsErrorIntensity*1
	rr, cc, val = SkimageDraw.circle_perimeter_aa(ScoreSizeBy2, ScoreSizeBy2, GpsErrorPixel-2)
	#ErrorTemplate[rr, cc] = GpsErrorIntensity*val
	ErrorTemplate[rr, cc] = GpsErrorIntensity*1
	#print np.max(ErrorTemplate)
	return sparse.coo_matrix(ErrorTemplate)

def CreateGroundTruthActivity(Video,TimeLineIndicator,ActivityScoreSmall):
	#raise ValueError('Inside Create Ground Truth!')
	ActivityGroundTruth = []
	ActivityGroundTruthTrail = []
	print 'Creating Ground Truth'
	for t in TimeLineIndicator:
		Index = t[-1]
		if not Index == -1:
			X = Video.sensor.TYXYE[Index][2]*ImprovementFactor
			Y = Video.sensor.TYXYE[Index][3]*ImprovementFactor
			GPSError = int(round(Video.sensor.TYXYE[Index][4]))
			ActivityGroundTruthTrail.append(GetActivitySmall(X,Y,ActivityScoreSmall))
			ActivityGroundTruthTemp = GPSErrorTemplate(GPSError)
			ActivityGroundTruthTemp.row = ActivityGroundTruthTemp.row - int(round(Y))
			ActivityGroundTruthTemp.col = ActivityGroundTruthTemp.col + int(round(X))
			ActivityGroundTruth.append(sparse.csc_matrix(ActivityGroundTruthTemp))
		else:
			raise ValueError('Ground Truth not available!')
	print 'Creating Ground Truth Trail'
	for i in range(1,len(ActivityGroundTruthTrail)):
		PrevActivityTrail = ActivityGroundTruthTrail[i-1].toarray()
		CurrentActivityTrail = ActivityGroundTruthTrail[i].toarray()+PrevActivityTrail
		PrevActivityTrail[PrevActivityTrail>0] = ActivityIntensity
		ActivityGroundTruthTrail[i] = sparse.csc_matrix(CurrentActivityTrail)
	return ActivityGroundTruth,ActivityGroundTruthTrail

def GetFinalScore(Scores,Activity):
	FinalScore = [sum(ConcurrentScore) for ConcurrentScore in zip(*Scores)]
	if not len(FinalScore) == len(Activity):
		print 'Shape of Final Score is '+str(len(FinalScore))
		print 'Length of Activity is '+str(len(Activity))
		raise ValueError('Final Score & Activity lengths do not match!')
	'''
	#Adding GPS Error to the Score
	for i in range(len(TimeLineIndicator)):
		for j in range(len(Score)):
			Index = TimeLineIndicator[i][j+1]
			if not Index==-1:
				GPSError=vidList.videoList[j].sensor.TYXYE[Index]
				GPSErrorScore = GPSErrorTemplate[int(round(GPSError))]
				Score[j][i]+=GPSErrorScore
	'''
	return FinalScore

def CreateAnimationData(vidList):
	global ScoreTemplate
	global ActivityScore
	global ActivityScoreSmall
	global ImprovementFactor
	GroundTruthAvailable = vidList.GroundTruthAvailable
	TimeLineIndicator = vidList.TimeLineIndicator

	Solution = vidList.Solution

	def ScaleUpSol(Sol):
		Sol[0][1] = Sol[0][1]*ImprovementFactor
		Sol[0][2] = Sol[0][2]*ImprovementFactor
		for i in range(len(Sol)-1):
			Sol[i+1][0] = Sol[i+1][0]*ImprovementFactor
			Sol[i+1][1] = Sol[i+1][1]*ImprovementFactor
		return Sol

	Solution = map(lambda Sol: ScaleUpSol(Sol), Solution)

	ScoreTemplate, ActivityScore, ActivityScoreSmall  = CreateScoreTemplates()
	if GroundTruthAvailable:
		Scores = CreateScore(Solution,vidList.videoList[0:-1],ScoreTemplate)
	else:
		Scores = CreateScore(Solution,vidList.videoList,ScoreTemplate)

	#Activity = CreateActivity(Solution)
	Activity,ActivityTrailed = CreateActivity(Solution,ActivityScoreSmall)

	#if vidList.GroundTruthAvailable:
	if GroundTruthAvailable:
		ActivityGroundTruth,ActivityGroundTruthTrail = CreateGroundTruthActivity(vidList.videoList[-1],TimeLineIndicator,ActivityScoreSmall)
	else:
		ActivityGroundTruth = None
		ActivityGroundTruthTrail = None
	FinalScore = GetFinalScore(Scores,Activity)

	for i in range(len(vidList.videoList)-1):
		vidList.videoList[i].Score = Scores[i]
	#if not vidList.GroundTruthAvailable:
	if not GroundTruthAvailable:
		vidList.videoList[-1].Score = Scores[-1]
	vidList.Map3D = FinalScore
	vidList.Activity = Activity
	vidList.ActivityTrailed = ActivityTrailed
	vidList.ActivityGroundTruth = ActivityGroundTruth
	vidList.ActivityGroundTruthTrail = ActivityGroundTruthTrail
	return vidList

def AppendVideoAndScore2(vidList,Index,StrtIndx=None,StpIndx=None):
	QueryName = vidList.QueryName
	videoFramePath = '/home/SharedData/AmitData/Video/VideoFrames/'
	'''
	Here we align video frame with corresponding score
	Timestaps for Score are taken from TimeLineIndicator
	Video Frame which is closest to current timestamp
	taken from the server
	'''
	Video = vidList.videoList[Index]
	Solution = vidList.Solution
	Activity = vidList.Activity
	ActivityTrailed = vidList.ActivityTrailed
	TimeLineIndicator = vidList.TimeLineIndicator
	GroundTruthAvailable = vidList.GroundTruthAvailable
	if GroundTruthAvailable:
		ActivityGroundTruth = vidList.ActivityGroundTruth
		ActivityGroundTruthTrail = vidList.ActivityGroundTruthTrail
		
	Lon, Lat = vidList.Misc['LonLat']
	#Lat = vidList.Misc['Lat']
	#Lon = vidList.Misc['Lon']
	Score = Video.Score
	VideoName = QueryName+'/'+Video.id+'.mp4'
	print 'Video Name: '+str(VideoName)
	if Video.start_time:
		raise ValueError('This is not a Fresh Segment!')

	VideoSegStartTime =  Video.start_time + int(Video.info['start_timestamp'])
	VideoSegStopTime = Video.stop_time + int(Video.info['start_timestamp'])

	if os.path.isfile(videoFramePath+str(Video.id)+'/time0.jpg'):
		TestFrame = np.array(Image.open(videoFramePath+str(Video.id)+'/time0.jpg'))
	else:
		raise ValueError('Video frames not found!')
		'''
		url = 'http://api.geovid.org/v1.0/gv/image/'+Video.id+'/time/0'
		print 'Downloading test frame at '+str(url)
		TestFrame = np.asarray(Image.open(StringIO(requests.get(url,proxies=proxies).content)))
		#upsidedown correction
		if Video.id == '6c5d0edc341582056a4970302450a8bd10ff4b02' or Video.id == '9c1d67ae739880ee446f26f67503f64d511ef18a':
			TestFrame = np.fliplr(np.flipud(TestFrame))
		#portraitmode correction
		elif Video.id == '33648240e598636cd94721fd998c52db2452fbfa' or Video.id == '9c55a2f841ac74c14b95ceb4ebe5030f6a87edb3':
			TestFrame = np.fliplr(TestFrame.transpose((1,0,2)))
		'''
	ZeroImage = 0*TestFrame
	[Vheight,Vwidth,channel] = np.shape(TestFrame)
	[Sheight,Swidth] = np.shape(Score[0])
	Sheight -= 2*CropPixels
	Swidth -= 2*CropPixels
	
	print 'Sheight'+str(Sheight)
	print 'Vheight'+str(Vheight)
	if Sheight > Vheight:
		Tall = 'Score'
		ExtraRows = np.zeros([int((Sheight-Vheight)/2),Vwidth,3],dtype=np.uint8)
		Scale = Sheight/Vheight
		NewVheight = int(Scale*Vheight)
		NewVwidth = int(Scale*Vwidth)
	else:
		Tall = 'Video'
		ExtraRows = np.zeros([(Vheight-Sheight)/2,Swidth,3],dtype=np.uint8)

	print Tall
	
	print 'Vheight is '+str(Vheight)
	print 'Vwidth is '+str(Vwidth)
	print 'Sheight is '+str(Sheight)
	print 'Swidth is '+str(Swidth)
	print 'Shape of ExtraRows is'+str(np.shape(ExtraRows))
	Oheight = max(Sheight,Vheight)
	Owidth = Swidth+NewVwidth
	print 'Oheight is '+str(Oheight)
	print 'Owidth is '+str(Owidth)
	print 'Scale:'+str(Scale)
	print 'NewVheight:'+str(NewVheight)
	print 'NewVwidth:'+str(NewVwidth)
	
	#OutFrameRate = int(round(1000/VideoPeriod))
	#OutFrameRate = int(round(1000/Period))
	OutFrameRate = 25 #fps
	FramePeriod = 40 #ms
	#OutFrameRate = 5 #fps
	#FramePeriod = 200 #ms
	#fourcc = cv2.cv.CV_FOURCC(*'XVID')
	#fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
	fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
	#vidput = cv2.VideoWriter('VideoAndScore'+str(Index)+'.avi',fourcc,30,(Owidth,Oheight))
	#vidput = cv2.VideoWriter('VideoAndScore'+str(Index)+'.avi',fourcc,OutFrameRate,(Owidth,Oheight))
	OutName = QueryName+'/'+QueryName+'VideoAndScore'+str(Index)+'.avi'
	vidput = cv2.VideoWriter(OutName,fourcc,OutFrameRate,(Owidth,Oheight))
	#vidput = cv2.VideoWriter('VideoAndScore'+str(Index)+'.mov',fourcc,OutFrameRate,(Owidth,Oheight))
	#Frames = GetVideoFrames(Solution,Index,Video,VideoSegStartTime)
	#Frames = GetVideoFramesFromVideo(Solution,Index,Video,VideoSegStartTime)
	
	#if not BlackBackGround:
	#	BackGround = GetGoogleMapImage(Lat,Lon)
	
	if StrtIndx == None:
		StrtIndx = 0
	if StpIndx == None:
		StpIndx = len(Score)
	simageDict={}
	#for time in range(Solution[0][0][0],Solution[-1][0][0],FramePeriod):
	for time in range(TimeLineIndicator[0,0],TimeLineIndicator[-1,0],FramePeriod):
		#print 'Time in ms '+str(time)
		SolIndx = np.argmin(abs(TimeLineIndicator[:,0]-time))
		i=SolIndx
		Sol = Solution[SolIndx]
		FrameTime = time - VideoSegStartTime
		if not Sol[Index+1]==[None]:
			if FrameTime<0:
				FrameTime = 0
			if FrameTime>Video.stop_time:
				FrameTime = Video.stop_time
			FrameTime = 10*int(round(FrameTime/10))
			print 'FrameTime: '+str(FrameTime)
			image = GetSingleFrame(Video,FrameTime,videoFramePath)
			image[:,:,[0,2]] = image[:,:,[2,0]]
			ActivityX = Sol[0][1]
			ActivityY = Sol[0][2]
			CamX = Sol[Index+1][0]
			CamY = Sol[Index+1][1]
			CamYaw = Sol[Index+1][2]

			#Add rectangular activity box!
			#image = AddActivityBox(image,ActivityX,ActivityY,CamX,CamY,CamYaw)
			image,ActivityDist = AddActivityBox2(image,ActivityX,ActivityY,CamX,CamY,CamYaw)
		else:
			image = ZeroImage

		if i in simageDict.keys():
			simage = deepcopy(simageDict[i])
		else:
			simage = np.empty([Sheight,Swidth,3],'uint8')

			#simage = np.empty([Sheight,Swidth,3],'uint8')
			ErrorScore,GPSErr,PhiErr = GetErrorScore(Video,Sol,TimeLineIndicator[i][Index+1],Index)
			#Creating Grid
			simage[:,:,0] = ScoreGrid[CropPixels:-CropPixels,CropPixels:-CropPixels]
			simage[:,:,1] = ScoreGrid[CropPixels:-CropPixels,CropPixels:-CropPixels]
			simage[:,:,2] = ScoreGrid[CropPixels:-CropPixels,CropPixels:-CropPixels]
			
			#Adding Activity
			simage[:,:,2] += Activity[i].toarray()[CropPixels:-CropPixels,CropPixels:-CropPixels]
			#Adding Activity Trail
			simage[:,:,2] += ActivityTrailed[i].toarray()[CropPixels:-CropPixels,CropPixels:-CropPixels]

			if GroundTruthAvailable:
				#Adding GroundTruth
				simage[:,:,1] += ActivityGroundTruth[i].toarray()[CropPixels:-CropPixels,CropPixels:-CropPixels]
				#Adding GroundTruth Trail
				simage[:,:,1] += ActivityGroundTruthTrail[i].toarray()[CropPixels:-CropPixels,CropPixels:-CropPixels]
				
			#Adding Current Score
			#simage[:,:,1] += Score[i].toarray()[CropPixels:-CropPixels,CropPixels:-CropPixels]
			simage[:,:,0] += Score[i].toarray()[CropPixels:-CropPixels,CropPixels:-CropPixels]
			simage[:,:,2] += ErrorScore[CropPixels:-CropPixels,CropPixels:-CropPixels].astype('uint8')
			
			#Adding GPS Image
			#if not BlackBackGround:
			#	simage[:,:,0] = np.where(simage[:,:,0]!=0,simage[:,:,0],BackGround[CropPixels:-CropPixels,CropPixels:-CropPixels,0])
			#	simage[:,:,1] = np.where(simage[:,:,1]!=0,simage[:,:,1],BackGround[CropPixels:-CropPixels,CropPixels:-CropPixels,1])
			#	simage[:,:,2] = np.where(simage[:,:,2]!=0,simage[:,:,2],BackGround[CropPixels:-CropPixels,CropPixels:-CropPixels,2])
			
			#'''
			simageDict[i]=deepcopy(simage)
		if Tall == 'Video':
			simage = np.append(ExtraRows,simage,axis=0)
			simage = np.append(simage,ExtraRows,axis=0)
		else:
			image = np.array(Image.fromarray(image).resize((NewVwidth,NewVheight)))
		#'''
		if not np.shape(image)[0] == np.shape(simage)[0]:
			print 'Shape of CurrentScore is'+str(np.shape(simage))
			print 'Shape of Video is '+str(np.shape(image))
			raise ValueError('Dimensions of Video Score do not match with Video!')
		#Adding Texts to the image frame 
		CroppedH, CroppedW,channel = np.shape(image)
		TexX = int(CroppedH/20)
		TexY = int(CroppedW/20)
		text_color = (0,255,255)
		#text_color = (0,0,255)
		Text = 'Event position in video frame '
		#cv2.putText(image,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2,linetype=cv2.CV_AA)
		cv2.putText(image,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2)
		Text = 'Cam '+str(Index)
		TexY = int(2.5*CroppedW/20)
		#cv2.putText(image,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2,linetype=cv2.CV_AA)
		cv2.putText(image,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2)

		#Adding Texts to the Score frame 
		CroppedH, CroppedW,channel = np.shape(simage)
		text_color = (255,255,255)
		if True:
		#if BlackBackGround:
			#Text = 'Event Track and Metadata'
			Text = 'Event Track and'
			TexX = int(CroppedH/20)
			TexY = int(CroppedW/20)
			#cv2.putText(simage,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2,linetype=cv2.CV_AA)
			cv2.putText(simage,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2)
			#Text = 'On Ground Plane'
			Text = 'Metadata,'+'C'+str(Index)+',D:'+str(int(ActivityDist))
			TexY = int(2.5*TexY)
			#cv2.putText(simage,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2,linetype=cv2.CV_AA)
			cv2.putText(simage,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2)
			#Text = 'C'+str(Index)+',D:'+str(int(ActivityDist))
			#Text = 'GEr:'+str(GPSErr)+',OEr:'+str(PhiErr)
			Text = 'GEr:'+str(GPSErr)
			TexY = int(4*TexY/2.5)
			#cv2.putText(simage,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2,linetype=cv2.CV_AA)
			cv2.putText(simage,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2)
		else:
			#Text = 'Event Track  and Metadata'
			Text = 'Event Track and'
			TexX = int(CroppedH/20)
			TexY = int(CroppedW/20)
			#cv2.putText(simage,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2,linetype=cv2.CV_AA)
			cv2.putText(simage,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2)
			#Text = 'On Google Map'
			Text = 'Metadata'
			TexY = 2*TexY
			#cv2.putText(simage,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2,linetype=cv2.CV_AA)
			cv2.putText(simage,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2)
			Text = 'Camera '+str(Index)
			TexY = 2*TexY
			#cv2.putText(simage,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2,linetype=cv2.CV_AA)
			cv2.putText(simage,Text , (TexX,TexY), cv2.FONT_HERSHEY_PLAIN, 4.0, text_color, thickness=2)

		image = np.append(image,simage,axis=1)
		vidput.write(image)
	return #End AppendVideoAndScore2

def GetSingleFrame(Video,time, videoFramePath):
	#Check for closest possible frame!
	#print 'Loading frame for time '+str(time)
	Done = False
	time1 = time
	time2 = time
	while not Done:
		FrameName1 = videoFramePath+Video.id+'/'+'time'+str(time1)+'.jpg'	
		FrameName2 = videoFramePath+Video.id+'/'+'time'+str(time2)+'.jpg'	
		if os.path.isfile(FrameName1):
			CurrentFrame = np.array(Image.open(FrameName1))	
			#print 'Loaded frame for time '+str(time1)
			Done = True
		elif os.path.isfile(FrameName2):
			CurrentFrame = np.array(Image.open(FrameName2))
			#print 'Loaded frame for time '+str(time2)
			Done = True
		else:
			time1 = time1-10
			time2 = time2+10
	return CurrentFrame

def AddActivityBox2(Img,ActivityX,ActivityY,CamX,CamY,Yaw):
	if ActivityX == None:
		return Img, 100
	x = int(round(ActivityX-(CamX)))
	y = int(round(ActivityY-(CamY)))
	ActivityDist = 0.00001+np.sqrt(x*x+y*y)/ImprovementFactor
	#if ActivityDist>MaxVisibleDistance:
	return Img, ActivityDist
	ViewAngleDegrees = 50
	[Height,Width,Channel] = np.shape(Img)	
	PixelsPerDegree = Width/50.0
	#Dimensions of the box for event at a d meters distance away!
	#BoxWidth = 0.5*Width*5/ActivityDist
	#BoxHeigth = 1.2*Height*5/ActivityDist

	#Not working too small
	BoxWidth = 0.3*Width*2/ActivityDist
	BoxHeight = 1.0*Height*2/ActivityDist

	#Working!
	BoxWidth = 0.4*Width*6/ActivityDist
	BoxHeight = 1.0*Height*6/ActivityDist
	
	#
	BoxWidth = 0.4*Width*6/ActivityDist
	BoxHeight = 1.0*Height*6/ActivityDist

	ActivityAngle = CamTempletAngle[ScoreSizeBy2-y,x+ScoreSizeBy2]
	Delta = (ActivityAngle-Yaw)%360
	if Delta <= 180:
		Bx= PixelsPerDegree*Delta
	else:
		Bx = -PixelsPerDegree*(360-Delta)
	Bx = Width/2.0+Bx
	By = Height/2.0
	#By is center of the event box to be drawn!
	P1x = int(Bx-BoxWidth/2.0)
	P1y = int(By-BoxHeight/2.0)

	P2x = int(Bx+BoxWidth/2.0)
	P2y = int(By+BoxHeight/2.0)

	cv2.rectangle(Img,(P1x,P1y),(P2x,P2y),(0,255,255),3)
	#Adding red box at the center of the frame!
	'''
	P1x = int(Width/2.0-BoxWidth/2.0)
	P1y = int(Height/2.0-BoxHeight/2.0)

	P2x = int(Width/2.0+BoxWidth/2.0)
	P2y = int(Height/2.0+BoxHeight/2.0)
	cv2.rectangle(Img,(P1x,P1y),(P2x,P2y),(0,0,255),3)
	'''
	'''
	print ActivityDist
	print ActivityAngle
	print Delta
	print BoxWidth
	print BoxHeigth
	print P1x
	print P1y
	print P2x
	print P2y
	'''
	return Img,ActivityDist #End AddActivityBox2

def GetPhiError(MagneticField):
	MagMean = 42.0
	MagVar = 7.0
	MagLambda = 0.5
	MagneticField = np.array(MagneticField)
	
	PhiErr = MagLambda*np.exp(np.square((MagneticField-MagMean)/MagVar))
	PhiErr = 5*np.round(PhiErr/5)
	PhiErr = np.array(PhiErr+10)
	np.minimum(PhiErr,180,PhiErr)
	return PhiErr

def GetCompassErrorTemplet(PhiErr,Yaw):
	#if PhiErr >=90:
	#	CompassErrorScore = 0*deepcopy(CompassErrorTemplate[Yaw%360])
	#	return CompassErrorScore
	if PhiErr == 5:
		CompassErrorScore = deepcopy(CompassErrorTemplate[Yaw%360])
	else:
		Width = PhiErr*2
		if (Width/10)%2==0:#if width is 20,40,60,80...
			CompassErrorScore = (GetCompassErrorTemplet(PhiErr/2,int(Yaw-PhiErr/2))).toarray()+(GetCompassErrorTemplet(PhiErr/2,int(Yaw+PhiErr/2))).toarray()
			CompassErrorScore = sparse.coo_matrix(CompassErrorScore)
		else:
			CompassErrorScore = (GetCompassErrorTemplet(5,Yaw)).toarray()+(GetCompassErrorTemplet((2*PhiErr-10)/4,int(Yaw-((2*PhiErr-10)/4+5)))).toarray()+(GetCompassErrorTemplet((2*PhiErr-10)/4,int(Yaw+((2*PhiErr-10)/4+5)))).toarray()
			CompassErrorScore = sparse.coo_matrix(CompassErrorScore)
	return CompassErrorScore

def GetErrorScore(Video,Sol,ii,Index):
	if not ii==-1:
		#cxi = Video.sensor.TYXYE[ii][2]-MedX
		#cyi = Video.sensor.TYXYE[ii][3]-MedY
		#cxi = Video.sensor.TYXYE[ii][2]-MeanX
		#cyi = Video.sensor.TYXYE[ii][3]-MeanY
		cxi = Video.sensor.TYXYE[ii][2]*ImprovementFactor
		cyi = Video.sensor.TYXYE[ii][3]*ImprovementFactor
		Yaw = int(round(Video.sensor.TYXYE[ii][1]))
		Yaw = Yaw%360
		GPSError = int(round(Video.sensor.TYXYE[ii][4]))
		xi = Sol[Index+1][0]
		yi = Sol[Index+1][1]
		#GPSErrorScore = deepcopy(GPSErrorTemplate[GPSError])
		GPSErrorScore = GPSErrorTemplate(GPSError)
		GPSErrorScore.row = GPSErrorScore.row - int(round(cyi))
		GPSErrorScore.col = GPSErrorScore.col + int(round(cxi))
		GPSErrorScore = GPSErrorScore.toarray()
		PhiErr = int(GetPhiError(Video.sensor.MagneticData[ii][3]))
		#CompassErrorScore = GetCompassErrorTemplet(PhiErr,Yaw)
		CompassErrorScore = GetCompassErrorTemplet(5,Yaw)
		CompassErrorScore.row = CompassErrorScore.row - int(round(yi))
		CompassErrorScore.col = CompassErrorScore.col + int(round(xi))
		CompassErrorScore = CompassErrorScore.toarray()
		#ErrorScore = (GPSErrorScore + CompassErrorScore)
		ErrorScore = (GPSErrorScore + .5*CompassErrorScore)
	else:
		#ErrorScore = GPSErrorTemplate[0].toarray()
		ErrorScore = GPSErrorTemplate(0).toarray()
	return ErrorScore,GPSError,PhiErr


