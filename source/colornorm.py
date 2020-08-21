import spams
import numpy as np
import cv2
import time
import sys, os
import copy
#source = cv2.imread('0.png')
#target = cv2.imread('1.png')
nstains=2
lam=0.02
param = {   'K' : 2, 
            'lambda1' : 0.02,
            'numThreads' : 4, 
            'mode':2,
            'iter' : 200,
            'posAlpha':True,
            'posD':True,
            'batchsize': 400,
            'clean':True,
            }

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__
def stainsep(I,nstains,lam):
    global param
    if I.ndim != 3:
        print('[stainsep]Input must be 3-D')
        return [] ,[],[]
    rows,cols  = I.shape[:2]
    V,V1= BLtrans(I)
    #print(V1,V1.shape,V1.dtype)
    param['batchsize']=round(0.2*V1.shape[0])
    Wi=get_staincolor_sparsenmf(V1)
    Hi=estH(V, Wi, rows,cols)
    #print(Wi)
    return Wi,Hi
def get_staincolor_sparsenmf(v):
    blockPrint()
    D=spams.trainDL(np.transpose(v),**param)
    enablePrint()
   
    a_arg = np.argsort(np.transpose(D)[:,1])
   # print(np.transpose(np.transpose(D)[a_arg]))
    return np.transpose(np.transpose(D)[a_arg])
def BLtrans(I):
    Ivecd=np.reshape(I,[I.shape[0]*I.shape[1],I.shape[2]])
    V=np.float64(np.log(255)-np.log(Ivecd+1))
    img_lab = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)
    luminlayer = np.reshape(np.array(img_lab[:,:,0],np.float64),[I.shape[0]*I.shape[1]])
    
    Inew=Ivecd[(luminlayer/255)<0.9] 
    #print(Inew.shape)
    VforW=np.log(255)-np.log(Inew+1)
    return V, np.float64(VforW)
def estH(v, Ws, nrows,ncols):
    par=copy.deepcopy(param)
    par['pos']=True
    del par['K']
    del par['iter']
    del par['posAlpha']
    del par['posD']
    del par['batchsize']
    del par['clean']
   # print(v.dtype)
    Hs_vec=np.transpose(spams.lasso(np.transpose(v),Ws,**par))
    Hs_vec=Hs_vec.toarray()
    #print(Hs_vec.shape)
    Hs = np.reshape(Hs_vec, [nrows, ncols, param['K']])
    #print(Hs.shape)
    return Hs
def SCN(source,Hta,Wta,Hso):
    Hso=np.reshape(Hso,[Hso.shape[0]*Hso.shape[1],Hso.shape[2]])
    Hso_Rmax =np.percentile(Hso,99, axis=0)#0是纵轴
    Hta=np.reshape(Hta,[Hta.shape[0]*Hta.shape[1],Hta.shape[2]])
    Hta_Rmax =np.percentile(Hta,99, axis=0)
   # print(Hso_Rmax.shape,Hta_Rmax.shape)
    normfac=Hta_Rmax/Hso_Rmax
    #print(normfac.shape)
    Hsonorm=Hso*normfac
    #print(Hsonorm.shape)
    Ihat=np.dot(Wta,np.transpose(Hsonorm))
    sourcenorm=np.uint8(255*np.exp(-np.reshape(np.transpose(Ihat),source.shape)))
    return sourcenorm

#strat_time=time.time()
def CN(source,target):
    Wis, His=stainsep(source,nstains,lam)
    #print(Wis,'#####',Wis.shape,'#####',His,'#####',His.shape)
    Wi, Hi=stainsep(target,nstains,lam)
    #print(Wi,'#####',Wi.shape,'#####',Hi,'#####',Hi.shape)
    out=SCN(source,Hi,Wi,His)
    return out
#cv2.imwrite('2.png',out)


