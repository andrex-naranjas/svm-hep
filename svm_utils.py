# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (Belle2)
#     language: python
#     name: python3
# ---

import uproot
import sys
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn import preprocessing
import ROOT
from sklearn.model_selection import train_test_split
import ctypes
import numpy as np
from random import sample
import pandas as pd
from sklearn.metrics import accuracy_score
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
# import h5py
# import pickle
import save_pickle
from save_pickle import models_pickle
from keras.models import model_from_json

# +

workpath=os.getcwd()
print(workpath)
file_train=uproot.open(workpath+'/'+'D0_Belle2_data.root')
tree_train=file_train['variables']
print(tree_train.keys())
df_train=tree_train.arrays(library="pd")
# -

shuffled_df = df_train.sample(frac=1,random_state=4)

# +

df_sig= shuffled_df.loc[shuffled_df['isSignal']==1].sample(n=10000,random_state=42)
df_bkg= shuffled_df.loc[shuffled_df['isSignal']==0].sample(n=10000,random_state=42)

# -

df_comb = pd.concat([df_sig, df_bkg])

# +

df_comb=df_comb.drop(['__experiment__', '__run__', '__event__', '__candidate__',
       '__ncandidates__', '__weight__','M','useCMSFrame__bop__bc','daughter__bo0__cmextraInfo__bodecayModeID__bc__bc',
       'daughter__bo1__cmextraInfo__bodecayModeID__bc__bc',
       'daughter__bo2__cmextraInfo__bodecayModeID__bc__bc'],axis=1)


# +


scaler=preprocessing.StandardScaler().fit_transform(df_comb)

df_comb = pd.DataFrame(scaler, index=df_comb.index, columns=df_comb.columns)

# +

df_comb.dropna(axis=0,inplace=True)

# +

df_sig= df_comb.loc[df_comb['isSignal']==1]
df_bkg= df_comb.loc[df_comb['isSignal']==-1]

# -

x=df_comb.iloc[:,0:16]
y=df_comb['isSignal']
xs=df_sig.iloc[:,0:16]
ys=df_sig['isSignal']
xb=df_bkg.iloc[:,0:16]
yb=df_bkg['isSignal']

# +

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.9)
x_trains,x_tests,y_trains,y_tests=train_test_split(xs,ys,random_state=42,test_size=0.9)
x_trainb,x_testb,y_trainb,y_testb=train_test_split(xb,yb,random_state=42,test_size=0.9)

# -


      





class linear():
    def __init__(self,x,y,y_train,x_train):
        self.x=np.array(x_train)
        self.y=np.array(x_train)
        self.y_train=y_train
        self.x_train =x_train
    #def gram(self,kern):
     #   return np.array([[kern(i,j) for j in self.y ]for i in self.x]) 
    #pure calculation
    def Linear_exp(self,v1,v2):
        prod=0
        for i in range(0,len(v1)):
            prod += v1[i]*v2[i]
        return prod      
    def gram_explicit_calc(self):
        return np.array([[self.Linear_exp(i,j) for j in self.y ]for i in self.x])   
    def linear_explicit_precompute(self):
        pre_lin_ker=SVC(kernel='precomputed')
        pre_lin_ker.fit(self.gram_explicit_calc(),self.y_train)
        return pre_lin_ker
    #using numpy functions
    def Linear_np(self,i,j):
        return np.dot(i,j.T)   
    def gram_np(self):
        return np.array([[self.Linear_np(i,j) for j in self.y ]for i in self.x])
    def linear_np_precompute(self):
        pre_lin_ker=SVC(kernel='precomputed')
        pre_lin_ker.fit(self.gram_np(),self.y_train)               
        return pre_lin_ker  
    #custom kernel
    def linear_kernel(self):
        lin_ker=SVC(kernel='linear')
        lin_ker.fit(self.x_train,self.y_train)
        #print('accuracy',accuracy_score(self.y_test,lin_ker.predict(self.x_test)))
        return lin_ker


class rbf():
    def __init__(self,x,y,y_train,x_train):
        self.x=x
        self.y=y
        self.y_train=y_train
        self.x_train =x_train
    #def gram(self,kern):
    #    return np.array([[kern(i,j) for j in self.y ]for i in self.x])   
    
   #pure calculation
    def RBF(self,x,y,fgamma=0.01):
        norm = 0
        for i in range(0,len(x)):
            norm += (x[i]-y[i]) * (x[i]-y[i])
        return math.exp(-(norm)*fgamma)    
    def rbf_explicit(self):
        return np.array([[self.RBF(i,j) for j in self.y ]for i in self.x])
    def rbf_explicit_precompute(self):
        pre_rbf_ker=SVC(kernel='precomputed',C=0.1)
        pre_rbf_ker.fit(self.rbf_explicit(),self.y_train)
        #gtest=np.dot(self.x_test,self.x_train.T)
        #self.pred=pre_lin_ker.predict(gtest)
        return pre_rbf_ker
    #using dot product
    def rbfkernelnp1(self,x,y,g=0.01):   
        norm = ((x-y).dot(x-y))
        return math.exp(-g * norm)
    def rbf_math(self):
        return np.array([[self.rbfkernelnp1(i,j) for j in self.y ]for i in self.x])
    def rbf_math_precompute(self):
        pre_rbf_ker=SVC(kernel='precomputed',C=0.1)
        pre_rbf_ker.fit(self.rbf_math(),self.y_train)       
        return pre_rbf_ker 
    #custom kernel
    def rbf_kernel(self):
        rbf_ker=SVC(kernel='rbf',gamma=0.01,C=0.1,shrinking = True, probability = False, tol = 0.001)
        rbf_ker.fit(self.x_train,self.y_train)
        #print('accuracy',accuracy_score(y_test,rbf_ker.predict(x_test)))
        return rbf_ker



class svc(linear,rbf,SVC):
    def __init__(self,kernel = ' '):
        self.kernel=kernel
        #super().__init__()
    def fit(self,x_train,y_train):
        self.x=np.array(x_train)
        self.y=np.array(x_train)
        self.y_train=y_train
        self.x_train =x_train
        self.val='dot product'
        super().__init__(self.x,self.y,self.y_train,self.x_train)
        if self.kernel == 'linear':
            self.k=self.linear_kernel()
            self.val= 'test values'
            return self.linear_kernel()
        elif self.kernel == 'linear_np_precompute':
            self.k=self.linear_np_precompute()
            return self.linear_np_precompute()
        elif self.kernel == 'linear_explicit_precompute':
            self.k=self.linear_explicit_precompute()
            return self.linear_explicit_precompute()
        elif self.kernel == 'rbf_math_precompute':
            self.k=self.rbf_math_precompute()
            return self.rbf_math_precompute()
        elif self.kernel == 'rbf_explicit_precompute':
            self.val= 'rbf'
            self.k=self.rbf_explicit_precompute()
            return self.rbf_explicit_precompute()
        elif self.kernel == 'rbf':
            self.k=self.rbf_kernel()
            self.val= 'test values'
            return self.rbf_kernel()
        
    def predict(self,xt,yt):
        self.x_test=np.array(xt)
        self.y_test=np.array(yt)
        gtest=np.dot(self.x_test,self.x_train.T)
        if self.val == 'test values':          
            pred=self.k.predict(self.x_test)
        elif self.kernel == 'rbf_explicit_precompute':
            rbftest = np.array([[self.RBF(i,j) for j in self.x]for i in self.x_test])
            #print([[self.rbfkernelnp1(i,j) for j in self.x_train]for i in self.x_test])
            pred=self.k.predict(rbftest)
        elif self.kernel == 'rbf_math_precompute':
            rbftest = np.array([[self.rbfkernelnp1(i,j) for j in self.x]for i in self.x_test])
            #print([[self.rbfkernelnp1(i,j) for j in self.x_train]for i in self.x_test])
            pred=self.k.predict(rbftest)
        else:
            pred=self.k.predict(gtest)
        return pred
    


l=svc(kernel='linear_np_precompute')
l.fit(x_tr,y_tr)
      

# # decision functions

model = SVC( kernel='linear', shrinking = True, probability = False, tol = 0.001)

model.fit(x_train, y_train)
pred=model.predict(x_test)
accuracy_score(y_test,pred)

# +

Decision_Function = model.decision_function(x_test)

Decision_Functions = model.decision_function(x_tests)

Decision_Functionb = model.decision_function(x_testb)


# +

h_data =ROOT.TH1D("hist","historgram",100,-2,2)
h_sig=ROOT.TH1D("hist","historgram",100,-2,2)
h_bkg=ROOT.TH1D("hist","historgram",100,-2,2)

# +

for i in range(len(Decision_Function)):
    h_data.Fill(Decision_Function[i])
for i in range(len(Decision_Functionb)):
    h_bkg.Fill(Decision_Functionb[i])
for i in range(len(Decision_Functions)):
    h_sig.Fill(Decision_Functions[i])

# -

mc = ROOT.TObjArray(3)
mc.Add(h_sig)
mc.Add(h_bkg)
fit = ROOT.TFractionFitter(h_data, mc)
# fit.Constrain(1,0.0,1.0) # constrain fraction 1 to be between 0 and 1
status = fit.Fit()
print("fit status: " , status)

# +

frac0=ctypes.c_double()
frac1=ctypes.c_double()
e0=ctypes.c_double()
e1=ctypes.c_double()
fit.GetResult(0, frac0, e0)
fit.GetResult(1, frac1, e1)
print(fit.GetChisquare()/fit.GetNDF(), 'Chi2/NDF goodness of fit')




# +


countdata, bindata = np.histogram(Decision_Function,bins=100,range=[-2,2], density=True)
countsig, binsig = np.histogram(Decision_Functions,bins=100,range=[-2,2], density=True)
countbkg, binbkg = np.histogram(Decision_Functionb,bins=100,range=[-2,2], density=True)
countfit, binfit = np.histogram(Decision_Function,bins=100,range=[-2,2], density=True)

# +

plt.rcParams.update({'figure.figsize':(7.5,5), 'figure.dpi':100})



# +

plt.hist(bindata[:-1], bindata, weights=countdata, label='data', alpha=0.5,histtype='step')
plt.legend()
plt.title('plots')
plt.show()

# +
  
    
plt.hist(binsig[:-1], binsig, weights=frac0*countsig, label='sig', alpha=0.5)
plt.hist(binbkg[:-1], binbkg, weights=frac1*countbkg, label='bkg', alpha=0.5)
plt.hist(binfit[:-1],binfit,  weights=countsig*frac0+countbkg*frac1,label='fit',alpha=0.5,histtype='step')
plt.legend()
plt.title('plots')
plt.show()
# -

a=5


# # FOM method

# + endofcell="--"

# -
def fomnewmethod_max(binsig,binbkg):
    
    k=0
    sig=[]
    bkg=[]
    for i in binsig:
        l=np.sum(countsig[k:100])
        k=k+1
        sig.append(l)
    k=0
    for i in binbkg:
        l= np.sum(countbkg[k:100])
        k=k+1
        bkg.append(l)
    a=np.add(sig,bkg)
    ss=1/(np.sqrt(bkg))
    b=np.add(bkg,ss*ss)
    mul=np.multiply(bkg,bkg)
    sum= (a*b)/(np.add(mul,a*ss*ss))
    first=a*np.log(sum)
    c=(sig*ss*ss)/(bkg*b)
    c1=np.add(1,c)
    clog=np.log(c1)
    sec=mul/(ss*ss)
    second=sec*clog
    final=2*(first-second)
    fpt=np.sqrt(final)
    countf, binf = np.histogram(fpt,bins=100,range=[-2,2])
    plt.plot(binf,fpt,marker='o',color='red')
    plt.savefig('fom_new')
    return np.nanmax(fpt)


def fomoldmethod_max(binsig,binbkg):
    k=0
    sig=[]
    bkg=[]
    for i in binsig:
        l=np.sum(countsig[k:100])
        k=k+1
        sig.append(l)
    k=0
    for i in binbkg:
        l= np.sum(countbkg[k:100])
        k=k+1
        bkg.append(l)
    k= sig/np.sqrt(np.add(sig,bkg))
    countf, binf = np.histogram(k,bins=100,range=[-2,2])
    plt.plot(binf,k,marker='o',color='red')
    plt.savefig('fom_old')
    return np.nanmax(k)

fomnewmethod_max(binsig,binbkg)



# --



len(bindata)


# + endofcell="--"

# -
def fomnewmethod_max(sig,bkg):
    a=np.add(sig,bkg)
    ss=1/(np.sqrt(bkg))
    b=np.add(bkg,ss*ss)
    mul=np.multiply(bkg,bkg)
    sum= (a*b)/(np.add(mul,a*ss*ss))
    first=a*np.log(sum)
    c=(sig*ss*ss)/(bkg*b)
    c1=np.add(1,c)
    clog=np.log(c1)
    sec=mul/(ss*ss)
    second=sec*clog
    final=2*(first-second)
    fpt=np.sqrt(final)
    countf, binf = np.histogram(fpt,bins=100,range=[-2,2])
    plt.plot(binf,fpt,marker='o',color='red')
    plt.savefig('fom_new')
    return np.nanmax(fpt)


def fomoldmethod_max(sig,bkg):
    k= sig/np.sqrt(np.add(sig,bkg))
    countf, binf = np.histogram(k,bins=100,range=[-2,2])
    plt.plot(binf,k,marker='o',color='red')
    plt.savefig('fom_old')
    return np.nanmax(k)

k=0
sig=[]
for i in binsig:
    l=np.sum(countsig[k:100])
    k=k+1
    sig.append(l)
#calculating Nb
k=0

bkg=[]
for i in binbkg:
    l= np.sum(countbkg[k:100])
    k=k+1
    bkg.append(l)
fomnewmethod_max(sig,bkg)



# --
