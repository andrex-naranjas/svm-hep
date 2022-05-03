import numpy as np
import pandas as pd
import multiprocessing
import copy
from sklearn.metrics import auc,roc_auc_score
from sklearn.inspection import permutation_importance
import shap
import pickle
import ROOT
import ctypes


# framework includes
import data_utils as du


import data_preparation as dp


class model_saving:

    def __init__(self, model):
        # input an already trained model
        self.m_model=model

    def save_model(self, file_name="model.pickle"):
        print("Saving the model into a pickle (binary) file")
        model_pickle = self.m_model
        pickle.dump(model_pickle, open(file_name,'wb'))
        # pickle.dump([sv_t,average,f1_score], open('complete_ml.pkl', 'wb' ))
        # sv_t,average,f1_score = pickle.load(open('complete_ml.pkl','rb'))

    def load_model(self, file_name="model.pickle"):
        with open(file_name, 'rb') as f:
            print("Loading model from a pickle (binary) file")
            return pickle.load(f)


class variable_importance:

    def __init__(self, model, X_train, Y_train, X_test, Y_test, global_auc=0.5, roc_area="prob"):
        self.m_model = model
        self.m_x_train = X_train
        self.m_y_train = Y_train
        self.m_x_test = X_test
        self.m_y_test = Y_test
        self.m_global_auc = global_auc
        self.m_roc_area = roc_area
        
    def data_split(self):
        x_trains = self.m_x_train.loc[self.m_x_train['isSignal']==1].drop(['isSignal'],axis=1)
        y_trains = self.m_y_train['isSignal'].loc[self.m_x_train['isSignal']==1]
        x_tests = self.m_x_test.loc[self.m_x_test['isSignal']==1].drop(['isSignal'],axis=1)
        y_tests = self.m_y_test['isSignal'].loc[self.m_x_test['isSignal']==1]
       
        x_trainb = self.m_x_train.loc[self.m_x_train['isSignal']==-1]
        y_trainb = self.m_y_train['isSignal'].loc[self.m_x_train['isSignal']==-1]
        x_testb = self.m_x_test.loc[self.m_x_test['isSignal']==-1]
        y_testb = self.m_y_test['isSignal'].loc[self.m_x_test['isSignal']==-1]
        
        self.m_model.fit(self.m_x_train, self.m_y_train)
        
        Decision_Function = self.m_model.decision_function(self.m_x_test)
        Decision_Functions = self.m_model.decision_function(x_tests)
        Decision_Functionb = self.m_model.decision_function(x_testb)
        
        h_data = ROOT.TH1D("hist","historgram",100,-2,2)
        h_sig = ROOT.TH1D("hist","historgram",100,-2,2)
        h_bkg = ROOT.TH1D("hist","historgram",100,-2,2)
        
        for i in range(len(Decision_Function)):
            h_data.Fill(Decision_Function[i])
        for i in range(len(Decision_Functionb)):
            h_bkg.Fill(Decision_Functionb[i])
        for i in range(len(Decision_Functions)):
            h_sig.Fill(Decision_Functions[i])
            
            
        mc = ROOT.TObjArray(3)
        mc.Add(h_sig)
        mc.Add(h_bkg)
        fit = ROOT.TFractionFitter(h_data, mc)
        # fit.Constrain(1,0.0,1.0) # constrain fraction 1 to be between 0 and 1
        status = fit.Fit()
        print("fit status: " , status)
        frac0=ctypes.c_double()
        frac1=ctypes.c_double()
        e0=ctypes.c_double()
        e1=ctypes.c_double()
        fit.GetResult(0, frac0, e0)
        fit.GetResult(1, frac1, e1)
        print(fit.GetChisquare()/fit.GetNDF(), 'Chi2/NDF goodness of fit')


        countdata, bindata = np.histogram(Decision_Function,bins=100,range=[-2,2], density=True)
        countsig, binsig = np.histogram(Decision_Functions,bins=100,range=[-2,2], density=True)
        countbkg, binbkg = np.histogram(Decision_Functionb,bins=100,range=[-2,2], density=True)
        countfit, binfit = np.histogram(Decision_Function,bins=100,range=[-2,2], density=True)
         
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


        fig_of_merit = self.fom_method(sig,bkg)
        fig_of_merit_simple = self.fom_simple_method_max(sig,bkg)
        print('fig_of_merit :',fig_of_merit)
        print('fig_of_merit_simple :',fig_of_merit_simple)
        
    
    def fom_method(self,sig,bkg):
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
        #plt.plot(binf,fpt,marker='o',color='red')
        #plt.savefig('fom_new')
        return np.nanmax(fpt)
    
    def fom_simple_method_max(sig,bkg):
        k= sig/np.sqrt(np.add(sig,bkg))
        countf, binf = np.histogram(k,bins=100,range=[-2,2])
        plt.plot(binf,k,marker='o',color='red')
        plt.savefig('fom_old')
        return np.nanmax(k)

    def roc_for_variable_set(self, variable):
        # perform the model training/testing skippinf feature "variable"
        method  = self.m_model
        X_train = self.m_x_train.drop(variable, axis=1)
        Y_train = self.m_y_train
        X_test  = self.m_x_test.drop(variable, axis=1)
        Y_test  = self.m_y_test

        method.fit(X_train, Y_train)
        if self.m_roc_area=="absv":
            y_thresholds = method.decision_thresholds(X_test, glob_dec=True)
            TPR, FPR = du.roc_curve_adaboost(y_thresholds, Y_test)
            method.clean()
            return auc(FPR,TPR)
        elif self.m_roc_area=="prob":
            Y_pred_prob = method.predict_proba(X_test)[:,1]
            return roc_auc_score(Y_test, Y_pred_prob)
        elif self.m_roc_area=="deci":
            Y_pred_dec = method.decision_function(X_test)
            return roc_auc_score(Y_test, Y_pred_dec)


    def one_variable_out(self):
        # Calculate the importance using the loss in AUC if a variable is removed
        p = multiprocessing.Pool(None, maxtasksperchild=1)
        variables = list(self.m_x_train.columns)
        print(variables)
        results = p.map(self.roc_for_variable_set, variables) # [[v for v in method.variables if v != variable] for variable in method.variables])
        sorted_variables_with_results = list(sorted(zip(variables, results), key=lambda x: x[1]))
        print("Variable importances calculated using loss if variable is removed")
        for variable, auc in sorted_variables_with_results:
            print(variable, self.m_global_auc - auc)


    def recursive_one_variable_out(self):
        # Calculate the importance using the loss in AUC if a variable is removed recursively.
        # p = multiprocessing.Pool(None, maxtasksperchild=1)
        # results = p.map(roc_for_variable_set, [[v for v in method.variables if v != variable] for variable in method.variables])
        # sorted_variables_with_results = list(sorted(zip(method.variables, results), key=lambda x: x[1]))
        p = multiprocessing.Pool(None, maxtasksperchild=1)
        variables = list(self.m_x_train.columns)
        results = p.map(self.roc_for_variable_set, variables)
        sorted_variables_with_results = list(sorted(zip(variables, results), key=lambda x: x[1])) 

        removed_variables_with_results = sorted_variables_with_results[:1]
        remaining_variables = [v for v, r in sorted_variables_with_results[1:]]
        while len(remaining_variables) > 1:
            results = p.map(self.roc_for_variable_set,
                            [[v for v in remaining_variables if v != variable] for variable in remaining_variables])
            sorted_variables_with_results = list(sorted(zip(remaining_variables, results), key=lambda x: x[1]))
            removed_variables_with_results += sorted_variables_with_results[:1]
            remaining_variables = [v for v, r in sorted_variables_with_results[1:]]
        removed_variables_with_results += sorted_variables_with_results[1:]
        
        print("Variable importances calculated using loss if variables are recursively removed")
        last_auc = self.m_global_auc
        for variable, auc in removed_variables_with_results:
            print(variable, last_auc - auc)
            last_auc = auc


    def permutation_feature(self):
        results = permutation_importance(self.m_model, self.m_x_train, self.m_y_train, n_repeats=50, random_state=None)
        variables = list(self.m_x_train.columns)
        sorted_variables_with_results = list(sorted(zip(variables, results.importances_mean), key=lambda x: x[1], reverse=True))
        print("Variable importances calculated using permutation importance")
        for variable, importance in sorted_variables_with_results:
            print(variable, importance)


    def shapley_values(self):

        method  = self.m_model
        X_train = self.m_x_train
        Y_train = self.m_y_train
        X_test  = self.m_x_test
        Y_test  = self.m_y_test

        method.fit(X_train, Y_train)
        if self.m_roc_area=="absv":
            f = lambda x: method.decision_thresholds(x, glob_dec=True)
            method.clean()            
        elif self.m_roc_area=="prob":
            f = lambda x: method.predict_proba(x)[:,1]
        elif self.m_roc_area=="deci":
            f = lambda x: method.decision_function(x)
            
        med = X_train.median().values.reshape((1,X_train.shape[1]))
        explainer = shap.Explainer(f, med)
        shap_values = explainer(X_test.iloc[0:1000,:])
        print(type(shap_values), len(shap_values))
        import matplotlib.pyplot as plt
        fig = shap.plots.heatmap(shap_values, show=False) # shap.summary_plot(shap_values, X_test.iloc[0:1000,:])
        f = plt.gcf()
        f.savefig('heatmap.png')
        plt.close(f)


    def shapley_values_second(self):
        # not well understood yet
        method = self.m_model

        X_train = self.m_x_train
        Y_train = self.m_y_train
        X_test  = self.m_x_test
        Y_test  = self.m_y_test

        method.fit(X_train, Y_train)
        if self.m_roc_area=="absv":
            f = lambda x: method.decision_thresholds(x, glob_dec=True)
            method.clean()            
        elif self.m_roc_area=="prob":
            f = lambda x: method.predict_proba(x)[:,1]
        elif self.m_roc_area=="deci":
            f = lambda x: method.decision_function(x)

        # second shapley values
        explainer = shap.KernelExplainer(method.decision_function, X_train)
        shap_values = explainer.shap_values(X_test.iloc[0:1000,:])
        print(type(shap_values), len(shap_values))
        print(shap_values)
        # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)

        # fig = shap.plots.heatmap(shap_values, show=False) # shap.summary_plot(shap_values, X_test.iloc[0:1000,:])
        # f = plt.gcf()
        # f.savefig('heatmap_dos.png')
        # plt.close(f)
        # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
        print("hello world!")


class grid_search:

    def __init__(self, model):
        self.m_model = model


    def cross_validation(self):
        # k-fold cross validation method
        kf = KFold(n_splits=5, shuffle=False, random_state=None)
        acc_arr = ([])
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]                
            model.fit(X_train, Y_train)
            acc_arr = np.append(acc_arr, self.m_model.score(X_test, Y_test))

        return acc_arr





