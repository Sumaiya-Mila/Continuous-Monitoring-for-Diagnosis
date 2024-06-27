#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 17:27:32 2022

@author: mila.s
"""

import math
import seaborn as sns
import pandas as pd
from csv import reader,writer
from pathlib import Path
import vaex
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif,f_classif
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier
import random



class Generate_Visit_Occurrence_Data:
    def __init__(self,
                 path_to_preprocessed_data_folder=None):

        if path_to_preprocessed_data_folder is None:
            path_to_preprocessed_data_folder = r"/home/UFAD/mila.s/mila_trident/preprocessed_files"

        self.path_to_preprocessed_data_folder = path_to_preprocessed_data_folder
        self.path_to_Visit_Occurrence_folder = r"/home/UFAD/mila.s/mila_trident/preprocessed_files/Files_with_Visit_Occurrence"

        Path(self.path_to_preprocessed_data_folder).mkdir(parents=True, exist_ok=True)

        self.GeneratePatient_VS_FeatureValue()
        # self.GenerateTargetValue()

        return

    def GeneratePatient_VS_FeatureValue(self):

        measurements_df = vaex.open(self.path_to_Visit_Occurrence_folder + '/covid_test_during_monitorable_measurements.hdf5')
        measurements_df = measurements_df["MEASUREMENT_SOURCE_VALUE","PERSON_ID","VALUE_AS_NUMBER"]

        patient_wise_mean_measurement_df = measurements_df.groupby(['PERSON_ID','MEASUREMENT_SOURCE_VALUE'],
                                                                   agg=vaex.agg.mean('VALUE_AS_NUMBER'))  #storing min/max/mean value of each measurement of a patient for all patients
        patient_wise_mean_measurement_df = patient_wise_mean_measurement_df.to_pandas_df()
        patient_wise_mean_measurement_df.to_csv(self.path_to_Visit_Occurrence_folder +
                                                '/patient_wise_mean_measurement_df.csv',index=None)
        del patient_wise_mean_measurement_df
        del measurements_df


        grouped_measurements_df = pd.read_csv(self.path_to_Visit_Occurrence_folder +
                                              '/patient_wise_mean_measurement_df.csv')

        covid_measurement_list = list(['TEMPERATURE','HEART RATE','MAP - Cuff','RESP RATE','SPO2'])    #storing the names of the measurements/ features

        patient_data = pd.read_csv(self.path_to_Visit_Occurrence_folder + "/patient_id_with_covid_result.csv") #reading covid tested patient id
        patient_list = list(patient_data.iloc[:, 0])
       

        count = 1
        for individual_patient in patient_list:
            selected_patient_df = grouped_measurements_df[grouped_measurements_df.PERSON_ID == individual_patient]  #at each time, taking one individual patient id

            measurements_array = [0] * 7
            measurements_array[-1] = individual_patient

            length = selected_patient_df.shape[0] # selected patient df has all the measurements of that patient

            for i in range(0, length):
                each_row_df = selected_patient_df.iloc[i]

                indx = covid_measurement_list.index(each_row_df['MEASUREMENT_SOURCE_VALUE'])    #for each row of a patient, takes the measurement source value name, finds it's index
                mean_value = each_row_df['VALUE_AS_NUMBER_mean']
                mean_nan = float(mean_value)

                #storing the measurement value in that particular index 
                if (np.isnan(mean_nan)):
                    measurements_array[indx] = 1
                elif math.isinf(mean_nan):
                    measurements_array[indx] = 1
                else:
                    measurements_array[indx] = mean_value

            grouped_measurements_df = grouped_measurements_df[grouped_measurements_df.PERSON_ID != individual_patient]

            with open(self.path_to_Visit_Occurrence_folder + r"/patient_vs_feature_value_mean.csv", 'a', newline='') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(measurements_array)
                f_object.close()

            del measurements_array
            print('patient saved', count)
            count = count + 1

        return

    def GenerateTargetValue(self):

        covid_positive_patient_ids_df = pd.read_csv(self.path_to_preprocessed_data_folder +
                                                    '/covid_positive_patient_ids.csv', sep='\t')
        covid_positive_patient_ids_df = covid_positive_patient_ids_df.to_numpy()

        covid_negative_patient_ids_df = pd.read_csv(self.path_to_preprocessed_data_folder +
                                                    '/covid_negative_patient_ids.csv', sep='\t')
        covid_negative_patient_ids_df = covid_negative_patient_ids_df.to_numpy()

        #     covid positive target value = 1
        #     covid negative target value = 0
        #     covid unknown target value = 2

        y = []
        patient_id = []

        with open(self.path_to_Visit_Occurrence_folder + r"/patient_id_with_covid_result.csv", 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)

            for row in csv_reader:
                a = int(row[0])
                patient_id.append(a)

                if a in covid_positive_patient_ids_df:
                    y.append(1)

                elif a in covid_negative_patient_ids_df:
                    y.append(0)

                else:
                    y.append(2)

        df = pd.DataFrame(list(zip(patient_id, y)),
                          columns=['PERSON_ID', 'COVID_STATE'])

        df.to_csv(self.path_to_Visit_Occurrence_folder + '/patient_with_covid_test_state.csv', index=None)
        read_obj.close()
        return


# =============================================================================
# Class for various feature selection algorithm and classifier
# =============================================================================
'''
class FeatureSelectionAlgorithm:
    def __init__(self,
                 path_to_SelectedFeature_data_folder=None,
                 path_to_preprocessed_data_folder=None,
                 path_to_Train_Test_data_folder=None):

        if path_to_SelectedFeature_data_folder is None:
            path_to_SelectedFeature_data_folder = r"/home/UFAD/mila.s/mila_trident/preprocessed_files/Selected_Feature"

        if path_to_preprocessed_data_folder is None:
            path_to_preprocessed_data_folder = r"/home/UFAD/mila.s/mila_trident/preprocessed_files"

        if path_to_Train_Test_data_folder is None:
            path_to_Train_Test_data_folder = r"/home/UFAD/mila.s/mila_trident/preprocessed_files/Train_Test_Data"
            
        self.path_to_SelectedFeature_data_folder = path_to_SelectedFeature_data_folder
        self.path_to_preprocessed_data_folder = path_to_preprocessed_data_folder
        self.path_to_Train_Test_data_folder = path_to_Train_Test_data_folder

        # self.Split_Data()
        # self.Get_Train_Test_Data()
        # self.Split_UnderSampled_Data()
        
        # self.Get_UnderSampled_Data()
        # self.ensamble()
        
        # self.Information_Gain()
        #self.Anova()
        # self.Feature_Importance()
        #self.FFS()
        #self.Accuracy_Check_Balanced_Patient()
        # self.Visualize_Target()
        #self.Imbalanced_Classifier()
        
        # self.baseline()
        return
    
    def Visualize_Target(self):
        Xdata = pd.read_csv(self.path_to_preprocessed_data_folder +'/patient_vs_feature_value_merged.csv')
        
        g = sns.countplot(Xdata['Covid_State'])
        g.set_xticklabels(['Covid Negative', 'Covid Positive'])
        total = float(len(Xdata['Covid_State']))
        
        plt.figure(figsize =(15,6))
        plt.rcParams.update({'font.size': 10})
        
        for p in g.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height()/total)
            x = p.get_x() + p.get_width()
            y = p.get_height()
            g.annotate(percentage, (x, y),ha='right', va = 'bottom')
        plt.show()
              
        return
    
    def Split_UnderSampled_Data(self):
        Xdata = pd.read_csv(self.path_to_preprocessed_data_folder +'/patient_vs_feature_value_merged.csv')
        ydata = pd.read_csv(self.path_to_preprocessed_data_folder + '/patient_with_covid_test_state.csv')
        
        X = Xdata.iloc[:, :]
        y = ydata.iloc[:, -1]
        
        ros = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
        x_ros, y_ros = ros.fit_resample(X, y)

        print('original dataset shape:', Counter(y))
        print('Resample dataset shape', Counter(y_ros))
        
        X_train, X_test, y_train, y_test = train_test_split(x_ros, y_ros ,
                                                             test_size= 0.2, random_state=42)
        
        X_train.to_csv(self.path_to_Train_Test_data_folder+ '/X_train_Under_Sample.csv', index=None)
        X_test.to_csv(self.path_to_Train_Test_data_folder+ '/X_test_Under_Sample.csv', index=None)
        y_train.to_csv(self.path_to_Train_Test_data_folder+ '/y_train_Under_Sample.csv', index=None)
        y_test.to_csv(self.path_to_Train_Test_data_folder+ '/y_test_Under_Sample.csv', index=None)

        return
    
    def Get_UnderSampled_Data(self):
        self.X_train = pd.read_csv(self.path_to_Train_Test_data_folder+ '/X_train_final.csv')
        self.X_test = pd.read_csv(self.path_to_Train_Test_data_folder+ '/X_test_final.csv')
        self.y_train = pd.read_csv(self.path_to_Train_Test_data_folder + '/y_train_final.csv')
        self.y_test = pd.read_csv(self.path_to_Train_Test_data_folder + '/y_test_final.csv')

        self.y_train = self.y_train.iloc[:, -1]
        self.y_test = self.y_test.iloc[0:, -1]
        self.X_train = self.X_train.iloc[:,0:-1]
        self.X_test = self.X_test.iloc[:,0:-1]
        return
    
    def baseline(self):
        # sz = self.y_test.size
        # ypred = list(np.random.randint(low = 0,high=2,size=sz))
        # ypred_df = pd.DataFrame (ypred, columns = ['y_predicted-baseline'])
        # ypred_df.to_csv(self.path_to_Train_Test_data_folder+ '/y_predicted_Baseline.csv', index=None)
        ypred_df = pd.read_csv(self.path_to_Train_Test_data_folder + '/y_predicted_Baseline.csv')
        ypred_df = ypred_df.iloc[:,-1]
        
        print('result for baseline: ')
        print('Accuracy score:',accuracy_score(self.y_test, ypred_df))
        print('F1 score:',f1_score(self.y_test, ypred_df))
        print('Precision:',precision_score(self.y_test, ypred_df))
        print('Recall:',recall_score(self.y_test, ypred_df))
        return
    
    def ensamble(self):
        selected_feature = pd.read_csv(self.path_to_SelectedFeature_data_folder +"/monitorable_features.csv")
        selected_feature_list = list(selected_feature.iloc[:, 0])
        
        self.X_train = self.X_train[selected_feature_list]
        self.X_test = self.X_test[selected_feature_list]
        # self.X_train = self.X_train.iloc[:,0:400]
        # self.X_test = self.X_test.iloc[:,0:400]
        
        
        # cv = KFold(n_splits=3, shuffle=True, random_state=1)
        # space = dict()
        # space['n_estimators'] = [10, 100, 500]
        
        model1 = ExtraTreesClassifier()
        model2 = RandomForestClassifier()
        model3= GradientBoostingClassifier()
        model4= AdaBoostClassifier()
        model5= LogisticRegression(max_iter = 1000, solver = 'liblinear')
        
        # grid2 = GridSearchCV(model2, space, scoring='f1', cv=cv, refit=True)
        # result2 = grid2.fit(self.X_train, self.y_train)
        # 	# get the best performing model fit on the whole training set
        # best_model2 = result2.best_estimator_
        # 	# evaluate model on the hold out dataset
        # pred2 = best_model2.predict(self.X_test)
        # 	# evaluate the model
        # acc = accuracy_score(self.y_test, pred2)

        
        model1.fit(self.X_train, self.y_train)
        model2.fit(self.X_train, self.y_train)
        model3.fit(self.X_train, self.y_train)
        model4.fit(self.X_train, self.y_train)
        model5.fit(self.X_train, self.y_train)
        
        pred1=model1.predict(self.X_test)
        pred2=model2.predict(self.X_test)
        pred3=model3.predict(self.X_test)
        pred4=model4.predict(self.X_test)
        pred5=model5.predict(self.X_test)
        
        finalpred=(pred1+pred2+pred3+pred4+pred5)/5
        import statistics
        # finalpred = np.array([])

        for i in range(0,len(self.y_test)):
            finalpred[i] = round(finalpred[i])
            # finalpred = np.append(finalpred, statistics.mode([pred2[i], pred3[i],pred4[i],pred5[i]]))
            # print(self.y_test[i],' ,  ',finalpred[i])
        print('result for monitorable features: ')
        print('Accuracy score:',accuracy_score(self.y_test, finalpred))
        print('F1 score:',f1_score(self.y_test, finalpred))
        print('Precision:',precision_score(self.y_test, finalpred))
        print('Recall:',recall_score(self.y_test, finalpred))
        tn, fp, fn, tp = confusion_matrix(self.y_test, finalpred).ravel()
        # print(tn,', ',tp,', ',fn,', ',fp)
        print('\n')
        return
    
    def Imbalanced_Classifier(self):
        
        selected_feature = pd.read_csv(self.path_to_SelectedFeature_data_folder +"/Information_Gain_with_min_max.csv")
        selected_feature_list = list(selected_feature.iloc[:, 0])
        
        self.X_train = self.X_train[selected_feature_list]
        self.X_test = self.X_test[selected_feature_list]
        # self.X_train = self.X_train.iloc[:,0:400]
        # self.X_test = self.X_test.iloc[:,0:400]
        
        rfc = RandomForestClassifier()
        rfc.fit(self.X_train, self.y_train)
                

        # predict
        rfc_predict = rfc.predict(self.X_test)# check performance
        print('ROCAUC score:',roc_auc_score(self.y_test, rfc_predict))
        print('Accuracy score:',accuracy_score(self.y_test, rfc_predict))
        print('F1 score:',f1_score(self.y_test, rfc_predict))
        print('Precision:',precision_score(self.y_test, rfc_predict))
        print('Recall:',recall_score(self.y_test, rfc_predict))
        
        
        # print('Actual Covid Positive Count:',list(self.y_test).count(1))
        # print('Predicted Covid Positive Count::',list(rfc_predict).count(1))
        # chk = self.X_test.iloc[0:2,:]
        # y = self.y_test.iloc[0:2]
        # chk = chk[selected_feature_list]
        # predict = rfc.predict(chk)
        # print('predicted', predict)
        # print('original', list(y))
        return

    def Split_Data(self):
        Xdata = pd.read_csv(self.path_to_preprocessed_data_folder +'/patient_vs_feature_value_merged.csv')
        ydata = pd.read_csv(self.path_to_preprocessed_data_folder + '/patient_with_covid_test_state.csv')

        self.X = Xdata.iloc[:, :]
        self.y = ydata.iloc[:, :]

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size= 0.33, random_state=42)

        X_train.to_csv(self.path_to_Train_Test_data_folder+ '/X_train2.csv', index=None)
        X_test.to_csv(self.path_to_Train_Test_data_folder+ '/X_test2.csv', index=None)
        y_train.to_csv(self.path_to_Train_Test_data_folder+ '/y_train2.csv', index=None)
        y_test.to_csv(self.path_to_Train_Test_data_folder+ '/y_test2.csv', index=None)

    def Get_Train_Test_Data(self):
        self.X_train = pd.read_csv(self.path_to_Train_Test_data_folder+ '/X_train2.csv')
        self.X_test = pd.read_csv(self.path_to_Train_Test_data_folder+ '/X_test2.csv')
        self.y_train = pd.read_csv(self.path_to_Train_Test_data_folder + '/y_train2.csv')
        self.y_test = pd.read_csv(self.path_to_Train_Test_data_folder + '/y_test2.csv')

        self.X_train = self.X_train.iloc[:, 0:-3]
        self.X_test = self.X_test.iloc[:, 0:-3]
        self.y_train = self.y_train.iloc[:, -1]
        self.y_test = self.y_test.iloc[:, -1]
        return

    def Information_Gain(self):

        bestfeatures = SelectKBest(score_func=mutual_info_classif, k=30)
        fit = bestfeatures.fit(self.X_train, self.y_train)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(self.X_train.columns)
        # concat two dataframes for better visualization
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Features/Measurements', 'Score']
        featureScores = featureScores.nlargest(30, 'Score')
        # featureScores.to_csv(self.path_to_SelectedFeature_data_folder + '/Information_Gain_with_min_max.csv', index=None)
        print(featureScores.nlargest(30, 'Score'))

        return
    
    def Feature_Importance(self):

        model = ExtraTreesClassifier()
        model.fit(self.X_train, self.y_train)

        feat_importances = pd.Series(model.feature_importances_, index=self.X_train.columns)
        feat_importances.nlargest(25).plot(kind='barh')
        # plt.figure(figsize =(20,10))
        # plt.rcParams.update({'font.size': 25})
        feat_importances = feat_importances.nlargest(30)
        df = pd.DataFrame({'Features/Measurements': feat_importances.index, 'Score': feat_importances.values})
        # df.to_csv(self.path_to_SelectedFeature_data_folder + '/feature_importances_with_min_max.csv', index=None)
        plt.show()

        return

    def FFS(self):
        
        clf = RandomForestClassifier(n_estimators=50, max_depth = 2,
                                   n_jobs=-1)
        sfs1 = sfs(clf, k_features=30, forward=True, verbose=2, scoring='accuracy')
        sfs1 = sfs1.fit(self.X_train, self.y_train)
        feat_names = list(sfs1.k_feature_names_)
        print(feat_names)
        feat_names_df = pd.DataFrame(feat_names, columns = ['selected features'])
        feat_names_df.to_csv(self.path_to_SelectedFeature_data_folder + '\FFS2.csv', index=None)

        return

    def Anova(self):

        bestfeatures = SelectKBest(score_func=f_classif, k=30)
        fit = bestfeatures.fit(self.X_train, self.y_train)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(self.X_train.columns)
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Features/Measurements', 'Score']
        featureScores = featureScores.nlargest(30, 'Score')
        featureScores.to_csv(self.path_to_SelectedFeature_data_folder + '\Anova.csv', index=None)
        print(featureScores.nlargest(30, 'Score'))

        return

 
'''  
    