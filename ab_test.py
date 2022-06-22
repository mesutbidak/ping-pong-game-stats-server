import pandas as pd
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# def feature(dataframe,sex_h,sex_a,weight_h,weight_a,height_h,height_a,clock,left_hand_h,left_hand_a,age_h,age_a,year_played_h,year_played_a):
#     dataframe["Sex_Home"] = sex_h
#     dataframe["Sex_Away"] = sex_a
#     dataframe["Weight_Home"] = weight_h
#     dataframe["Weight_Away"] = weight_a
#     dataframe["Height_Home"] = height_h
#     dataframe["Height_Away"] = height_a
#     dataframe["Clock"] = clock
#     dataframe["Left_hand_Home"] = left_hand_h
#     dataframe["Left_hand_Away"] = left_hand_a
#     dataframe["Age_Home"] = age_h
#     dataframe["Age_Away"] = age_a
#     dataframe["Year_Played_Home"] = year_played_h
#     dataframe["Year_Played_Away"] = year_played_a
#     dataframe["BMI_Home"] = dataframe["Weight_Home"] / ((dataframe["Height_Home"]/100)**2)
#     dataframe["BMI_Away"] = dataframe["Weight_Away"] / ((dataframe["Height_Away"]/100)**2)
#     dataframe["Total_Piezzo"] = dataframe["piezzo1"] + dataframe["piezzo2"] + dataframe["piezzo3"] + dataframe["piezzo4"]
#     # dataframe.loc[(dataframe["BMI_Home"] < 20), "BMI_CAT_Home"] = "Thin"
#     # dataframe.loc[((dataframe["BMI_Home"] >= 20) & (dataframe["BMI_Home"] < 25)), 'BMI_CAT_Home'] = "Normal"
#     # dataframe.loc[(dataframe["BMI_Home"] >= 25), "BMI_CAT_Home"] = "Fat"
#     # dataframe.loc[(dataframe["BMI_Away"] < 20), "BMI_CAT_Away"] = "Thin"
#     # dataframe.loc[((dataframe["BMI_Away"] >= 20) & (dataframe["BMI_Away"] < 25)), 'BMI_CAT_Away'] = "Normal"
#     # dataframe.loc[(dataframe["BMI_Away"] >= 25), "BMI_CAT_Away"] = "Fat"
#     # dataframe.loc[(dataframe["Age_Home"] < 18), "AGE_CAT_Home"] = "Child"
#     # dataframe.loc[((dataframe["Age_Home"] >= 18) & (dataframe["Age_Home"] < 30)), 'AGE_CAT_Home'] = "Young"
#     # dataframe.loc[(dataframe["Age_Home"] >= 30), "AGE_CAT_Home"] = "Adult"
#     # dataframe.loc[(dataframe["Year_Played_Home"] < 3), "Level_Home"] = "Beginner"
#     # dataframe.loc[((dataframe["Year_Played_Home"] >= 3) & (dataframe["Year_Played_Home"] < 5)), 'Level_Home'] = "Mid"
#     # dataframe.loc[(dataframe["Year_Played_Home"] >= 5), "Level_Home"] = "Professional"
#     # dataframe.loc[(dataframe["Year_Played_Away"] < 3), "Level_Away"] = "Beginner"
#     # dataframe.loc[((dataframe["Year_Played_Away"] >= 3) & (dataframe["Year_Played_Away"] < 5)), 'Level_Away'] = "Mid"
#     # dataframe.loc[(dataframe["Year_Played_Away"] >= 5), "Level_Away"] = "Professional"
#     return dataframe
#sex_h,sex_a,weight_h,weight_a,height_h,height_a,clock,left_hand_h,left_hand_a,age_h,age_a,year_played_h,year_played_a




# BÜTÜN TXT DOSYASINA BURADAN İTİBARENKİ İŞLEM UYGULANACAK YUKARISI SADECE VERİYİ BİRLEŞTİRME İÇİN HAZIRLANAN KOD

#A ŞAĞIDA YER ALAN KÜTÜPHANELERİ
# pip install scipy
# İLE İNDİREBİLİRSİN

from scipy.stats import shapiro, levene, ttest_ind,mannwhitneyu

# def ab_test(dataframe,col,target):
#     test_stat, pvalue1 = shapiro(dataframe.loc[dataframe[col] == dataframe[col].unique()[0], target])
#     test_stat, pvalue2 = shapiro(dataframe.loc[dataframe[col] == dataframe[col].unique()[1], target])
#     if pvalue1 > 0.05 and pvalue2 > 0.05:
#         test_stat, pvalue_levene = levene(dataframe.loc[dataframe[col] == dataframe[col].unique()[0], target],
#                                    dataframe.loc[dataframe[col] == dataframe[col].unique()[1], target])
#         if pvalue_levene > 0.05:
#             test_stat, pvalue_ttest_ind = ttest_ind(dataframe.loc[dataframe[col] == dataframe[col].unique()[0], target],
#                                           dataframe.loc[dataframe[col] == dataframe[col].unique()[1], target],
#                                           equal_var=True)
#             if pvalue_ttest_ind > 0.05:
#                 return "iki grup ortalamaları arasında ist ol.anl.fark yoktur."
#         else:
#             test_stat, pvalue_ttest_ind_2 = ttest_ind(dataframe.loc[dataframe[col] == dataframe[col].unique()[0], target],
#                                                     dataframe.loc[dataframe[col] == dataframe[col].unique()[1], target],
#                                                     equal_var=False)
#             if pvalue_ttest_ind_2 < 0.05:
#                 return "iki grup ortalamaları arasında ist ol.anl.fark yoktur."
#             else:
#                 return "iki grup ortalamaları arasında ist ol.anl.fark vardır."

#     elif pvalue1 < 0.05 or pvalue2 < 0.05:
#         test_stat, pvalue_man = mannwhitneyu(dataframe.loc[dataframe[col] == dataframe[col].unique()[0], target],
#                                          dataframe.loc[dataframe[col] == dataframe[col].unique()[1], target])
#         if pvalue_man < 0.05:
#             return "iki grup ortalamaları arasında ist ol.anl.fark yoktur."
#         else:
#             return "iki grup ortalamaları arasında ist ol.anl.fark vardır"

def ab_test(dataframe,col,target):
    test_stat, pvalue1 = shapiro(dataframe.loc[(dataframe["Data_Type"]=="test") , target])
    test_stat, pvalue2 = shapiro(dataframe[~dataframe[col].str.contains(dataframe[dataframe["Data_Type"] == "test"][col].unique()[0])][dataframe["Data_Type"] == "train"][target])
    if pvalue1 > 0.05 and pvalue2 > 0.05:
        test_stat, pvalue_levene = levene(dataframe.loc[(dataframe["Data_Type"]=="test"), target],
                                   dataframe[~dataframe[col].str.contains(dataframe[dataframe["Data_Type"] == "test"][col].unique()[0])][dataframe["Data_Type"] == "train"][target])
        if pvalue_levene > 0.05:
            test_stat, pvalue_ttest_ind = ttest_ind(dataframe.loc[(dataframe["Data_Type"]=="test") , target],
                                          dataframe[~dataframe[col].str.contains(dataframe[dataframe["Data_Type"] == "test"][col].unique()[0])][dataframe["Data_Type"] == "train"][target],
                                          equal_var=True)
            if pvalue_ttest_ind > 0.05:
                return "According to the parametric test results, there is no statistically significant difference between the two groups."
        else:
            test_stat, pvalue_ttest_ind_2 = ttest_ind(dataframe.loc[(dataframe["Data_Type"]=="test") , target],
                                                    dataframe[~dataframe[col].str.contains(dataframe[dataframe["Data_Type"] == "test"][col].unique()[0])][dataframe["Data_Type"] == "train"][target],
                                                    equal_var=False)
            if pvalue_ttest_ind_2 < 0.05:
                return "According to the weltch test results, there is no statistically significant difference between the two groups."
            else:
                return "There is a statistically significant difference between the two groups according to the weltch test result."

    elif pvalue1 < 0.05 or pvalue2 < 0.05:
        test_stat, pvalue_man = mannwhitneyu(dataframe.loc[(dataframe["Data_Type"]=="test") , target],
                                         dataframe[~dataframe[col].str.contains(dataframe[dataframe["Data_Type"] == "test"][col].unique()[0])][dataframe["Data_Type"] == "train"][target])
        if pvalue_man < 0.05:
            return "According to the mannwhitneyu test result, there is no statistically significant difference between the two groups."
        else:
            return "According to the mannwhitneyu test result, there is a statistically significant difference between the two groups."



