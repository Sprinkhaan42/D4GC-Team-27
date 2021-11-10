# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fgBEb-G6fjAmA3s53Rs6xGSfo4Ey7rsE
"""

#import libraries
import pandas
import numpy
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle

##Set up the data file and the parameters you want to use
File = pandas.read_csv("Dataset 4 - Emissions _ Energy Consumption.csv",sep=",")
Data = File[["country","iso_alpha3","region","sub_region","year","co2","co2_growth_abs","co2_per_capita","share_global_co2","co2_per_gdp","total_ghg","ghg_per_capita","primary_energy_consumption","energy_per_capita","energy_per_gdp"]]

##Set variable you wish to predict
#Label = "G3"

##Set up two tables, one table with features and one with labels
#Returns a dataframe without G3
X = numpy.array(Data.drop([Label], 1))
#Set the dataframe you wish to test against
y = numpy.array(Data[Label])
## Take all the attributes and labels and split them into four arrays
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)