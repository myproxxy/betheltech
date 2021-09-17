import pandas as pd 
import os
from pdDataFrameTools import pdDataFrame
import numpy as np
import csv
import logging
import logging.handlers
from datetime import datetime

#setting up the logging output
defPath = os.path.dirname(__file__)
logFlName = '_analysis.log'
logging.basicConfig(filename=logFlName, filemode='a+', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

curTime = datetime.now()
logging.info(curTime)

#some minor Pandas settings...
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#load the original data
curPath = "/Users/hansjoerg.stark/Library/Mobile Documents/com~apple~CloudDocs/Privat/Weiterbildung/Bethel-DataScience/Module5_DataWrangling/DS Data Wrangling and Visualization/Assignments/FinalAssignmentDataWranglingAndVisualisation"
curPath = defPath
flName = "motor_vehicle_crashes.csv"
motorVehicleCrashesDf = pd.read_csv(os.path.join(curPath,flName))
#create a dataframe-object with specific methods for data wrangling
myDf = pdDataFrame(motorVehicleCrashesDf,curPath,'Crash_ID')

#log some basic information of the original data
rowCnt = myDf.curDf.shape[0]
colCnt = myDf.curDf.shape[1]
basicDSInfo = "Number of Records: {}, Number of Columns: {}".format(rowCnt,colCnt)
logging.info(basicDSInfo)

#export a small sample set transposed to get a first impression of the data, their type etc. in a csv file
numRecs = 5
firstImpressionDS = myDf.getMostPopulatedRecord(numRecs,'0_firstImpressionMotorVehicleCrashes_%i.csv' %numRecs)

#Checking for Nullvalues
myDf.check4NullValues('1_nullValueExamination.csv')

#provide a correct interpretable date field
myDf.properDateFormat('Crash_Date','CrashDateFormated','Crash_Time')

#change Yes/No Columns to 0/1 for correlation analysis or counting
#first get a list of all columns that have 'y'/'n' values: they all end with "_Fl"
substringInColTitle = '_Fl'
colNameList = myDf.getColnameList(substringInColTitle)
#add the additional new columns with 0/1 values to the dataset
success = myDf.yesNoTo01(colNameList,'Y')

#there are many columns with IDs that only contain foreign keys 
#that are of no use for analysis if the data they are referring to is not available
#therefore erase them to get a more performant dataset:
#get a thinned out dataframe by removing all columns with a certain substring
colSubStr4Removal = "_ID"
thinnedDfRaw = myDf.thinOutDataframeFromSubstringInName(colSubStr4Removal,excludeString="Crash_ID")
logging.info("\nthinned out dataframe by removing all columns with a certain substring. In this case: {}".format(colSubStr4Removal))
logging.info(thinnedDfRaw.head(n=5))
logging.info(len(thinnedDfRaw.columns))
thinnedDf = pdDataFrame(thinnedDfRaw,curPath,'Crash_ID')
#replace the existing df-object with the thinned out
myDf = thinnedDf

#get a Dataframe without any null values for easier analysis - 
#BUT: if sparsely populated this might end-up in an empty dataframe
completeDf = myDf.getCompleteDf('2_fullyFilledDataFrameMotorVehicleCrashes.csv')
#print(completeDf.info())
#unfortunately this returns an empty dataset - so we need to be less restrictive

#optimised version as sort of compromise: get a dataset with the top n populated records, n=500
numRecs = 500
completeDf = myDf.getMostPopulatedRecord(numRecs,'3_mostPopulatedDataFrameMotorVehicleCrashes_%i.csv' %numRecs)
#print(completeDf.info())
#create another df-object of the resulting dataframe with top n records
bestPossibleCompleteDfObj = pdDataFrame(completeDf,curPath,'Crash_ID')
#get rid of empty columns in this sample to size down and make it more manageable
bestPossibleCompleteDf = bestPossibleCompleteDfObj.dataframeWithoutEmptyColumns(expFlName='4_mostPopulatedDataFrameWOEmptyColsMotorVehicleCrashes.csv')
#print(bestPossibleCompleteDf.info())

#get columnlist of best fitting dataframe for sub-selecting from main dataframe
#Problem: with this selection potential loss of well filled columns
colList4Export = list(bestPossibleCompleteDf.columns)
#make sure we do not lose spatial information!
if not 'Longitude' in colList4Export:
    colList4Export.append('Longitude')
if not 'Latitude' in colList4Export:
    colList4Export.append('Latitude')

#create a subset of the original DF by chosing all records but only a selection of variables
chk = myDf.exportSubDataFrameBasedOnColumnList(colList4Export,'5_exportBestPossibleDataset_%i.csv' %numRecs)
#create downsized dataset:
#myDf.curDf = myDf.curDf[colList4Export]

#Alternative approach or to be added:
#get a thinned out dataframe: remove variables that have too many empty values
#optional threshold parameter in %; if not given default of 50 is used
nullThreshold = 40
thinnedDfRaw = myDf.thinOutDataframeFromSparselyPopulatedVariables(threshold=nullThreshold)
logging.info("\nremove variables that have too many empty values optional threshold parameter in %; if not given default of 50 is used; in this case: {}%".format(nullThreshold))
logging.info(thinnedDfRaw.head(n=2))
logging.info("Remaining number of columns after thinning out: {}".format(len(thinnedDfRaw.columns)))
colList4Export = list(thinnedDfRaw.columns)
#create a subset of the original DF by chosing all records but only a selection of variables
chk = myDf.exportSubDataFrameBasedOnColumnList(colList4Export,'6_exportOptimisedDataset_Threshold_%iPercent.csv' %nullThreshold)
#create downsized dataset:
myDf.curDf = myDf.curDf[colList4Export]

#Get global information on columns like min,max,type etc.
columnInfoList = myDf.getColSummary()
#motorVehicleCrashesDf.info()
#export column analysis for external inspection
exportFileName = "7_columnAnalysisMotorVehicleCrashesNP.csv"
np.savetxt(os.path.join(curPath,exportFileName), columnInfoList, delimiter=";", fmt='%s', header='')

#for ease of readability export just a few records in normal and transposed form
numRecs = 5
finalCompleteDf = myDf.getMostPopulatedRecord(numRecs,'8_sampleOptimisedDataFrameMotorVehicleCrashes_%i.csv' %numRecs)


#get a subset of original dataframe based on columnlist and export it
myColList = ['Crash_ID','CrashDateFormated','Longitude','Latitude']
#myColList = myDf.getColumnNameList()
subSetFlName = "subSetGeoMotorVehicleCrashes.csv"
subSetDf = myDf.subsetDataframe(myColList,subSetFlName,True)
logging.info("Subset '{}' exported.".format(subSetFlName))

#Final Check for Nullvalues
myDf.check4NullValues('9_finalExaminationThinnedDf.csv')


#scatterplot matrix:
#get a dataframe with only the columns of numeric type / quantitative values
#myNumDf = thinnedDf.numColsOnly()
myNumDf = myDf.numColsOnly()
logging.info("\nScatterplot")
logging.info("\nstarting matrix")
mySPMDf = pdDataFrame(myNumDf,curPath,'Crash_ID')
mySPMx = mySPMDf.scatterPlotMatrix(['Death_Cnt','Tot_Injry_Cnt','Non_Injry_Cnt'])
logging.info("matrix done")
logging.info(mySPMx)
logging.info("Scatterplot done")


logging.info("\n\n\n"+"*"*75)
logging.info("Finished")
print("\n\n\n"+"*"*75)
print("F i n i s h e d")
print("*"*75+"\n\n\n")
