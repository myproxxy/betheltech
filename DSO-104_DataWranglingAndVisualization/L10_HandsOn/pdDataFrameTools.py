import pandas as pd 
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import os, sys
import logging

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class pdDataFrame():
    def getDfName(self):
        name = ''
        try:
            name =[x for x in globals() if globals()[x] is self.curDf][0]
        except:
            pass
        return name 

    def __init__(self, dataFrameObj,defaultPath=None,dsID=None):
        self.curDf = dataFrameObj
        self.dfName = self.getDfName()
        self.dfID = dsID
        if defaultPath == None:
            self.defaultPath = os.path.dirname(__file__)
        else:
            self.defaultPath = defaultPath

    def getColnameList(self,substringInColTitle):
        #create and return a list of column-names of the dataframe 
        #that contain a certain substring in their name
        colNameList = []
        for (columnName, columnData) in self.curDf.iteritems():
            if columnName.find(substringInColTitle) != -1:
                colNameList.append(columnName)
        return colNameList

    def properDateFormat(self, actualDateColumnName, newDateColumnName, actualTimeColumnName=None):
        if actualTimeColumnName:
            self.curDf[newDateColumnName] = pd.to_datetime(self.curDf[actualDateColumnName] + " " + self.curDf[actualTimeColumnName])
        else:
            self.curDf[newDateColumnName] = pd.to_datetime(self.curDf[actualDateColumnName])

    def getColSummary(self):
        colSummary = []
        colSummary.append(["name","type","min","max","sum","mean","median","stdev"])
        sortedDf = self.curDf
        sortedDf.sort_index(axis=1, inplace=True)
        for (columnName, columnData) in self.curDf.iteritems():
            curMin = None
            curMax = None
            curSum = None
            curMean = None
            curMedian = None
            curStdev = None
            #print('Colunm Name : ', columnName)
            colObj = self.curDf[columnName]
            curDataTypeObj = self.curDf.dtypes[columnName]
            colType = curDataTypeObj.name
            if is_numeric_dtype(self.curDf.dtypes[columnName]):
                curMin = colObj.min(skipna=True)
                curMax = colObj.max(skipna=True)
                curSum = colObj.sum(skipna=True)
                curMean = colObj.mean(skipna=True)
                curMedian = colObj.median(skipna=True)
                curStdev = colObj.std(skipna=True)
            elif columnData[0]=='N' or columnData[0]=='Y':
                colType = "boolean"
            colSummary.append([columnName,colType,curMin,curMax,curSum,curMean,curMedian,curStdev])
        return colSummary
    
    def yesNoTo01(self,colNameList,trueTest):
        #this method adds a new column with only 0 and 1 based on an initial column given
        #as parameter that contains binary values such as "Y" and "N" or similar
        #second parameter stand for the "true"-value
        for colName in colNameList:
            binaryColName = colName+"_01"
            try:
                self.curDf[binaryColName]=(self.curDf[colName]==trueTest).astype(int)
                chk = True
            except:
                chk = False
        return chk


    def ZeroOneNoToYesNo(self,colNameList,trueTest):
        #this method adds a new column with only "Yes" and "No" based on an initial column given
        #as parameter that contains binary values such as "1" and "0"
        for colName in colNameList:
            binaryColName = colName+"_YN"
            try:
                self.curDf[binaryColName]=self.curDf['beta'].map({True: 'yes', False: 'no'})
                chk = True
            except:
                chk = False
        return chk


    def check4NullValues(self,exportFileName = None):
        if exportFileName == None:
            exportFileName = "nullValues_{}.csv".format(self.dfName)
        exportFileName = os.path.join(self.defaultPath,exportFileName)    
        #get statistics on missing values per variable
        nullDf = self.curDf.isnull().sum().to_frame('nulls')
        recCnt = self.countOfRowsInDataFrame()
        logging.info("\nTotal count of rows in Dataframe: {}".format(recCnt))
        nullDf['Percent'] = round((nullDf['nulls']/int(recCnt))*100,3)
        #export column analysis for external inspection
        nullDf.sort_index(axis=1, inplace=True)
        nullDf.to_csv(os.path.join(self.defaultPath,exportFileName))

    def countOfRowsInDataFrame(self):
        return len(self.curDf.index)

    def scatterPlotMatrix(self,colList=None,expFlName=None):
        if colList==None:
            spmDf = self.numColsOnly()
        else:
            colMatch = set(colList) & set(self.curDf.columns)
            spmDf = self.curDf[colMatch]
        if len(spmDf.columns)>12:
            logging.warning("\n{}\nToo many columns for scatterplot matrix (max 12)!! \n{}\n".format("*"*70,"*"*70))
            spmx = None
        else:
            if len(spmDf.columns)>0:
                spmx = pd.plotting.scatter_matrix(spmDf, alpha=0.2)
                if expFlName==None:
                    expFlName="scatterPlotMatrix.png"
                # Save the figure 
                expFlNamePath = os.path.join(self.defaultPath,expFlName)
                plt.savefig(expFlNamePath)
            else:
                logging.info("\n{}\nNo columns to create a scatterplot matrix (min 1)!! \n{}\n".format("*"*70,"*"*70))
            spmx = None

        return spmx

    def numColsOnly(self):
        #get only numeric variables
        numDf = self.curDf.select_dtypes(include=['float64','int64'])
        return numDf
        
    def thinOutDataframeFromSparselyPopulatedVariables(self,threshold=50):
        thinnedDf = self.curDf.copy()
        numOfRecs = self.countOfRowsInDataFrame()
        for column in self.curDf:
            nullCounts = self.curDf[column].isnull().sum()
            if ((1-nullCounts/numOfRecs)*100) < threshold:
                thinnedDf.drop([column],axis=1, inplace=True)
        return thinnedDf


    def thinOutDataframeFromSubstringInName(self,subString,excludeString=None):
        thinnedDf = self.curDf.copy()
        for (columnName, columnData) in self.curDf.iteritems():
            if columnName.find(subString) > -1 and columnName != excludeString:
                thinnedDf.drop([columnName],axis=1, inplace=True)
        return thinnedDf


    def transposeAndSaveAsCsv(self,localDf,expFlName,transpose=True):
        if not expFlName:
            expFlName = os.path.join(self.defaultPath,"subset_{}.csv".format(self.dfName))
        else:
            expFlName = os.path.join(self.defaultPath,expFlName)
        expFlNameT = expFlName.split(".csv")[0] + "_Transposed.csv"
        localDf.sort_index(axis=1, inplace=True)
        localDf.to_csv(expFlName)
        if transpose:
            if self.dfID:
                localDfT = localDf.set_index(self.dfID).transpose()
            else:
                localDfT = localDf.transpose()

            localDfT.to_csv(expFlNameT)


    def dataframeWithoutEmptyColumns(self,expFlName=False):
        nonNullColumnsList = [col for col in self.curDf.columns if self.curDf.loc[:, col].notna().any()]
        nonNullColumnsDf = self.curDf[nonNullColumnsList]
        self.transposeAndSaveAsCsv(nonNullColumnsDf,expFlName)
        '''
        if not expFlName:
            expFlName = os.path.join(self.defaultPath,"nonNullColumns_{}.csv".format(self.dfName))
        nonNullColumnsDf.sort_index(axis=1, inplace=True)
        if transpose:
            nonNullColumnsDf = nonNullColumnsDf.transpose()
        nonNullColumnsDf.to_csv(expFlName)
        '''
        return nonNullColumnsDf

    def exportSubDataFrameBasedOnColumnList(self, colList4Export,expFlName=False):
        self.transposeAndSaveAsCsv(self.curDf[colList4Export],expFlName,False)

    def subsetDataframe(self,colList,expFlName=False,save=False):
        if len(colList)>0:
            columnMatch = set(colList) & set(self.curDf.columns)
            subsetDf = self.curDf[columnMatch]
            if save:
                self.transposeAndSaveAsCsv(subsetDf,expFlName)
                '''
                if not expFlName:
                    expFlName = os.path.join(self.defaultPath,"subset_{}.csv".format(self.dfName))
                subsetDf.sort_index(axis=1, inplace=True)
                if transpose:
                    subsetDf = subsetDf.transpose()
                subsetDf.to_csv(expFlName)
                '''
        else:
            logging.warning("\nno Columnlist provided!\nNo subsetting of Dataframe!!")
            subsetDf = None
        return subsetDf

    def getColumnNameList(self):
        return list(self.curDf.columns)

    def getCompleteDf(self,expFlName=False):
        completeDf = self.curDf.dropna(how='all', axis=1)
        completeDf = completeDf.dropna()
        self.transposeAndSaveAsCsv(completeDf,expFlName)
        '''
        if not expFlName:
            expFlName = os.path.join(self.defaultPath,"completeDf_{}.csv".format(self.dfName))
        completeDf.sort_index(axis=1, inplace=True)
        if transpose:
            completeDf = completeDf.transpose()
        completeDf.to_csv(expFlName)
        '''
        if len(completeDf) == 0:
            logging.warning("*"*75)
            logging.warning("ATTENTION: THERE IS NO SUBSET OF DATAFRAME WITHOUT ANY MISSING VALUE!!")
            logging.warning("*"*75)
        return completeDf

    def getMostPopulatedRecord(self,recCnt=1,expFlName=False):
        recsCntMissingValsList = self.curDf.isnull().sum(axis=1).tolist()
        if recCnt >=1:
            maxIndex = sorted(range(len(recsCntMissingValsList)), key=lambda i: recsCntMissingValsList[i], reverse=True)[:recCnt]
        else:
            highestPopulated = min(recsCntMissingValsList)
            maxIndex = [recsCntMissingValsList.index(highestPopulated)]
        maxPopulatedRecordDf = self.curDf.iloc[maxIndex]

        self.transposeAndSaveAsCsv(maxPopulatedRecordDf,expFlName)

        return maxPopulatedRecordDf

'''
#sample call to run the class
flName = "/Users/hansjoerg.stark/Library/Mobile Documents/com~apple~CloudDocs/Privat/Weiterbildung/Bethel-DataScience/Module5_DataWrangling/DS Data Wrangling and Visualization/Assignments/lesson03_1/superheroes.csv"
superheroes1 = pd.read_csv(flName)
myDf = myDataFrame(superheroes1)
print(myDf.getColSummary())
'''