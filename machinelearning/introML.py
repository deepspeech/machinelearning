import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL') #using wiki data set 
print(df.head())

# in machine learning you want meaningful "features" (aka data sets)
# think about relationships between features, labels for algorithms 
# below create the data frame and long list - reference data set

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

# margin of High and Low tells us about volitility
# define what the special relationships are subtract - divide and mult. percent.

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']*100.0

#Spyder3 in Anaconda let's you know about syntax errors!
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']*100.0
# once you have that data - define a new data set - think about features
#and or labels - your wanted outcome.
#
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume',]]

forecast_col = 'Adj. Close'
#In machine learning you don't want to leave out data (integers) so replace
#with a negative set of numbers

df.fillna(-9999, inplace=True)
#math.ceil rounds to whole number **** don't forget to import math
forecast_out = int(math.ceil(0.01*len(df))) 
#The labeled column will be 10 days into the future without specified timeframe
df['label'] = df[forecast_col].shift(-forecast_out)
 #now you can compare forcast price to adjusted
df.dropna(inplace=True)
print(df.head())

#Now you will be able to train, test and predict and run the algorihm!

