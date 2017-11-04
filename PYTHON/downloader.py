#!/usr/bin/env python
import Quandl
import datetime as dt
import cPickle
import pandas
from numpy import cumsum, log, polyfit, sqrt, std, subtract, insert
import numpy as np

def hurst(ts):
	# Set the range of lag
	lags = range(2, 100)
	# Calculate the the variances
	tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
	# Estimate the Hurst Exponent
	poly = polyfit(log(lags), log(tau), 1)
	# Return the Hurst exponent from the polyfit output
	return poly[0]*2.0

instruments = ['EURUSD', 'JPYUSD', 'GBPUSD', 'AUDUSD', 'CHFUSD', 'CADUSD', 'HKDUSD', 'SEKUSD', 'NZDUSD', 'KRWUSD', 'SGDUSD', 'NOKUSD', 'MXNUSD', 'INRUSD', 'JPYEUR', 'GBPEUR', 'AUDEUR', 'CHFEUR', 'CADEUR', 'HKDEUR', 'SEKEUR', 'NZDEUR', 'KRWEUR', 'SGDEUR', 'NOKEUR', 'MXNEUR', 'INREUR', 'GBPJPY', 'AUDJPY', 'CHFJPY', 'CADJPY', 'SEKJPY', 'NZDJPY', 'NOKJPY', 'INRJPY', 'AUDGBP', 'CHFGBP', 'CADGBP', 'HKDGBP', 'SEKGBP', 'NZDGBP', 'KRWGBP', 'SGDGBP', 'NOKGBP', 'MXNGBP', 'INRGBP', 'CHFAUD', 'CADAUD', 'HKDAUD', 'SEKAUD', 'NZDAUD', 'KRWAUD', 'SGDAUD', 'NOKAUD', 'MXNAUD', 'INRAUD', 'CADCHF', 'SEKCHF', 'NZDCHF', 'NOKCHF', 'INRCHF', 'SEKCAD', 'NZDCAD', 'NOKCAD', 'INRCAD', 'INRHKD', 'NZDSEK', 'NOKSEK', 'INRSEK', 'NOKNZD', 'INRNZD', 'INRKRW', 'INRSGD', 'INRNOK', 'INRMXN']

start=dt.datetime(1990,1,1)
end=dt.datetime(2010,12,31)
data=pandas.DataFrame()

print 'Download started...'
number_of_assets = 0
for symbol in instruments:
  fx = Quandl.get("CURRFX/"+symbol, authtoken="DHRfTADW3mz8jee-sRcb", trim_start=start, trim_end=end).Rate
  print ". "+symbol+" downloaded"
  if hurst(fx)<0.5:
    data[symbol]=fx
    number_of_assets=number_of_assets+1
    print ".. "+symbol+" passed Hurst-test"
#  print tmp.shape
#  print tmp.head()
#print data.head()
print 'Download completed.'
pandas.DataFrame.to_csv(data, "fx_data_mean_rev.csv", header=True)
print 'Data modification...'
# exponential moving average
ema200 = pandas.stats.moments.ewma(data, span=200)
tmp=list()
for name in ema200.columns.tolist():
  tmp.append(name+'_ema200')
ema200.columns=tmp
print '... ema(200) calculated'

ema50 = pandas.stats.moments.ewma(data, span=50)
tmp=list()
for name in ema50.columns.tolist():
  tmp.append(name+'_ema50')
ema50.columns=tmp
print '... ema(50) calculated'

ema10 = pandas.stats.moments.ewma(data, span=10)
tmp=list()
for name in ema10.columns.tolist():
  tmp.append(name+'_ema10')
ema10.columns=tmp
print '... ema(10) calculated'

data = data.join(ema200).join(ema50).join(ema10)

# pandas.DataFrame.to_csv(data, "fx_data_mean_rev_ewma.csv", header=True)
data=data.dropna()
print '.. NaN values dropped'
data=data.as_matrix() # convert to nparray
target=data[:,0:number_of_assets-1]
target=target[1:,:] # every value but the first
target=insert(target, len(target)-1, 1)
print '.. target variable set-up'
print 'Data modification completed'
data = data[0:(data.shape[0]-data.shape[0]%50), ]
print data.shape
target = target[0:(data.shape[0]-data.shape[0]%10), ]
print target.shape
dataset=(data, target)
full_dataset=(dataset, dataset, dataset)
print 'Dump output...'
filename='full_data.save'
out_file=file(filename, 'wb')
cPickle.dump(full_dataset, out_file)
out_file.close()
print 'Dump completed to '+filename
