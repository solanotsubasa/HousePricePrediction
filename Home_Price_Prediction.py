#import some necessary libraries
import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# %matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
#import warnings
#def ignore_warn(*args, **kwargs):
#    pass
#warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
import time

from scipy import stats
from scipy.stats import norm, skew, kurtosis, boxcox #for some statistics
from scipy.special import boxcox1p, inv_boxcox, inv_boxcox1p
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, LabelEncoder, RobustScaler, StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

# default competition
competition = 'SR' # StackedRegression

try:
    a = check_output(["ls", "../input"]).decode("utf8") # new Kaggle competition
except:
    a=''
finally:
    print('')
try:
    b = check_output(["ls", "-rlt", "../StackedRegression"]).decode("utf8")
except:
    b=''
finally:
    print('')    
#if (competition == 'SRP_2'): # Stacked Regressions Part 2
if (len(a) > 0): # new competition
    competition = 'SR'
    train = pd.read_csv('home-data-for-ml-course/train.csv')#,index_col='Id')
    test = pd.read_csv('home-data-for-ml-course/test.csv')#,index_col='Id')
elif (len(b)): # run locally
    competition = 'SR'
    train = pd.read_csv('home-data-for-ml-course/train.csv')
    test = pd.read_csv('home-data-for-ml-course/test.csv')
else: # old competition
    competition = 'SRP_2'
    train = pd.read_csv('home-data-for-ml-course/train.csv')
    test = pd.read_csv('home-data-for-ml-course/test.csv')

from subprocess import check_output

StartTime = datetime.datetime.now()

fig, ax = plt.subplots()
#ax.scatter(x = train['GrLivArea'], y = train['SalePrice']
ax.scatter(x = train['GrLivArea'], y = np.log(train['SalePrice']))
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

#m, b = np.polyfit(train['GrLivArea'], train['SalePrice'], 1)
m, b = np.polyfit(train['GrLivArea'], np.log(train['SalePrice']), 1)
#m = slope, b=intercept
plt.plot(train['GrLivArea'], m*train['GrLivArea'] + b)

plt.show()