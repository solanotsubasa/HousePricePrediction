import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import numpy as np

DataPath = 'home-data-for-ml-course/train.csv'
TrainData = pd.read_csv('home-data-for-ml-course/train.csv')#,index_col='Id')
TestData = pd.read_csv('home-data-for-ml-course/test.csv')#,index_col='Id')

print(TrainData.head(10))

fig, ax = plt.subplots()
ax.scatter(x = TrainData['GrLivArea'], y = np.log(TrainData['SalePrice']))
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

m, b = np.polyfit(TrainData['GrLivArea'], np.log(TrainData['SalePrice']), 1)
#m = slope, b=intercept
plt.plot(TrainData['GrLivArea'], m*TrainData['GrLivArea'] + b)

plt.show()

TrainData.shape[1]
#a = int(np.sqrt(train.shape[1]))
a = 4
b = int(TrainData.shape[1]/4)
rows = int(TrainData.shape[1]/a)
cols = int(TrainData.shape[1]/b)
i = 0
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 60))
for row in ax:
    for col in row:
        try:
            col.scatter(x = TrainData[TrainData.columns[i]], y = np.log(TrainData['SalePrice']))
            col.title.set_text(TrainData.columns[i])
        except:
            temp=1
        #except Exception as e:
        #    print(e.message, e.args)
        finally:
            temp=1
        i = i + 1
        
plt.show()