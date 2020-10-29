from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('population.csv')
df.head()

x = df["year"]
y = df["count"]

plt.plot(x, y, 'o') #x, y insert 'o' 
plt.show() # show

line_fitter = LinearRegression()
line_fitter.fit(x.values.reshape(-1,1), y)
# values reshape를 한 이유는 다중회귀분석 때문에 배열로 넣어야해서 넣음

res = line_fitter.predict([[2021]])
print(res)
res2 = line_fitter.coef_  #기울기
res3 = line_fitter.intercept_  #질펀
#print(res2)
#print(res3)

# 선 그리기 
plt.plot(x, y, 'o')
plt.plot(x, line_fitter.predict(x.values.reshape(-1,1)))
plt.show()