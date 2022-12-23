import sqlite3
import pandas as pd
import statistics
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns


con = sqlite3.connect(r'C:\Users\YASASWI\Downloads\DSCI6002_prj1_data.db')
df = pd.read_sql_query('SELECT * FROM prj1', con)


df.to_csv(r'C:\Users\YASASWI\Downloads\data_header.csv')  

df2 = pd.read_csv(r'C:\Users\YASASWI\Downloads\data_header.csv', header=None)

df2.drop(df2.columns[[0,1]], axis=1, inplace=True)

df2.columns = ['Class', 'Score']

# Eliminate the rows with blank score

df2.dropna(inplace = True)

df4 = df2.pivot(index=None, columns='Class', values = 'Score')

df4.boxplot(column = ["Bachelor's","Graduate","HS","Jr Coll","Less than HS"])

model = ols('Score ~ C(Class)', data=df2).fit()
model.summary()

anova_table = sm.stats.anova_lm(model, typ = 2)
print(anova_table)

esq_sm = anova_table['sum_sq'][0]/(anova_table['sum_sq'][0] + anova_table['sum_sq'][1])

anova_table['Etasq'] = [esq_sm, 'NaN']

print(anova_table)

pair_t = model.t_test_pairwise('C(Class)')
print(pair_t.result_frame)

mc = sm.stats.multicomp.MultiComparison(df2['Score'], df2['Class'])
mc_results = mc.tukeyhsd()
print(mc_results)


res = model.resid
fig = sm.qqplot(res, line='s')

sns.distplot(res,bins='auto', hist = True)




