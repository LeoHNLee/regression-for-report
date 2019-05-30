from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import OLSInfluence

def min_max_to_real(x):
        mn = min(x)-1
        mx = max(x)+1
        x = (x-mn)/(mx-mn)
        x = x/(1-x)
        return np.log(x)

def mode(x):
        return stats.mode(x)[0][0]

def mode_val(x):
        return stats.mode(x)[1][0]

class Reg(object):
    def __init__(self, df):
        self.df = df
        self.n = df.shape[0]
        self.p = df.shape[1]
        self.fontsize = 30

    def describe(self):
        x = self.df
        idx = x.columns
        ret = pd.DataFrame(index = idx)
        foos = {'mean':np.mean, 'median':np.median, 'mode':mode, 'std':np.std, 'skew':stats.skew,
                'kurt':stats.kurtosis, 'min':min, '25%':np.quantile, '75%':np.quantile, 'max':max}
        for new_col, foo in foos.items():
            items = []
            for col in idx:
                if foo != mode :
                    if x[col].dtype == 'O':
                        items.append(np.nan)
                        continue
                if new_col == '25%' :
                    items.append(foo(x[col], 0.25))
                elif new_col == '75%':
                    items.append(foo(x[col], 0.75))
                else :
                    items.append(foo(x[col]))
            ret[new_col] = items
        self.description = ret
        print(x.shape)

    def show_before_reg(self, is_corr=None, is_single= None, is_pair = None, print_corr = None, cols = None):
        if cols is None : cols = self.df.columns
        if is_pair is not None :
            sns.pairplot(self.df[cols])
            plt.show()

        if is_corr is not None :
            fig = plt.figure(figsize=(17,8))
            corr = self.df[cols].corr()
            sns.heatmap(corr)
            plt.show()

        if print_corr is not None :
            print(corr)

        if is_single is not None :
            for x in cols :
                fig = plt.figure(figsize=(17,8))
                sns.distplot(self.df[x])
                plt.show()

    def regression(self, y_col, x_cols, reg_foo = sm.OLS, is_constant=None):
        model_x = self.df[x_cols].values
        if is_constant is not None :
            model_x = sm.add_constant(model_x)
        model_y = self.df[y_col].values
        model = reg_foo(model_y, model_x).fit()
        temp = deepcopy(self)
        temp.model = model
        temp.model_property = (y_col, x_cols, is_constant, reg_foo)
        temp.coef = model._results.params
        temp.resid = model.resid
        temp.std_resid = model.resid_pearson
        temp.pred = model.predict(model_x)
        temp.p = len(x_cols)
        return temp

    def show_after_reg(self, leverage):
        self.show_qq_plot()
        self.show_pred_resid_plot()
        self.show_index_resid_plot()
        for x in self.model_property[1]:
            self.show_reg_line(x, leverage)
            self.show_resid_plot(x)

    def show_reg_line(self, x, leverage, scatter=True):
        fontsize = self.fontsize
        y = self.model_property[0]

        fig = plt.figure(figsize=(17,8))

        sns.regplot(x, y, data = self.df, scatter=scatter)

        plt.xlabel('{}'.format(x), fontdict={'fontsize': fontsize,})
        plt.ylabel('{}'.format(y), fontdict={'fontsize': fontsize,})
        plt.title('Regression Line : {}'.format(x), fontdict={'fontsize': fontsize,})
        plt.show()

    def show_qq_plot(self):
        fontsize = self.fontsize
        resid = self.std_resid
        fig = plt.figure(figsize=(17,8))
        stats.probplot(resid, plot=plt)
        plt.xlabel('Standard Residual', fontdict={'fontsize': fontsize,})
        plt.title('Q-Q Plot', fontdict={'fontsize': fontsize,})
        plt.show()

    def show_pred_resid_plot(self):
        fontsize = self.fontsize
        x, y = self.pred, self.std_resid
        fig = plt.figure(figsize=(17,8))
        sns.scatterplot(x,y)
        sns.lineplot(x, 0, color='red')

        q1 = np.quantile(y, .25)
        q3 = np.quantile(y, .75)
        iqr = q3-q1
        q1 -= iqr*1.5
        q3 += iqr*1.5
        sns.lineplot(x, q1, color='red')
        sns.lineplot(x, q3, color='red')

        plt.xlabel('Y hat', fontdict={'fontsize': fontsize,})
        plt.ylabel('Standard Residual', fontdict={'fontsize': fontsize,})
        plt.title('Y_hat ~ Residual Plot', fontdict={'fontsize': fontsize,})
        plt.show()

    def show_index_resid_plot(self):
        fontsize = self.fontsize
        x, y = self.df.index, self.std_resid

        fig = plt.figure(figsize=(17,8))
        sns.scatterplot(x, y)
        sns.lineplot(x, 0, color='red')
        plt.xlabel('Index', fontdict={'fontsize': fontsize,})
        plt.ylabel('Standard Residual', fontdict={'fontsize': fontsize,})
        plt.title('Index ~ Residual Plot', fontdict={'fontsize': fontsize,})
        plt.show()

    def show_resid_plot(self, x):
        fontsize = self.fontsize
        fig = plt.figure(figsize=(17,8))
        plt.xlabel('{}'.format(x), fontdict={'fontsize': fontsize,})
        plt.ylabel('Standard Residual', fontdict={'fontsize': fontsize,})
        plt.title('{} ~ Residual Plot'.format(x), fontdict={'fontsize': fontsize,})
        x, y = self.df[x], self.std_resid

        sns.scatterplot(x, y)
        sns.lineplot(x, 0, color='red')

        plt.show()

    def find_outlier(self):
        y = self.std_resid
        q1 = np.quantile(y, .25)
        q3 = np.quantile(y, .75)
        iqr = q3-q1
        q1 -= iqr*1.5
        q3 += iqr*1.5

        outlier = (y<q1)|(y>q3)

        influence = self.model.get_influence()
        outlier += OLSInfluence.cooks_distance(influence)[0]>1
        diffits = 2*np.sqrt((self.p+1)/(self.n-self.p-1))
        outlier += OLSInfluence.dffits(influence)[0]<-diffits
        outlier += OLSInfluence.dffits(influence)[0]>diffits

        self.non_outlier = (outlier<1)

    def remove_outlier(self):
        ret = deepcopy(self)
        ret.df = self.df[self.non_outlier]
        y,x,c,r = self.model_property
        return ret.regression(y_col = y, x_cols = x, is_constant = c, reg_foo=r)

    def show_eda(self, x, y=None):
        fig = plt.figure(figsize=(17,8))
        x = self.df[x]
        if y is None :
            sns.distplot(x)
        else :
            y = self.df[y]
            sns.scatter(x,y)
            plt.ylabel(y, fontdict={'fontsize':fontsize})
        plt.xlabel(x, fontdict={'fontsize' : fontsize})
        plt.show()
