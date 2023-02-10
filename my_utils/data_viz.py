import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.mosaicplot import mosaic
import pandas as pd
import numpy as np

def mosaic_plot(df,X,y, ax=None):
    default_colors =plt.rcParams['axes.prop_cycle'].by_key()['color']
    cross = pd.crosstab(df[X],df[y])
    couples = cross.unstack().index
    props = lambda x: {'facecolor': default_colors[int(x[0])],'edgecolor':'w'}
    labelizer = lambda k: {(str(cpl[0]),str(cpl[1])) : f'{cpl[0]}-{cpl[1]}\n{round(cross.loc[cpl[1],cpl[0]]/cross.loc[:,cpl[0]].sum()*100,2)}%'  for cpl in couples}[k]
    mosaic(df, [y, X],properties=props,labelizer = labelizer, ax=ax)
    
def turbo_plot(df, X, y,classification):
    fig = plt.figure(constrained_layout=True,figsize=(15,round(10/3*df.shape[1])))
    subfigs = fig.subfigures(X.shape[1], 1,squeeze=False,hspace=20)

    for outerind, subfig in enumerate(subfigs.flat):
        #plotting numerical features
        if X[X.columns[outerind]].dtypes not in ['object','categorical','string'] and round(X[X.columns[outerind]].nunique()/df.shape[0]*100,2)>9:
            subfig.suptitle(f'Subfig {X.columns[outerind]}')
            axs = subfig.subplots(1, 4)
            sns.histplot(data = X, x = X.columns[outerind], kde=True, ax = axs[0])
            sns.boxplot(data = X, x = X.columns[outerind], ax = axs[1])
            qqplot(X[X.columns[outerind]],line='s',ax=axs[2])
            if classification: 
                sns.stripplot(data = X, x = y, y=X.columns[outerind], hue=y, ax = axs[3])
            else: 
                sns.scatterplot(data = X, x = X.columns[outerind], y=y, ax = axs[3])

        #plotting categorical features
        else:
            subfig.suptitle(f'Subfig {X.columns[outerind]}')
            axs = subfig.subplots(1, 4)
            sns.countplot(data = X, x = X.columns[outerind], ax = axs[0],order=X[X.columns[outerind]].value_counts().sort_values(ascending=False).index)
            sns.countplot(data = X, x = X.columns[outerind], hue=y, ax = axs[1],order=X[X.columns[outerind]].value_counts().sort_values(ascending=False).index)
            mosaic_plot(df,X.columns[outerind],df.survived.name,ax=axs[2])
            if classification: 
                sns.stripplot(data = X, x = y, y=X.columns[outerind], hue=y, ax = axs[3])
            else:
                sns.scatterplot(data = X, x = X.columns[outerind], y=y, ax = axs[3])
    return plt.show()

def quick_check(df, target:str, classification=True, to_drop=None):
    if target not in df.columns:
        raise ValueError('target not in df.columns')
    if not isinstance(target,str):
        raise TypeError('target must str')
    if to_drop:
        if all(x in df.columns for x in to_drop):
            raise ValueError('all elements in to_drop are not in df.columns')

        if not isintance(to_drop,list) and isintance(to_drop,str):
            to_drop=[to_drop]
        else:
            raise TypeError('to_drop type must be list of string')
    
    
    #Checking nan
    check= round(df.isna().sum()/df.shape[0]*100,2).sort_values(ascending=False)
    filtered = check[check>0]
    print(f'You have : {len(filtered)} features over {len(check)} ({round(check[check>0].shape[0]/check.shape[0],2)}% of whole df) that include np.nan')    

    #Features with nan
    print(f'\nHave a look at these features (% of nan): {", ".join([f"{i}: {str(v)}%" for i,v in filtered.items()])}')

    #Features to drop
    super_drop = check[check>15]
    print(f'\nYou might want to drop these features: {", ".join(super_drop.index)}')
    imputation = df[check[(check>0) & (check<15)].index].dtypes
    
    print('\n')
    print(df.info())

    print('\n')
    print("Let's have a look at all the features")
    X=df.drop(columns=(target if not to_drop else [target_name]+to_drop))
    y=df[target]
    turbo_plot(df, X,y,classification)
