#TODO: Add scaler recognition with pd.skew() values 
#TODO: Add threshold of categorical features
#TODO: Update title with subtitle

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.mosaicplot import mosaic
import pandas as pd

def mosaic_plot(df : pd.DataFrame , X : str , y : str , ax):
    '''Function to plot a mosaic plot using statsmodels.
    Base color is the one for matplotlib'''
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    #Setting cross tables to create couples between target and feature unique values
    cross = pd.crosstab(df[X],df[y])
    couples = cross.unstack().index
    
    #Set colors and labels for the mosaic_plot
    props = lambda x: {'facecolor': default_colors[int(x[0])],'edgecolor':'w'}
    labelizer = lambda k: {(str(cpl[0]),str(cpl[1])) : f'{cpl[0]}-{cpl[1]}\n{round(cross.loc[cpl[1],cpl[0]]/cross.loc[:,cpl[0]].sum()*100,2)}%'  for cpl in couples}[k]
    
    #Mosaic plot
    mosaic(df, [y, X],properties=props,labelizer = labelizer, ax=ax)
    
def is_cat(data : pd.Series, percentage_cat : int | float = 0.09):
    if not isinstance(data,pd.Series):
        raise TypeError(f'data should be a pd.Series')
    
    if not (isinstance(percentage_cat, int) or isinstance(percentage_cat, float)):
        raise TypeError(f'percentage_cat should be int or float')
    
    if data.dtypes not in ['object','categorical','string','bool'] \
        and data.nunique()/data.shape[0]>percentage_cat:
        return False
    return True
    
def turbo_plot(df : pd.DataFrame, target : str, classification : bool, num_only : bool = False):
    
    #Define X and y
    X= df.drop(columns=target)
    y= df[target]
    
    fig = plt.figure(constrained_layout=True,figsize=(15,round(10/3*df.shape[1])))
    subfigs = fig.subfigures(X.shape[1], 1,squeeze=False,hspace=20)
    for outerind, subfig in enumerate(subfigs.flat):
        #plotting numerical features
        
        if is_cat(X[X.columns[outerind]]):
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
            if not num_only:
                subfig.suptitle(f'Subfig {X.columns[outerind]}')
                axs = subfig.subplots(1, 4)
                sns.countplot(data = X, x = X.columns[outerind], ax = axs[0],order=X[X.columns[outerind]].value_counts().sort_values(ascending=False).index)
                if classification: 
                    sns.countplot(data = X, x = X.columns[outerind], hue=y, ax = axs[1],order=X[X.columns[outerind]].value_counts().sort_values(ascending=False).index)
                    mosaic_plot(df,X.columns[outerind],y.name,ax=axs[2])
                    sns.stripplot(data = X, x = y, y=X.columns[outerind], hue=y, ax = axs[3])
                else:
                    qqplot(X[X.columns[outerind]],line='s',ax=axs[1])
                    sns.scatterplot(data = X, x = X.columns[outerind], y=y, ax = axs[3])
    return plt.show()

def quick_check(df : pd.DataFrame, target : str, classification : bool = None, plot_overview : bool = True, to_drop : list = None, num_only : bool = False):


    #Tests all arguments
    if target not in df.columns:
        raise ValueError('target not in df.columns')
    if not isinstance(target,str):
        raise TypeError('target must str')
    if to_drop:
        if all(x in df.columns for x in to_drop):
            raise ValueError('all elements in to_drop are not in df.columns')

        if isinstance(to_drop,str):
            to_drop=[to_drop]
        if not isinstance(to_drop, list):
            raise TypeError('to_drop type must be list of string')
    
    #Target review
    check_nan =f'{round(df[target].isna().sum()*100,2)}% of'
    
    if check_nan == 0:
        check_nan ='no'
    
    if classification is None:
        data_type = 'categorical' if is_cat(df[target]) else 'continuous'
        classification = True if data_type=='categorical' else False
    else :
        data_type = 'categorical' if classification else 'continuous'
    
    
    #Target_overview
    prompt = '\u0332'.join('Target overview:')
    print(f'{prompt}\n')
    print(f'''  The target is {target}.
  It is composed of {check_nan} np.nan.
  Its dtypes is {df[target].dtype.name}.
  Without specifying an estimator_job, the target will be considered as {data_type}.\n
''')
    
    #Columns overview
    prompt = '\u0332'.join('Features overview:')
    cat_feats = [cat_feat for cat_feat in df.columns if is_cat(df[cat_feat])]
    num_feats = [num_feat for num_feat in df.columns if not is_cat(df[num_feat])]
    
    print(f'{prompt}\n')
    print(f'  There is {df.shape[1]} columns which are:\n')
    if cat_feats != []:
        print(f'    - {round(len(num_feats)/df.shape[1]*100,2)}% of them seem to be numerical;\n')
    if num_feats != []:
        print(f'    - {round(len(cat_feats)/df.shape[1]*100,2)}% of them seem to be categorical;\n')
    
    #Dtypes overview
    prompt = '\u0332'.join('Dtypes overview:')
    print(f'{prompt}\n')
    print(f'  Dataset has {len(df.dtypes.unique())} different dtypes which are:\n')
    for d_type in df.dtypes.unique():
        columns_in_dtypes = df.dtypes[df.dtypes==d_type].index
        num_d_types = round(len(columns_in_dtypes)/df.shape[1]*100,2)
        print(f'    - {d_type} corresponds to {num_d_types}% of columns and includes columns: {", ".join(columns_in_dtypes)}\n')
        
    
    #Nan overview    
    prompt = '\u0332'.join('Nan overview:')
    print(f'{prompt}\n')
    check_nan = round(df.isna().sum()/df.shape[0]*100,2).sort_values(ascending=False)
    filtered = check_nan[check_nan>0]
    print(f'There are {len(filtered)} features over {len(check_nan)} ({round(check_nan[check_nan>0].shape[0]/check_nan.shape[0],2)}% of whole df) that include np.nan')    
    print(f'\nHave a look at these features (% of nan): {", ".join([f"{i}: {str(v)}%" for i,v in filtered.items()])}')

    #Features to drop
    prompt = '\u0332'.join('Features to_drop')
    print(f'{prompt}\n')
    super_drop = check_nan[check_nan>15]
    if len(super_drop)!=0:
        print(f'\nYou might want to drop these features: {", ".join(super_drop.index)}')
    print(f'\nNo features to drop found.')
    
    print('\n')
    print(df.info())

    print('\n')
    print("Let's have a look at all the features")
    if to_drop : df.drop(columns=to_drop,inplace=True)
    if plot_overview: turbo_plot(df, target, classification, num_only)
