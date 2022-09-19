import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Classification models that will be used
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# Models evaluation
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc


###############################################################################
## INITIALISATION                                                            ##
###############################################################################

#path_data = r'https://github.com/fadilnjimoun/PyBet---Bet/tree/main/Streamlit/Data/'
path_data = r'Data/'
#path_data = r'https://github.com/fadilnjimoun/PyBet---Bet/tree/main/Streamlit/Images/'
path_imag = r'Images/'

###############################################################################
## MENU                                                                      ##
###############################################################################

title = 'PyBet - Can we beat the bookmakers ?'
menu_lst = ['1. Context and objectives',
            '2. Data Exploration and Cleaning',
            '3. Data Visualisation',
            '4. Machine learning',
            '5. Bet : Win or Loose ?',
            '6. Model Interpretability',
            '7. Conclusion']

st.sidebar.header(title)
st.sidebar.write('Make a choice :')
menu_sel = st.sidebar.selectbox('', menu_lst)

# Les auteurs
st.sidebar.subheader('Autor')
st.sidebar.write("""[Fadil NJIMOUN](https://www.linkedin.com/in/fadil-njimoun-a232ba101/)""")

###############################################################################
## PART 1 : INTRODUCTION                                                     ##
###############################################################################

if menu_sel == menu_lst[0]:

    st.header(title)
    st.subheader(menu_sel, anchor='1. Context and objectives')
    
    st.markdown("""Sports betting is a game of money on the prediction of an event during a sports meeting. There are many types of bets, the simplest of which consist of betting on the victory of a team or an athlete. Each bet has a rating and it is this that allows you to know in advance, depending on the amount wagered, the amount won if the event occurs.
If we bet €10 on team A whose odds are 1.2 for example:
- If A wins, the amount recovered (gain) is €12, i.e. a capital gain of €2 (**+20%**)
- If A does not win, the amount wagered is lost, i.e. €10 (**-100%**).
                """)
    st.markdown("""Bets are offered by bookmakers which are organizations authorized to offer players bets. In French and with the rise of gambling on the internet, the term bookmaker generally refers to the online betting site.
                """)
    st.markdown("""The objective of this project is to try to **beat bookmakers' algorithms on estimating the probability of a team winning a tennis match**.
                    In a first step we will study and apply methods to clean the dataset. Once the dataset is clean, a second step is to extract from the match history the characteristics that seem to be relevant to estimate the performance of a player (ranking, type of tournament, etc.). Finally, from these features, we will estimate the probability that a player A beats a player B.
                 """)
    st.image(path_imag + 'bookies.png')
###############################################################################
## PART 2 : DATA EXPLORATION                                                 ##
###############################################################################

df = pd.read_csv(path_data + 'atp_data.csv')

def remove_space(nom):
    if type(nom) == str:
        resultat = nom
        if nom[0] == ' ':
            resultat = nom[1:]
        if resultat[-1] == ' ':
            resultat = resultat[:-1]
        return resultat
    return nom

def add_mark(nom):
    if type(nom) == str:
        if nom[-1] != '.' and nom != np.nan:
            return nom + '.'
        return nom
    return nom

def add_dash(nom):
    if type(nom) == str:
        compteur = 0
        indice = 0
        for i,j in enumerate(reversed(nom)):
            if j == ' ':
                compteur += 1
                indice = i
        if compteur == 2:
            return nom[:indice-1] + '-' + nom[indice:]
    return nom

df[['Winner', 'Loser']] = df.apply({'Winner' : remove_space, 'Loser' : remove_space})
df[['Winner', 'Loser']] = df.apply({'Winner' : add_dash, 'Loser' : add_dash})
df[['Winner', 'Loser']] = df.apply({'Winner' : add_mark, 'Loser' : add_mark})

df1 = df.loc[(df.Comment == 'Completed') & (df.Wsets.notna()) & (df.Lsets.notna())]

for i in ['PSW', 'PSL', 'B365W', 'B365L']:
    df1[i].fillna(1, axis=0, inplace=True)

df1 = df1.astype({'Wsets' : 'int', 'Lsets' : 'int', 'Date' : 'datetime64'})


if menu_sel == menu_lst[1]:
    #######################################
    ## Data Exploration and Cleaning     ##
    #######################################
    
    st.header(title)
    st.subheader(menu_sel, anchor='2. Data Exploration and Cleaning')
    
    st.write("""The dataset comes from the website ([kaggle](https://www.kaggle.com/edouardthomas/atp-matches-dataset)) . 
                This is a csv file csv atp_data.csv (du site tennis.co.uk data) listing all tennis matches between the years 2000 and 2018.
                The dataset contains 23 variables and 44 708 matches (rows).""")
    
       
    #######################################
    ## Dataset                           ##
    #######################################
    
    st.subheader('Dataset overview')
    
    if st.checkbox('Show the dataset'):       
        df
    
    st.markdown("- <font color=" + 'blue' + ">*ATP*</font> : (*int*) ATP tournament number",              unsafe_allow_html=True)
    st.markdown("- <font color=" + 'blue' + ">*Location*</font> : (*object*) Tournament city",              unsafe_allow_html=True)
    st.markdown("- <font color=" + 'blue' + ">*Date*</font> : (*object*) Match day",              unsafe_allow_html=True)
    st.markdown("- <font color=" + 'blue' + ">*Series*</font> : (*object*) Tournament type",              unsafe_allow_html=True)

    #######################################
    ## DUPLICATES PRESENCE               ##
    #######################################
    st.subheader('Duplicates presence')
    st.code("""df.duplicated().sum()""", language='python')
    
    st.write("""There is no duplicated values within our dataset.""")
    
    #######################################
    ## MISSING VALUES PRESENCE           ##
    #######################################
    st.subheader('Missing values presence')
    
    fig, ax = plt.subplots()
    df.isnull().sum().plot(kind='bar')
    ax.set_xlabel('Variables')
    ax.set_ylabel('Total')
    ax.set_title('NA values per column')
    st.pyplot(fig)

    st.write("""Unfinished matches or whose final and even partial score is not known are of limited interest in the context of our analysis.
                We will therefore now focus on the matches for which we know the outcome (“Comment” = 'Completed').""")
    st.write("""Also, as seen in the previous figure, we are missing some odds for bookmaker B365 and not for PS, and vice versa.
                We choose to replace the missing values of a bookmaker with the values of the other bookmaker.""")
    
    #######################################
    ## CLEANED DATASET             ##
    #######################################
    st.subheader('Cleaned dataset')
    st.write("""In order to get a clean dataset, we therefore apply this code.""")
    
    st.code("""
    df1 = df.loc[(df.Comment == 'Completed') & (df.Wsets.notna()) & (df.Lsets.notna())]
    df1.loc[(df1.B365W.isna()==True) & (df1.PSW > 0), 'B365W'] = df1.loc[(df1.B365W.isna()==True) & (df1.PSW > 0), 'PSW']
    df1.loc[(df1.B365L.isna()==True) & (df1.PSW > 0), 'B365L'] = df1.loc[(df1.B365L.isna()==True) & (df1.PSW > 0), 'PSL']
    df1.loc[(df1.PSL.isna()==True) & (df1.B365W > 0), 'PSL'] = df1.loc[(df1.PSL.isna()==True) & (df1.B365W > 0), 'B365L']
    df1.loc[(df1.PSW.isna()==True) & (df1.B365W > 0), 'PSW'] = df1.loc[(df1.PSW.isna()==True) & (df1.B365W > 0), 'B365W']
    
    # Lets replace remaining missing odds values by 1
    
    for i in ['PSW', 'PSL', 'B365W', 'B365L']:
        df1[i].fillna(1, axis=0, inplace=True)
    """, language='python')
 
    st.write(df1.describe(include=[np.number]))
    
###############################################################################
## PART 3 : DATA VIZ                                                         ##
###############################################################################
win_loss = pd.DataFrame(df1.Winner.value_counts()).join(pd.DataFrame(df1.Loser.value_counts()))
win_loss['w_per_match'] = round(win_loss.Winner / (win_loss.Winner + win_loss.Loser),3)
win_loss['matches'] = win_loss.Winner + win_loss.Loser
win_loss.sort_values('w_per_match', ascending=False, inplace=True)

labels = ['Very irregular player', 'Irregular player', 'Regular player', 'Very regular player']
win_loss['cat_match'] = pd.cut(x=win_loss.matches, bins=4, labels=labels)

winner_tampon = pd.DataFrame(df1[['Winner', 'Series']].value_counts()).reset_index()
winner_tampon.rename({'Winner' : 'Player', 0: 'Winner'}, inplace=True, axis=1)
loser_tampon = pd.DataFrame(df1[['Loser', 'Series']].value_counts()).reset_index()
loser_tampon.rename({'Loser' : 'Player', 0: 'Loser'}, inplace=True, axis=1)
win_loss_ser = winner_tampon.merge(right=loser_tampon, on=['Player','Series'], how='right'); win_loss_ser = win_loss_ser.fillna(0)
win_loss_ser['matches'] = win_loss_ser.Winner + win_loss_ser.Loser
win_loss_ser['w_per_match'] = round(win_loss_ser.Winner / win_loss_ser.matches,3)
win_loss_ser.sort_values(by = ['w_per_match', 'Series'], ascending=False, inplace=True)
win_loss_ser.set_index('Player', inplace=True)
labels = ['Very irregular player', 'Irregular player', 'Regular player', 'Very regular player']
win_loss_ser['cat_match'] = pd.cut(x=win_loss_ser.matches, bins=4, labels=labels)


    #######################################
    ## TOP WINNERS                       ##
    #######################################
    
if menu_sel == menu_lst[2]:
    
    st.header(title)
    st.subheader(menu_sel, anchor='3. Data Visualisation')
    st.subheader("Players who have won the most matches")
    sns.set_theme()

    plt.figure(figsize=(14,7))

    plt.subplot(121)
    data = df1.Winner.value_counts()[:20]
    x, y = data.values, data.index
    sns.barplot(x=x, y=y, data=pd.DataFrame(data), order = y, orient='h')
    plt.xlim((300,None)); plt.title('Top 20 won matches')

    # Les joueurs (Top 20) ayant perdu le plus de matchs
    plt.subplot(122)
    data = df1.Loser.value_counts()[:20]
    x, y = data.values, data.index
    sns.barplot(x=x, y=y, data=pd.DataFrame(data), order = y, orient='h')
    plt.xlim((200,None)); plt.title('Top 20 lost matches')
    
    st.pyplot()

    st.write("Federer, Nadal and Djokovic form the top 3.")
    
    #######################################
    ## PLAYERS PERFORMANCE               ##
    #######################################

    st.subheader("Global performances")
    st.write("Players not having the same number of matches played, we introduce the variable **w_per_match**. The later corresponds to the ratio between the number of matches won over the number of matches played.")

    st.write(win_loss)

    #######################################
    ## PERFORMANCE BY...                 ##
    #######################################

    st.subheader("Performance by...")
    
    x_list = ['...by number of matches played', '...by tournament']
    abscissa = st.radio("Choose an abscissa :", x_list)

    if abscissa == x_list[0]:
        plt.figure(figsize=(13,7))
        sns.boxplot(x='cat_match', y='w_per_match', data=win_loss)
        plt.title('Performances distribution')
        st.pyplot()
        st.write("The more matches we play, the better the performance. The more you play, the more chances you have of winning.")

    if abscissa == x_list[1]:
        plt.figure(figsize=(16,7))
        sns.boxplot(data=win_loss_ser, x='Series', y='w_per_match', hue='cat_match')
        plt.title('Distribution des performances'); st.pyplot()
        st.write('Performance and tournament type are dependent on each other.')

#######################################
## PART 4 : MACHINE LEARNING         ##
#######################################

df2 = df1.copy()

df2['player1'] = df2.Winner.copy()
df2['player2'] = df2.Loser.copy()
df2.drop('Loser', axis=1, inplace=True)
df2.rename({'Winner': 'player1_wins', 'WRank': 'rank1', 'LRank': 'rank2', 'Wsets': 'sets1', 'Lsets': 'sets2',
            'PSW': 'ps1', 'PSL': 'ps2', 'B365W': 'b365_1', 'B365L': 'b365_2', 'elo_winner': 'elo1', 
            'elo_loser': 'elo2', 'Date':'date'}, axis=1, inplace=True)

df3 = df2.copy()

df2.loc[:,'player1_wins'], df3.loc[:,'player1_wins'] = -1, 1

df2.player1, df2.player2 = df3.player2, df3.player1
df2.rank1, df2.rank2 = df3.rank2, df3.rank1
df2.sets1, df2.sets2 = df3.sets2, df3.sets1
df2.ps1, df2.ps2 = df3.ps2, df3.ps1
df2.b365_1, df2.b365_2 = df3.b365_2, df3.b365_1
df2.elo1, df2.elo2 = df3.elo2, df3.elo1

df2 = df2.append(df3, ignore_index=True)
del df3

col_to_keep = ['Tournament', 'Round', 'player1_wins', 'rank1', 'rank2', 'ps1', 'ps2', 'b365_1', 'b365_2',
               'elo1', 'elo2', 'player1', 'player2'] 
df_final = df2[col_to_keep]

data = df_final.drop(['player1_wins'], axis=1)
target = df_final.player1_wins

X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=.7)
X_train_num, X_test_num = X_train.select_dtypes(include=np.number), X_test.select_dtypes(include=np.number)
X_train_cat1, X_test_cat1 = X_train[['Round']], X_test[['Round']]
X_train_cat2, X_test_cat2 = X_train[['Tournament', 'player1', 'player2']], X_test[['Tournament', 'player1', 'player2']]

# We standardize numerical values :
scaler = StandardScaler()
X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), index=X_train.index, columns = X_train_num.columns)
X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_num), index=X_test.index, columns = X_test_num.columns)

# We encode our categorical values :
oneh = OneHotEncoder(handle_unknown = 'ignore')

X_train_cat_encod1 = pd.DataFrame(oneh.fit_transform(X_train_cat1).toarray(), index=X_train.index, columns = oneh.get_feature_names())
X_test_cat_encod1 = pd.DataFrame(oneh.transform(X_test_cat1).toarray(), index=X_test.index, columns = oneh.get_feature_names())

# We encode our categorical values :
le1 = LabelEncoder()
le2 = LabelEncoder()

le2.fit(data.loc[:,'player1'].append(data.loc[:,'player2'], ignore_index=True).drop_duplicates())

X_train_cat_encod2 = pd.DataFrame({X_train_cat2.columns[0] : le1.fit_transform(X_train_cat2.iloc[:,0]),
                                   X_train_cat2.columns[1] : le2.transform(X_train_cat2.iloc[:,1]),
                                   X_train_cat2.columns[2] : le2.transform(X_train_cat2.iloc[:,2])}, index=X_train.index)
X_test_cat_encod2 = pd.DataFrame({X_test_cat2.columns[0] : le1.fit_transform(X_test_cat2.iloc[:,0]),
                                   X_test_cat2.columns[1] : le2.transform(X_test_cat2.iloc[:,1]),
                                   X_test_cat2.columns[2] : le2.transform(X_test_cat2.iloc[:,2])}, index=X_test.index)

X_train = pd.concat([X_train_cat_encod1, X_train_cat_encod2, X_train_num_scaled], axis=1)
X_test = pd.concat([X_test_cat_encod1, X_test_cat_encod2, X_test_num_scaled], axis=1)

if menu_sel == menu_lst[3]:
    
    st.header(title)
    st.subheader(menu_sel, anchor='Data preprocessing / Features engineering')
    st.write('**Changing the structure of the dataset**')
    st.write("""In the original dataset presented above, the variable "Winner" indicates the winner of the match. In order to prepare our data, we choose to modify the structure of the initial table as follows:
- "Winner" and "Lower" variables are changed to "player1" and "player2"
- A new variable **“player1_wins”** is created. It is categorical and will be our target variable. It takes as values:\n
        o 1: if player 1 wins the match \n
        o -1: if player2 wins the match
- In order to balance the data set, it seems relevant to split it. The length of the dataframe is thus multiplied by 2. On one half of it, player1 is the winner, on the other half it is player2 who is the winner.""")

    if st.checkbox('Show the new dataset'):
        df2
    
    st.subheader('Features engineering')

    if st.checkbox('Features engineering code'):
        st.code("""col_to_keep = ['Tournament', 'Round', 'player1_wins', 'rank1', 'rank2', 'ps1', 'ps2', 'b365_1', 'b365_2',
               'elo1', 'elo2', 'player1', 'player2'] 
df_final = df2[col_to_keep]

data = df_final.drop(['player1_wins'], axis=1)
target = df_final.player1_wins

X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=.7)
X_train_num, X_test_num = X_train.select_dtypes(include=np.number), X_test.select_dtypes(include=np.number)
X_train_cat1, X_test_cat1 = X_train[['Round']], X_test[['Round']]
X_train_cat2, X_test_cat2 = X_train[['Tournament', 'player1', 'player2']], X_test[['Tournament', 'player1', 'player2']]

# We standardize numerical values :
scaler = StandardScaler()
X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), index=X_train.index, columns = X_train_num.columns)
X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_num), index=X_test.index, columns = X_test_num.columns)

# We encode our categorical values :
oneh = OneHotEncoder(handle_unknown = 'ignore')

X_train_cat_encod1 = pd.DataFrame(oneh.fit_transform(X_train_cat1).toarray(), index=X_train.index, columns = oneh.get_feature_names())
X_test_cat_encod1 = pd.DataFrame(oneh.transform(X_test_cat1).toarray(), index=X_test.index, columns = oneh.get_feature_names())

# We encode our categorical values :
le1 = LabelEncoder()
le2 = LabelEncoder()

le2.fit(data.loc[:,'player1'].append(data.loc[:,'player2'], ignore_index=True).drop_duplicates())

X_train_cat_encod2 = pd.DataFrame({X_train_cat2.columns[0] : le1.fit_transform(X_train_cat2.iloc[:,0]),
                                   X_train_cat2.columns[1] : le2.transform(X_train_cat2.iloc[:,1]),
                                   X_train_cat2.columns[2] : le2.transform(X_train_cat2.iloc[:,2])}, index=X_train.index)
X_test_cat_encod2 = pd.DataFrame({X_test_cat2.columns[0] : le1.fit_transform(X_test_cat2.iloc[:,0]),
                                   X_test_cat2.columns[1] : le2.transform(X_test_cat2.iloc[:,1]),
                                   X_test_cat2.columns[2] : le2.transform(X_test_cat2.iloc[:,2])}, index=X_test.index)

X_train = pd.concat([X_train_cat_encod1, X_train_cat_encod2, X_train_num_scaled], axis=1)
X_test = pd.concat([X_test_cat_encod1, X_test_cat_encod2, X_test_num_scaled], axis=1)""")
        
    st.subheader('Models training')

    lst_algos    = ['Decision Tree', 'Logistic Regression', 'KNN', 'Random Forest'] # List of models
    
    # User choose one ML algo
    sel_algo = st.selectbox("Sélection de l'algorithme de machine learning", lst_algos)

    #######################################
    ## DECISION TREE                     ##
    #######################################
    
    if sel_algo == 'Decision Tree':
        st.title('DECISION TREE ')

        if st.checkbox('Script'):
            st.code("""dt = DecisionTreeClassifier()
# Model training
dt.fit(X_train, y_train)
# Model prediction
y_pred_dt = dt.predict(X_test) 
# Classification report
print(classification_report(y_test, y_pred_dt))""")

        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test) 
        st.code(classification_report(y_test, y_pred_dt))   

    #######################################
    ## LOGISTIC REGRESSION                     ##
    #######################################
    
    if sel_algo == 'Logistic Regression':
        st.title('LOGISTIC REGRESSION')

        if st.checkbox('Script'):
            st.code("""lr = LogisticRegression(max_iter=1000)
# Model training
lr.fit(X_train, y_train)
# Model prediction
y_pred_lr = lr.predict(X_test) 
# Classification report
print(classification_report(y_test, y_pred_lr))""")

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test) 
        st.code(classification_report(y_test, y_pred_lr)) 

    st.subheader('Final Model')

    st.write("""We also did *hyperparameters tunning*.
    Although some models such as the knn obtain better accuracy following cross-validation, these do not exceed the **70%** obtained previously. Also, the score of our **logistic regression** model stagnates around 70%.""")

    st.write("So we won't get a better model with better accuracy than bookmakers. As a reminder, we saw above that bookmakers see the outcome of a match correctly 7 times out of 10. This is equivalent to the accuracy of our model.")
            

#######################################
## PART 5 : BET : WIN OR LOOSE ?     ##
#######################################

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test) 

def bet_allocation(base_cagnotte=0, mise_min=100, mise_max=1000, sureté=0.8):
    y_pred_proba = lr.predict_proba(X_test) 
    
    cagnotte = base_cagnotte
    
    mise_totale = 0
    
    y_pred_proba2 = [probas for probas in y_pred_proba]
    
    for i, probas in zip(y_test.index, y_pred_proba2):
    #for i, probas in enumerate(y_pred_proba):    
        cotes_1 = df_final[['ps1', 'b365_1']].loc[i]
        cotes_2 = df_final[['ps2', 'b365_2']].loc[i]
        
        probas_test = probas[1]
        if probas[1]>sureté:
            if y_test.loc[i]==1:
                print('bet {}€ on player1 victory -'.format(round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))) ,
                      'match id {}'.format(i),
                      'on {}, odd to {} -'.format(cotes_1.idxmax()[:-1], cotes_1.max()),
                      'WIN!! - total : {}€'.format(cagnotte + round((mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))*(cotes_1.max()-1))))
                cagnotte += round((mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))*(cotes_1.max()-1))
            else:
                print('bet {}€ on player1 victory -'.format(round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))) ,
                      'on {}, odd to {} -'.format(cotes_2.idxmax()[:-1], cotes_2.max()),
                      'LOOSE!! - total : {}€'.format(cagnotte - round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))))
                cagnotte -= round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))
            mise_totale += round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))
        
        probas_test = probas[0]
        if probas[0]>sureté:
            if y_test.loc[i]==-1:
                print('bet {}€ on player2 victory -'.format(round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))) ,
                      'on {}, odd to {} -'.format(cotes_2.idxmax()[:-1], cotes_2.max()),
                      'WIN!! - total : {}€'.format(cagnotte + round((mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))*(cotes_2.max()-1))))
                cagnotte += round((mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))*(cotes_2.max()-1))
            else:
                print('bet {}€ on player2 victory -'.format(round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))) ,
                      'on {}, odd to {} -'.format(cotes_2.idxmax()[:-1], cotes_2.max()),
                      'LOOSE!! - total : {}€'.format(cagnotte - round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))))
                cagnotte -= round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))
            mise_totale += round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))
                
        
    print("La mise totale a été de ", mise_totale)
    print("La cagnote est de ", cagnotte)
    return cagnotte

if menu_sel == menu_lst[4]:

    st.header(title)
    st.subheader(menu_sel, anchor='Bet : win or loose ?')

    if st.checkbox('Expand text...'):
        st.write("""One of the ways to “beat the bookmakers” would have been to have a more accurate model than theirs. We didn't make it. 
    We now want to study the possibility of making money despite this 70% accuracy (good but only equal to that of bookmakers).""")

        st.write("Our model does no better than the bookmakers' predictions. We will therefore follow their strategy and check if we can still win by betting on several matches (following their strategy).")

        st.write("""The goal being to beat the bookmakers, you have to find the best way to bet without losing money.
        This involves not only accurately predicting the winner, but also choosing the right matches to bet on.""")

        st.write("""We create a function that performs the following tasks:
- First of all it predicts the outcome of all the matches of the test set
- Then, it looks for each outcome if the probability of its prediction is higher or not than a certain safety threshold (we set it to 0.8 by default).
o If the probability is lower, then we do not bet.
o If it is above, we bet.
- The stake bet on a match varies from 100 euros to €1 000.
o The more the algorithm is sure of its prediction, the closer we get to the maximum 1 000 euros bet.
o If the algorithm is at the level of the safety threshold, then we only bet 100 euros.""")

        st.write("""The jackpot displayed on each line corresponds to the net gain since the start of the simulation.
Recall that the algorithm is trained on the training game, and that this simulation is performed on the test game.""")

    if st.checkbox('Show bet_allocation() function...'):
        st.code("""
    def bet_allocation(base_cagnotte=0, mise_min=100, mise_max=1000, sureté=0.8):
        y_pred_proba = lr.predict_proba(X_test) 
        
        cagnotte = base_cagnotte
        
        mise_totale = 0
        
        y_pred_proba2 = [probas for probas in y_pred_proba]
        
        for i, probas in zip(y_test.index, y_pred_proba2):
        #for i, probas in enumerate(y_pred_proba):    
            cotes_1 = df_final[['ps1', 'b365_1']].loc[i]
            cotes_2 = df_final[['ps2', 'b365_2']].loc[i]
            
            probas_test = probas[1]
            if probas[1]>sureté:
                if y_test.loc[i]==1:
                    print('bet {}€ on player1 victory -'.format(round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))) ,
                        'match id {}'.format(i),
                        'on {}, odd to {} -'.format(cotes_1.idxmax()[:-1], cotes_1.max()),
                        'WIN!! - total : {}€'.format(cagnotte + round((mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))*(cotes_1.max()-1))))
                    cagnotte += round((mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))*(cotes_1.max()-1))
                else:
                    print('bet {}€ on player1 victory -'.format(round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))) ,
                        'on {}, odd to {} -'.format(cotes_2.idxmax()[:-1], cotes_2.max()),
                        'LOOSE!! - total : {}€'.format(cagnotte - round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))))
                    cagnotte -= round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))
                mise_totale += round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))
            
            probas_test = probas[0]
            if probas[0]>sureté:
                if y_test.loc[i]==-1:
                    print('bet {}€ on player2 victory -'.format(round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))) ,
                        'on {}, odd to {} -'.format(cotes_2.idxmax()[:-1], cotes_2.max()),
                        'WIN!! - total : {}€'.format(cagnotte + round((mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))*(cotes_2.max()-1))))
                    cagnotte += round((mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))*(cotes_2.max()-1))
                else:
                    print('bet {}€ on player2 victory -'.format(round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))) ,
                        'on {}, odd to {} -'.format(cotes_2.idxmax()[:-1], cotes_2.max()),
                        'LOOSE!! - total : {}€'.format(cagnotte - round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))))
                    cagnotte -= round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))
                mise_totale += round(mise_min+(mise_max-mise_min)*(probas_test-sureté)/(1-sureté))
                    
            
        print('Jackpot is : ', cagnotte) 
        print('...For a total bet of : ', mise_totale)
        return cagnotte""")
    
    st.image(path_imag + 'bet_allocation results.png', caption='bet_allocation() results.')

    st.write("""By betting on all the matches of our test game, respecting the safety criteria mentioned above, we nevertheless end up with a **negative balance (- €7 822)**. 
    Despite a total bet of **€2 840 464**.""")

#######################################
## PART 6 : INTERPRETABILITY         ##
#######################################

if menu_sel == menu_lst[5]:

    st.header(title)
    st.subheader(menu_sel, anchor='6. Model Interpretability')
    
    st.subheader('Features importance')

    st.write("""Regarding our model, the bookmakers odds, as well as the elo and ATP rankings are the variables that weigh the most in the choice of the winner between the challenger and the outsider.
    See the figure below : """)

    pd.Series(np.abs(lr.coef_[0]), X_train.columns).sort_values(ascending=False).plot(kind='barh', figsize=(4,8))
    plt.xlabel('Importance'); plt.ylabel('Features')
    st.pyplot()

#######################################
## PART 7 : CONCLUSION               ##
#######################################

if menu_sel == menu_lst[6]:

    st.header(title)
    st.subheader(menu_sel, anchor='Conclusion')

    st.write("""Throughout this project, it was a question of answering the question: “can we beat the bookmakers? ". The matches concerned being tennis matches.""")
    st.write("""To answer this question, we had at our disposal a dataset providing information on many matches from the period 2010 – 2018. The characteristics of these matches are numerous.""")
    st.write("""We first visualized the data via the matplotlib and seaborn python libraries.
    We then established relationships between certain variables, necessary for understanding the data and preparing the model.""")
    st.write("""We then built a model (logistic regression), after transforming our initial dataset. The model obtained finally obtained an accuracy equivalent to that of the bookmakers (70%).
    We wanted to get better accuracy.""")
    st.write("""Because beyond predicting better than the bookmakers, the implicit objective is to make money, we have to test the possibility of making money by following the predictions of our model.
    It was found that :""")
    st.write("**It is impossible to beat the bookmakers.**")
    st.write("""Suppose that 10 people bet at the same bookmaker, even if one of them manages to win (due to luck), the other 9 who will lose, will contribute to the bookmaker still winning overall.
    We understand better the expression “the casino never loses”.""")
    st.write("""The readings we conducted allowed us to discover a tiny way to beat the bookmakers. 
    We consider that this means is almost impossible to put into practice since it is more an error of the bookmakers than the performance of a machine learning model that we would have developed.
     *This is called odds arbitrage*.""")