import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os

############################################# Preprocessing

# definitions
punctuation = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
translationtable = str.maketrans("", "", punctuation)
dropcontactsduringcampaign = True  # for evaluating new campaign probabilities
datafile = 'bank.csv'
# datafile = 'bank-full.csv'
# fillnamethod = 'average'
fillnamethod = 'outlier'
# modelmethod = 'rf'  # random forest
# modelmethod = 'nusvc'  # Nu Support Vector Classifier
modelmethod = 'logreg'
plotfolder = modelmethod + 'plots'
# scoring = 'recall'  # good when you want to minimize FN
# scoring = 'precision'  # good when you want to minimize FP
scoring = 'f1_weighted'  # mix of recall and precision
rfmaxfeatures = ['sqrt', 'log2']
rfnestimators = [2, 5, 10, 15, 20, 50]
rfmaxdepth = [5, 10, 15, 20, 25]
nusvcgamma = [0.0001, 0.001, 0.01]
nusvckernel = ['linear', 'poly', 'rbf', 'sigmoid']
nusvcnu = [0.05, 0.1, 0.15, 0.2, 0.22]
logregc = [0.01, 0.1, 1e1, 1e2]
logregpenalty = ['l1', 'l2', 'none']  # elasticnet was not improving model
logregsolver = ['lbfgs', 'saga']
# these parameters were determined through MDI feature analysis of random forests
bestfeaturesfromrf = ['successpreviousoutcome', 'age', 'balance', 'dayofmonth', 'durationoflastcall',
                      'contactsbeforecampaign', 'dayssincecontactfrompreviouscampaign']
testresults = True  # DO NOT ENABLE UNLESS YOU ARE REALLY SURE YOUR MODEL IS GRRRRRRRRREAT
validationfolder = 'validationDONOTPEEK'

# Make a new folder to store plots in
if not os.path.exists(plotfolder):
    os.makedirs(plotfolder)

# Make a new folder for validation results
if testresults == True:
    if not os.path.exists(validationfolder):
        os.makedirs(validationfolder)

# grab the data to analyze
rawbankdata = pd.read_csv(datafile, sep=';', header=0)

# rename columns with clunky naming conventions
rawbankdata = rawbankdata.rename(columns={'y': 'subscribetermdeposit',
                                          'duration': 'durationoflastcall',
                                          'campaign': 'contactsduringcampaign',
                                          'contact': 'contacttype',
                                          'previous': 'contactsbeforecampaign',
                                          'poutcome': 'previousoutcome',
                                          'pdays': 'dayssincecontactfrompreviouscampaign',
                                          'day': 'dayofmonth'})
# %%
# preprocess the data

# dropping contactsduringpreviouscampaign should be toggled depending on if you want to evaluate the model for
# a new campaign OR evaluate within a current campaign. For new campaign, it should be off, as you will not have
# that information. If within a current campaign, you couls use the information.
if dropcontactsduringcampaign:
    rawbankdata = rawbankdata.drop(axis=1, labels=['contactsduringcampaign'])

# create list of variables for group processing
listofvariablestoonehotencode = ['job', 'marital', 'education', 'contacttype', 'month', 'previousoutcome']
listofvariablesnotonehotencoded = [x for x in rawbankdata.columns.tolist()
                                   if x not in listofvariablestoonehotencode]
listofvariablestomakebinary = ['default', 'housing', 'loan', 'subscribetermdeposit']
if dropcontactsduringcampaign:
    listofvariablestostandardize = ['age', 'balance', 'dayofmonth', 'durationoflastcall',
                                    'contactsbeforecampaign',
                                    'dayssincecontactfrompreviouscampaign']
if not dropcontactsduringcampaign:
    listofvariablestostandardize = ['age', 'balance', 'dayofmonth', 'durationoflastcall',
                                    'contactsbeforecampaign',
                                    'contactsduring campaign', 'dayssincecontactfrompreviouscampaign']

# create encoder and transform listofvariablestoonehotencode
enc = OneHotEncoder(handle_unknown='ignore')
jobonehotencoded = enc.fit_transform(rawbankdata[listofvariablestoonehotencode]).toarray()

# get column names for transformed variable data
listofuniquevaluesforvariablestoonehotencode = []
for variable in listofvariablestoonehotencode:
    # create list of sorted unique values for each variable
    listofsortedvariables = sorted(rawbankdata[variable].unique().tolist())
    # add the variable name to all items to avoid duplicate value names
    listofsortedvariables = [(item.translate(translationtable) + variable) for item in listofsortedvariables]
    listofuniquevaluesforvariablestoonehotencode += listofsortedvariables

# grab info from columns that were not onehotencoded
processedbankdata = pd.DataFrame(
    jobonehotencoded, columns=listofuniquevaluesforvariablestoonehotencode).join(
    rawbankdata[listofvariablesnotonehotencoded])

# convert yes/no values for variables to 1/0
for nonbinaryvariable in listofvariablestomakebinary:
    processedbankdata[nonbinaryvariable] = pd.Series(
        np.where(processedbankdata[nonbinaryvariable].values == 'yes', 1, 0),
        processedbankdata.index)

# convert -1 value in pdays for client not contacted previously to None
processedbankdata['dayssincecontactfrompreviouscampaign'] = processedbankdata[
    'dayssincecontactfrompreviouscampaign'].replace(-1, np.nan)

######################## makes plots of processed data
# plot counts of target
# plt.rcdefaults
fig, ax = plt.subplots()
model_list = sorted(rawbankdata['subscribetermdeposit'].unique().tolist())
model_dict = {}
for i in model_list:
    model_dict[i] = len(rawbankdata[rawbankdata['subscribetermdeposit'] == i])
y = []
x = []
for i in range(0, len(model_list)):
    x.append(model_dict[model_list[i]])
    y.append(model_list[i])
y_pos = np.arange(len(y))
ax.barh(y_pos, x, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(y)
# ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Count')
plt.xlim(0, len(rawbankdata))
ax.set_title('Clients that subscribe a term deposit')
for i, v in enumerate(x):
    ax.text(v + 10, i, str(v), color='black', va='center', fontweight='normal')
plt.savefig(f'{modelmethod}TargetCounts.png')

# plot percentage of target
fig1, ax1 = plt.subplots()
ax1.pie(x, labels=y, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Percentage of clients that subscribe a term deposit', size=16)
plt.savefig(f'{modelmethod}TargetPercentage.png')  # pie chart, slices ordered and plotted CCW
plt.close()

# make pie charts for multilabeled variables
for classifiervariable in listofvariablestoonehotencode:
    # plt.rcdefaults
    fig, ax = plt.subplots(figsize=(16, 16))
    model_list = sorted(rawbankdata[classifiervariable].unique().tolist())
    model_dict = {}
    for i in model_list:
        model_dict[i] = len(rawbankdata[rawbankdata[classifiervariable] == i])
    y = []
    x = []
    for i in range(0, len(model_list)):
        x.append(model_dict[model_list[i]])
        y.append(model_list[i])
    # plot percentage of target
    fig1, ax1 = plt.subplots()
    ax1.pie(x, labels=y, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f'Client {classifiervariable}', size=16)
    plt.savefig(f'{plotfolder}/{classifiervariable}TargetPercentage.png')
    plt.close()

# heatmap to show correlation between input parameters and target
processedbankdatacorr = processedbankdata.copy()
# Build unique columns for each best_model and combine them at the end
j = 0
unique_model_list = []
for i in processedbankdatacorr['subscribetermdeposit'].unique():
    processedbankdatacorr[f'{i}'] = np.where(processedbankdatacorr['subscribetermdeposit'] == i, j, 0)
    unique_model_list.append(str(i))
    j += 1
processedbankdatacorr['category'] = processedbankdatacorr.iloc[:, -len(unique_model_list):].sum(axis=1)
processedbankdatacorr = processedbankdatacorr.drop(columns=unique_model_list)
processedbankdatacorr = processedbankdatacorr.drop(columns=['subscribetermdeposit'])
processedbankdatacorr = processedbankdatacorr.astype(float)  # corr only works on float, default is object in dataframe
top = len(processedbankdatacorr.columns.tolist())
corr = processedbankdatacorr.corr()
top_top = corr.nlargest(top, 'category')['category'].index
corr_top_top = processedbankdatacorr[top_top].corr()
f, ax = plt.subplots(figsize=(20, 20))
# plt.rcParams['font.size'] = 40
heatmap = sns.heatmap(
    corr_top_top, square=True, ax=ax, annot=False, cmap='coolwarm', fmt='.2f', annot_kws={'size': 30})
plt.title('Top correlated features of dataset', size=40)
plt.savefig(f'{modelmethod}TargetHeatmap.png')
plt.close()


def removeextremeoutliers(dataseries):
    """Remove extreme outliers from the dataset"""
    seriesave = dataseries.mean()
    dataseriesnooutliers = dataseries[abs(dataseries) / seriesave <= 5]
    return dataseriesnooutliers


for i in listofvariablestostandardize:
    # get rid of extreme outliers to clear up histograms
    ax = plt.hist(removeextremeoutliers(processedbankdata[i]))
    plt.xlabel(i, fontsize=28)
    plt.ylabel('normalized counts', fontsize=28)
    plt.tick_params(axis='both', labelsize=28)
    plt.savefig(f'{plotfolder}/Target_{i}')
    plt.close()


# try:
#     processedbankdata.to_csv('bankdataprocessed.csv')
# except:
#     print('csv file is open, cannot make local copy of data')

############################################### Modeling

class Standardizer:
    """Class that standardizes the data, then applies either an average or outlier technique on values in pdays.
    Required for pipeline usage."""

    def __init__(self, fillnamethod):
        self.scaler = StandardScaler()
        self.fillnamethod = fillnamethod

    # def __init__(self):
    #     self.scaler = StandardScaler()

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, dataset, datasetoutput=0):
        """Used in pipeline to fit the training data only"""
        self.scaler.fit(dataset)
        return self

    def transform(self, dataset, datasetoutput=0):
        """Used in pipeline to transform all data"""
        self.datasetscaled = self.scaler.transform(dataset)
        # replace nan values in pdays column
        if self.fillnamethod == 'average':
            self.datasetscaled[np.isnan(self.datasetscaled)] = np.nanmean(self.datasetscaled[:, -2])
        if self.fillnamethod == 'outlier':
            self.datasetscaled[np.isnan(self.datasetscaled)] = -5
        return self.datasetscaled


# create test dataset only to be examined at end of experiment
bankdatatrainandvalidate, bankdatatest, bankdatatrainandvalidateoutput, bankdatatestoutput = train_test_split(
    processedbankdata.iloc[:, :-1],
    processedbankdata.iloc[:, -1:], test_size=0.15,
    stratify=processedbankdata.iloc[:, -1:],
    random_state=42)

gridsearchcvparameters = {}
pipe = []

if modelmethod == 'rf':
    gridsearchcvparameters = {
        'model__max_features': rfmaxfeatures,
        'model__n_estimators': rfnestimators,
        'model__max_depth': rfmaxdepth,
    }

    pipe = Pipeline([
        ('scaler', Standardizer(fillnamethod=fillnamethod)),
        ('model', RandomForestClassifier(class_weight='balanced'))
    ])

if modelmethod == 'nusvc':
    gridsearchcvparameters = {
        'model__gamma': nusvcgamma,
        'model__kernel': nusvckernel,
        'model__nu': nusvcnu,
    }

    pipe = Pipeline([
        ('scaler', Standardizer(fillnamethod=fillnamethod)),
        ('model', NuSVC(class_weight='balanced'))
    ])

if modelmethod == 'logreg':
    gridsearchcvparameters = {
        'model__C': logregc,
        'model__penalty': logregpenalty,
        'model__solver': logregsolver,
    }

    pipe = Pipeline([
        ('scaler', Standardizer(fillnamethod=fillnamethod)),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

logscaleparameters = ['model__C', 'model__gamma']

model = GridSearchCV(pipe, gridsearchcvparameters, cv=5, scoring=scoring, return_train_score=True)

if modelmethod == 'rf':
    model.fit(bankdatatrainandvalidate, bankdatatrainandvalidateoutput.values.ravel())
# trim variables for nusvc that were poor performers from rf
if modelmethod == 'nusvc':
    bankdatatrainandvalidate = bankdatatrainandvalidate[bestfeaturesfromrf]
    model.fit(bankdatatrainandvalidate, bankdatatrainandvalidateoutput.values.ravel())
    bankdatatest = bankdatatest[bestfeaturesfromrf]
if modelmethod == 'logreg':
    model.fit(bankdatatrainandvalidate, bankdatatrainandvalidateoutput.values.ravel())
results = pd.DataFrame(model.cv_results_)
results.to_csv(f'{modelmethod}CVresults.csv')

# todo only works for three hyperparameters
for hyperparameter in gridsearchcvparameters.keys():

    # add string value so hyperparameter can be read as results column
    resultscolumnaname = 'param_' + hyperparameter
    # list comprehension to extract other hyperparameter names
    otherhyperparameters = [hyperparameteritem for hyperparameteritem in gridsearchcvparameters.keys() if
                            hyperparameteritem not in hyperparameter]

    # plot how hyperparameter changes as we alter a single other hyperparameter at a time
    for allotherhyperparameters in range(0, len(otherhyperparameters)):
        # select a single parameter from the remaining
        otherhyperparameter = otherhyperparameters[allotherhyperparameters]
        # add string value so hyperparameter compared can be read as results column
        resultscolumnothername = 'param_' + otherhyperparameter

        # plot for each value of other hyperparameter given to gridsearchcv
        for otherhyperparametervariationlength in gridsearchcvparameters[otherhyperparameter]:
            # take one value for hyperparameter given by gridsearchcvparameters
            resultsbyotherhyperparameter = results.loc[results[resultscolumnothername] ==
                                                       otherhyperparametervariationlength]

            finalhyperparameter = [hyperparameteritem for hyperparameteritem in gridsearchcvparameters.keys() if
                                   hyperparameteritem not in [hyperparameter, otherhyperparameter]][0]
            resultscolumnfinalname = 'param_' + finalhyperparameter

            for hyperparametervalue in gridsearchcvparameters[hyperparameter]:
                resultsbyhyperparameter = resultsbyotherhyperparameter.loc[
                    resultsbyotherhyperparameter[resultscolumnaname] == hyperparametervalue]

                barplotlabels = []
                train_mean_list = []
                train_std_list = []
                validate_mean_list = []
                validate_std_list = []

                barplotlabels = barplotlabels + resultsbyhyperparameter[resultscolumnfinalname].tolist()
                validate_mean_list = validate_mean_list + resultsbyhyperparameter[
                    'mean_test_score'].tolist()
                validate_std_list = validate_std_list + resultsbyhyperparameter['std_test_score'].tolist()
                train_mean_list = train_mean_list + resultsbyhyperparameter['mean_train_score'].tolist()
                train_std_list = train_std_list + resultsbyhyperparameter['std_train_score'].tolist()
                # setup plots
                barplotposition = np.arange(len(barplotlabels))
                width = 0.35  # width of bars
                plt.clf()
                fig = plt.figure(figsize=[12, 10])
                # fig, ax = plt.subplots()
                trainbarplots = plt.bar(barplotposition - width / 2, train_mean_list, width, label='train',
                                        yerr=train_std_list)
                validatebarplots = plt.bar(barplotposition + width / 2, validate_mean_list, width, label='validate',
                                           yerr=validate_std_list)
                plt.title(label=(hyperparameter[7:] + '=' + str(
                    hyperparametervalue) + ', ' + otherhyperparameter[7:] + '=' + str(
                    otherhyperparametervariationlength) + ', with changing ' + finalhyperparameter[7:]))
                plt.xticks(barplotposition, barplotlabels)
                plt.ylabel(f'{modelmethod} score')
                plt.tick_params(labelsize='large')
                # if finalhyperparameter in logscaleparameters:
                #     plt.xscale('log')
                plt.xlabel(finalhyperparameter[7:], fontsize='large')
                plt.ylabel('F1 score', fontsize='large')
                plt.autoscale(tight=True)
                plt.legend(loc='best')
                plt.savefig(
                    f'{plotfolder}/{modelmethod}{hyperparameter[7:]}{hyperparametervalue}and{otherhyperparameter[7:]}{otherhyperparametervariationlength}constantvs{finalhyperparameter[7:]}trainvstest.png')
                plt.close()
                # print(hyperparameter, otherhyperparameter, finalhyperparameter)
                # input('check plot')

# create subset of data to model using confusion matrices and permutation importance
bankdatatrain, bankdatavalidate, bankdatatrainoutput, bankdatavalidateoutput = train_test_split(
    bankdatatrainandvalidate,
    bankdatatrainandvalidateoutput, test_size=0.15,
    stratify=bankdatatrainandvalidateoutput,
)

if modelmethod == 'rf':
    bestrfmodel = model.best_estimator_.steps[1][1]
    importances = bestrfmodel.feature_importances_
    stdfeatures = np.std([tree.feature_importances_ for tree in bestrfmodel.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=bankdatatrainandvalidate.columns.tolist())

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=stdfeatures, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(f'{modelmethod}MDIfeatureanalysis.png')

    rfpermutationimportance = permutation_importance(
        model, bankdatatrain, bankdatatrainoutput, n_repeats=10,
        random_state=42, n_jobs=2)

    sortedrfpermutation = rfpermutationimportance.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(rfpermutationimportance.importances[sortedrfpermutation].T,
               vert=False, labels=bankdatatrain.columns[sortedrfpermutation])
    ax.set_title("Permutation importance (train set)")
    plt.tight_layout()
    plt.savefig(f'{modelmethod}Permutationimportances.png')
    plt.close()

if modelmethod == 'nusvc':
    nusvcweights = pd.Series(model.best_estimator_.steps[1][1].class_weight_)
    nusvcclasses = pd.Series(model.best_estimator_.steps[1][1].classes_)
    nusvcdualcoefs = model.best_estimator_.steps[1][1].dual_coef_
    nusvcsupportindexes = pd.Series(model.best_estimator_.steps[1][1].support_)
    nusvcsupportvectors = pd.DataFrame(model.best_estimator_.steps[1][1].support_vectors_)
    nusvcnsupportvectors = pd.DataFrame(model.best_estimator_.steps[1][1].n_support_)
    nusvcparams = pd.Series(model.best_estimator_.steps[1][1].get_params())
    nusvcweights.to_csv(f'{modelmethod}weights.csv')
    nusvcsupportindexes.to_csv(f'{modelmethod}indexes.csv')
    nusvcsupportvectors.to_csv(f'{modelmethod}supportvectors.csv')
    nusvcnsupportvectors.to_csv(f'{modelmethod}supportnvectors.csv')
    nusvcparams.to_csv(f'{modelmethod}params.csv')

if modelmethod == 'logreg':
    logregcoefsarray = model.best_estimator_.steps[1][1].coef_
    # logregcoefsarrayimpact = [impact for impact in logregcoefsarray[0] if abs(impact) > 0.02]
    # logregcoefsimpactvariables = [bankdatatrain.columns[variablenumber] for variablenumber in
    #                               range(len(logregcoefsarray[0])) if abs(variablenumber) > 0.02]
    logregcoefs = pd.DataFrame(logregcoefsarray, columns=bankdatatrain.columns)
    logregcoefs.to_csv(f'{modelmethod}coefs.csv')
    plt.bar(bankdatatrain.columns, logregcoefsarray[0])
    plt.xticks(rotation=90)
    plt.ylabel('logreg variable coefficient')
    plt.tight_layout()
    plt.savefig(f'{modelmethod}featureimportance.png')
    plt.close()
    # plt.bar(bankdatatrain.columns, logregcoefsarrayimpact)
    # plt.xticks(rotation=90)
    # plt.ylabel('logreg variable coefficient')
    # plt.tight_layout()
    # plt.savefig(f'{modelmethod}featureimportance.png')
    # plt.close()

# performance on test data DO NOT LOOK AT UNTIL YOU FEEL PRETTY GOOD ABOUT THE MODEL PERFORMANCE
if testresults:
    # compute the class weight for the test data set
    testclasstrue = np.ones(bankdatatestoutput.sum())
    testclassfalse = np.zeros(len(bankdatatestoutput) - bankdatatestoutput.sum())
    testclass = np.concatenate([testclassfalse, testclasstrue])
    testclassweight = compute_class_weight('balanced', [0, 1], testclass)
    testclassweight = testclassweight / sum(testclassweight)
    testclasssampleweight = np.where(bankdatatestoutput == 1, testclassweight[1], testclassweight[0])
    disp = plot_confusion_matrix(model.best_estimator_, bankdatatest, bankdatatestoutput,
                                 cmap=plt.cm.Blues, normalize='true', display_labels=['no', 'yes'])
    disp.ax_.set_title('Confusion matrix')
    plt.savefig(f'{validationfolder}/{modelmethod}confusionmatrix.png')
    # make predictions on test data set
    testpredictions = model.best_estimator_.predict(bankdatatest)
    modelpredictions = pd.DataFrame(testpredictions,
                                    index=bankdatatest.index).join(bankdatatestoutput)
    # calculate f1 score of model based on weight of respective outputs
    modelscore = pd.Series(f1_score(testpredictions, bankdatatestoutput,
                                    sample_weight=testclasssampleweight.tolist()),
                           index=modelpredictions.index)
    modelpredictions['f1score'] = modelscore
    modelpredictions = modelpredictions.rename(columns={
        '0' : 'modelpredictions',
        'subscribetermdeposit' : 'trueresults'
    })
    modelpredictions.to_csv(f'{validationfolder}/{modelmethod}testpredictions.csv')

# generate example confusion matrix based on model best estimator on bankdatatrain only
if not testresults:
    confusionmatrixmodel = model.best_estimator_.fit(
        bankdatatrain, bankdatatrainoutput.values.ravel())
    disp = plot_confusion_matrix(confusionmatrixmodel, bankdatavalidate,
                                 bankdatavalidateoutput.values.ravel(),
                                 cmap=plt.cm.Blues, normalize='true', display_labels=['no', 'yes'])
    disp.ax_.set_title('Confusion matrix')
    print(disp.confusion_matrix)
    plt.savefig(f'{modelmethod}exampleconfusionmatrix.png')
