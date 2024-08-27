# Analysing News Articles Dataset

![](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](URL_TO_YOUR_APP)

<div id="main image" align="center">
  <img src="https://github.com/ereshia/2401FTDS_Classification_Project/blob/main/announcement-article-articles-copy-coverage.jpg" width="550" height="300" alt=""/>
</div>

## Table of contents
* [1. Project Overview](#project-description)
* [2. Dataset](#dataset)
* [3. Packages](#packages)
* [4. Environment](#environment)
* [5. MLFlow](#mlflow)
* [6. Streamlit](#streamlit)

  ## 1. Project Overview <a class="anchor" id="project-description"></a>

  Your team has been hired as data science consultants for a news outlet to create classification models using Python and deploy it as a web application with Streamlit. 
The aim is to provide you with a hands-on demonstration of applying machine learning techniques to natural language processing tasks.  This end-to-end project encompasses the entire workflow, including data loading, preprocessing, model training, evaluation, and final deployment. The primary stakeholders for the news classification project for the news outlet could include the editorial team, IT/tech support, management, readers, etc. These groups are interested in improved content categorization, operational efficiency, and enhanced user experience.

## 2. Dataset <a class="anchor" id="dataset"></a>
The dataset is comprised of news articles that need to be classified into categories based on their content, including `Business`, `Technology`, `Sports`, `Education`, and `Entertainment`. You can find both the `train.csv` and `test.csv` datasets [here](https://github.com/ereshia/2401FTDS_Classification_Project/tree/main/Data/processed).

**Dataset Features:**
| **Column**                                                                                  | **Description**              
|---------------------------------------------------------------------------------------------|--------------------   
| Headlines   | 	The headline or title of the news article.
| Description | A brief summary or description of the news article.
| Content | The full text content of the news article.
| URL | The URL link to the original source of the news article.
| Category | The category or topic of the news article (e.g., business, education, entertainment, sports, technology).

## 3. Packages <a class="anchor" id="packages"></a>

To carry out all the objectives for this repo, the following necessary dependencies were loaded:
+ `Pandas 2.2.2` and `Numpy 1.26`
+ `Matplotlib 3.8.4`
+ `nlk 3.8.1`
+ `Gensim 4.3.2`
+ `scipy 1.13.2`
+ `pyLDAvis 3.4.1`
+ `mpcursors 0.5.3`
+ `S[acy 3.7.5`
+ `Wordcloud 1.9.3`
+ `umap-learn seaborn 0.5.5`
+ `Seaborn`
+ `collections`
+ `nltl.sentiment.vader`
+ `sklearn.feature_extraction.text`
+ `sklearn`
+ `umap`
+ `textblob`
+ `sklearn.feature_selection`
+ `sklearn.preprocessing`
+ `sklearn.feature_extraction.text`
+ `sklearn.preprocessing and StandardScaler`
+ `imblearn.over_selection and SMOTE`
+ `sklearn.model_selection and train_test_split`
+ `sklearn.ensemble and RandomForestClassifier`
+ `sklearn.pipeline and classification_report, roc_auc_score`
+ `imblearn.pipeline and make_pipeline`
+ `matplotlib.pyplot`
+ `sklearn.model_select and make_pipeline`
+ `sklearn.metrics and classification_report`
+ `sklearn.linear_model and logisticRegression`
+ `sklearn.metrics and confusion_matrix`
+ `sklearn.tree and DecisionTreeClassifier`
+ `sklearn.svm and SVC`
+ `sklearn.ensemble and GradientBoostingClassifier`
+ `sklearn.metrics and mean_squared_error`
+ `sklearn.model_selection and GridSearchCV`
+ `sklearn.metrics and accuracy_score, classification_report, f1_score`
+ `mlflow`
+ `mlflow.sklearn`
+ `pickle` 

## 4. Environment <a class="anchor" id="environment"></a>
### Create the new evironment - you only need to do this once

```bash
# create the conda environment
conda create --name <env>
```

### This is how you activate the virtual environment in a terminal and install the project dependencies

```bash
# activate the virtual environment
conda activate <env>
# install the pip package
conda install pip
# install the requirements for this project
pip install -r requirements.txt
```

## 5. MLFlow<a class="anchor" id="mlflow"></a>
MLOps, which stands for Machine Learning Operations, is a practice focused on managing and streamlining the lifecycle of machine learning models. The modern MLOps tool, MLflow is designed to facilitate collaboration on data projects, enabling teams to track experiments, manage models, and streamline deployment processes. For experimentation, testing, and reproducibility of the machine learning models in this project, you will use MLflow. MLflow will help track hyperparameter tuning by logging and comparing different model configurations. This allows you to easily identify and select the best-performing model based on the logged metrics.

## 6. Streamlit<a class="anchor" id="streamlit"></a>

[Streamlit](https://www.streamlit.io/)  is a framework that acts as a web server with dynamic visuals, multiple responsive pages, and robust deployment of your models.

In its own words:
> Streamlit ... is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours!  All in pure Python. All for free.

> It’s a simple and powerful app model that lets you build rich UIs incredibly quickly.

[Streamlit](https://www.streamlit.io/)  takes away much of the background work needed in order to get a platform which can deploy your models to clients and end users. Meaning that you get to focus on the important stuff (related to the data), and can largely ignore the rest. This will allow you to become a lot more productive.  

