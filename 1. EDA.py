import os, sys
sys.path.append(os.path.expanduser('~') + '/Documents/Python/Custom Modules')
from DataScience import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize project directory:
project_dir = os.path.expanduser('~') + "/OneDrive/Kaggle/Sberbank Russian Housing Market/Sberbank Russian Housing Market"
os.chdir(project_dir)

color = sns.color_palette()
pd.set_option('display.max_columns', 500)
sns.set(font_scale=1.75)

# Read in training data:
train_df = pd.read_csv(project_dir + "/Data/train.csv", parse_dates=['timestamp'])
train_df['price_doc_log'] = np.log1p(train_df['price_doc'])

# Some basic information
train_df.shape
train_df.head()
train_df.columns
train_df.describe()

# Visualize missing data:
train_na = (train_df.isnull().sum() / len(train_df)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)

f, ax = plt.subplots(figsize=(24, 16))
plt.xticks(rotation='90')
sns.barplot(x=train_na.index, y=train_na)
ax.set(title='Percent missing data by feature', ylabel='% missing')
plt.subplots_adjust(top = 0.95, bottom = 0.3)
plt.show()

# Visualize the response variable:
plt.hist(train_df['price_doc'], bins = 50)
plt.show()

plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
plt.xlabel('Index', fontsize=20)
plt.ylabel('Price', fontsize=20)
plt.show()

plt.hist(train_df['price_doc_log'], bins = 50)
plt.show()

plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc_log.values))
plt.xlabel('Index', fontsize=20)
plt.ylabel('Price', fontsize=20)
plt.show()

# Visualize median house prices over time:
train_df['yearmonth'] = train_df['timestamp'].map(lambda x: 100*x.year + x.month)
train_monthgrp = train_df.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()

plt.figure(figsize=(24, 16))
sns.barplot(train_monthgrp.yearmonth.values, train_monthgrp.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=18)
plt.xlabel('Year Month', fontsize=18)
plt.xticks(rotation='vertical')
plt.show()

# Some other useful plots:
sns.regplot(x = 'full_sq', y = 'price_doc', data = train_df, fit_reg = False)
plt.show()
sns.regplot(x = 'full_sq', y = 'price_doc', data = train_df[train_df['full_sq'] < 1000]  , fit_reg = False)
plt.show()

sns.regplot(x = 'num_room', y = 'price_doc', data = train_df, fit_reg = False)
plt.show()

train_df['work_share'] = train_df['work_all']/train_df['raion_popul']
train_workshrgrp = train_df.groupby('sub_area')[['work_share', 'price_doc']].mean()
sns.regplot(x = 'work_share', y = 'price_doc', data = train_workshrgrp, order = 4, ci = 95)
plt.show()

train_sportsgrp = train_df.groupby('sub_area')[['sport_objects_raion', 'price_doc']].median()
sns.regplot(x = 'sport_objects_raion', y = 'price_doc', data = train_sportsgrp, ci = 95)
plt.show()

sns.violinplot(x = 'state', y = 'price_doc_log', data = train_df, inner='quartile')
plt.show()

sns.violinplot(x = 'material', y = 'price_doc_log', data = train_df, inner='quartile')
plt.show()

# Heatmap of correlation matrix for certain variables in the data:
school_chars = ['children_preschool', 'preschool_quota', 'preschool_education_centers_raion', 'children_school', 
                'school_quota', 'school_education_centers_raion', 'school_education_centers_top_20_raion', 
                'university_top_20_raion', 'additional_education_raion', 'additional_education_km', 'university_km', 'price_doc']
corrmat1 = train_df[school_chars].corr()

plt.subplots(figsize=(18, 15))
sns.heatmap(corrmat1, square=True, linewidths=.5, annot=True)
plt.xticks(rotation='90')
plt.yticks(rotation='0')
plt.subplots_adjust(top = 0.95, bottom = 0.3, left = 0.2, right = 1)
plt.show()

inf_features = ['nuclear_reactor_km', 'thermal_power_plant_km', 'power_transmission_line_km', 'incineration_km',
                'water_treatment_km', 'incineration_km', 'railroad_station_walk_km', 'railroad_station_walk_min', 
                'railroad_station_avto_km', 'railroad_station_avto_min', 'public_transport_station_km', 
                'public_transport_station_min_walk', 'water_km', 'mkad_km', 'ttk_km', 'sadovoe_km','bulvar_ring_km',
                'kremlin_km', 'price_doc']
corrmat2 = train_df[inf_features].corr()

plt.subplots(figsize=(20, 18))
sns.heatmap(corrmat2, square=True, linewidths=.5, annot=True)
plt.xticks(rotation='90')
plt.yticks(rotation='0')
plt.subplots_adjust(top = 0.95, bottom = 0.25, left = 0.2, right = 1)
plt.show()