import os, sys
sys.path.append(os.path.expanduser('~') + '/Documents/Python/Custom Modules')
from DataScience import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_dir = os.path.expanduser('~') + "/OneDrive/Kaggle/Sberbank Russian Housing Market/Sberbank Russian Housing Market"

#################################
# Import Data and Initial Setup #

# Read in csv:
Sb_train = pd.read_csv(project_dir + "/Data/train.csv")
Sb_test = pd.read_csv(project_dir + "/Data/test.csv")
Sb_macro = pd.read_csv(project_dir + "/Data/macro.csv")

# Check ID columns:
len(Sb_train)
len(Sb_test)
len(Sb_macro)

len(Sb_train.columns)
len(Sb_test.columns)
len(Sb_macro.columns)

len(Sb_train['id'].unique())
len(Sb_test['id'].unique())
len(Sb_macro['timestamp'].unique())

# Pull out target variable from training set:
Sb_train_price = Sb_train[['id', 'price_doc']].copy()
Sb_train.drop('price_doc', axis = 1, inplace = True)
len(Sb_train.columns)

# Concatenate train and test, and feature engineering:
Sb_all = pd.concat([Sb_train, Sb_test])
Sb_all.shape

##################
# Data Cleansing #

# Drop features that has little to no variance:
zero_variance = ["culture_objects_top_25_raion", "oil_chemistry_raion", "railroad_terminal_raion", "nuclear_reactor_raion",
				"build_count_foam", "big_road1_1line", "railroad_1line", "office_sqm_500", "trc_sqm_500",
				"cafe_count_500_price_4000", "cafe_count_500_price_high", "mosque_count_500", "leisure_count_500",
				"office_sqm_1000", "trc_sqm_1000", "cafe_count_1000_price_high", "mosque_count_1000", "cafe_count_1500_price_high",
				"mosque_count_1500", "cafe_count_2000_price_high"]
useless_ids = ["ID_metro", "ID_railroad_station_walk", "ID_railroad_station_avto", "ID_big_road1", "ID_big_road2",
			   "ID_railroad_terminal", "ID_bus_terminal"]

Sb_all.drop(zero_variance + useless_ids, axis = 1, inplace = True)
Sb_all.shape

# Identified data quality checks:
Sb_all['state'].value_counts()
Sb_all['state'].replace(to_replace = 33, value = 3, inplace = True)

Sb_all['build_year'].value_counts()
bld_yr_replace = {20052009:2005, 0:np.nan, 1:np.nan, 2:np.nan, 3:np.nan, 20:2000, 215:2015, 4965:1965, 71:1971}
Sb_all['build_year'].replace(to_replace = bld_yr_replace, inplace = True)

Sb_all['material'].value_counts()
Sb_all['material'].replace(to_replace = 3, value = 1, inplace = True)

Sb_all['full_sq'].describe()
Sb_all['full_sq'].replace(to_replace = 0, value = np.nan, inplace = True)

Sb_all['num_room'].describe()
Sb_all['num_room'].replace(to_replace = 0, value = np.nan, inplace = True)

Sb_all['max_floor'].describe()
max_flr_replace = {117:17, 99:np.nan, 0:np.nan}
Sb_all['max_floor'].replace(to_replace = max_flr_replace, inplace = True)

#######################
# Feature Engineering #

# Additional timestamp variables:
Sb_all['year'] = Sb_all['timestamp'].apply(lambda x: int(x[0:4]))
Sb_all['year_mo'] = Sb_all['timestamp'].apply(lambda x: x[0:7])

# Years-old variable:
Sb_all['years_old'] = 2020 - Sb_all['build_year']

# Residential & kitchen area to total area ratio:
Sb_all['resident_to_total_ratio'] = Sb_all['life_sq']/Sb_all['full_sq']
Sb_all['kitchen_to_total_ratio'] = Sb_all['kitch_sq']/Sb_all['full_sq']

# Average area per room:
Sb_all['avg_room_area'] = Sb_all['life_sq']/Sb_all['num_room']

# Extra area:
Sb_all['extra_area'] = Sb_all['full_sq'] - Sb_all['life_sq']
Sb_all['extra_area_ratio'] = Sb_all['extra_area']/Sb_all['full_sq']

# Percentage of population in labor force:
Sb_all['pct_labor_force'] = Sb_all['work_all']/Sb_all['raion_popul']

# Apartment floor relative to building height:
Sb_all['floor_rel_total'] = Sb_all['floor']/Sb_all['max_floor']

# Ratio of schoolage children to available seats:
Sb_all['ratio_school_age_seats'] = Sb_all['children_school']/Sb_all['school_quota']
Sb_all['ratio_preschool_age_seats'] = Sb_all['children_preschool']/Sb_all['preschool_quota']

# Demographic structures of subareas:
Sb_all['young_proportion'] = Sb_all['young_all']/Sb_all['full_all']
Sb_all['work_proportion'] = Sb_all['work_all']/Sb_all['full_all']
Sb_all['retire_proportion'] = Sb_all['ekder_all']/Sb_all['full_all']
Sb_all['female_to_male'] = Sb_all['female_f']/Sb_all['male_f']

# Some additional binary variables:
Sb_all['metro_flag'] = np.where(Sb_all['raion_popul'] > 150000, 1, 0)
Sb_all['large_flag'] = np.where(Sb_all['max_floor'] >= 20, 1, 0)
Sb_all['small_flag'] = np.where(Sb_all['max_floor'] <= 20, 1, 0)

# Average building height for subarea:
sub_area_means = Sb_all.groupby('sub_area').agg({'max_floor':np.mean}).reset_index().rename(columns={'max_floor':'mean_bldg_height'})
Sb_all = pd.merge(Sb_all, sub_area_means, on = ['sub_area'], how = 'left')

# Sales by month:
n_sales_months = Sb_all.groupby('year_mo').size().reset_index().rename(columns={0:'n_sales_month'})
Sb_all = pd.merge(Sb_all, n_sales_months, on = ['year_mo'], how = 'left')

# Average distance to Kremlin by subarea:
dist_to_kremlin = Sb_all.groupby('sub_area').agg({'kremlin_km':np.nanmean}).reset_index().rename(columns={'kremlin_km':'subarea_dist_to_kremlin'})
Sb_all = pd.merge(Sb_all, dist_to_kremlin, on = ['sub_area'], how = 'left')

# Count NaNs per row:
Sb_all['count_nan_per_row'] = Sb_all.isnull().sum(axis = 1)

# Apartment name:
Sb_all['apt_name'] = Sb_all['sub_area'] + Sb_all['metro_km_avto'].astype(str).apply(lambda x: x[0:5])
Sb_all['apt_name_yrmo'] = Sb_all['apt_name'] + Sb_all['year_mo']

# Floored full_sq variable:
Sb_all['full_sq_floored'] = Sb_all['full_sq'].apply(lambda x: max(x, 50))

##########################
# Merge in Macro Dataset #

# Note: may not want to do this - likely to bring down score
#Sb_all = pd.merge(Sb_all, Sb_macro, on = ['timestamp'], how = 'left')
#Sb_all.shape

##################################
# Re-separate Training & Testing #

Sb_all.shape

Sb_train_fe = Sb_all[Sb_all['id'].isin(Sb_train_price['id'])]
Sb_train_fe = pd.merge(Sb_train_fe, Sb_train_price, on = ['id'], how = 'inner')
Sb_train_fe.shape

Sb_test_fe = Sb_all[~Sb_all['id'].isin(Sb_train_price['id'])]
Sb_test_fe.shape