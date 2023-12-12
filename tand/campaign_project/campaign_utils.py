import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from datetime import timedelta


class CampaignAnalysis:

    def __init__(self):
        pass

    def calc_kpi_conv(df_campaign, df_event, col_key, col_open_boolean, col_day, conv_period = 7):
        # 1. find device/user which/who open campaign
        df_open_target = df_campaign[df_campaign[col_open_boolean]==True]
        set_key_open = set()
        for i in range(0, conv_period):
            set_key_open = set_key_open | set(df_open_target[col_key] + '_' + df_open_target[col_day] + timedelta(days=i))

        # 2, find device/user which/who did kpi event
        df_event['kpi_key'] = df_event[col_key] + '_' + df_event[col_day]
        set_key_kpi = set(df_event[col_key] + '_' + df_event[col_day])
        
        # 3. find device/user which/who is converted to kpi within conv_period
        set_key_conv = set_key_open & set_key_kpi
        df_event['conv_boolean'] = df_event['kpi_key'].apply(lambda x: True if x in set_key_conv else False)

        # 4. make dict_output
        dict_output = {
            'sent': len(df_campaign),
            'open': len(df_open_target),
            'kpi': len(df_event.drop_duplicates()),
            'sent_open_conv': round( len(df_open_target) / len(df_campaign), 4) if len(df_campaign)!=0 else None,
            'open_kpi_conv': round( len(df_event.drop_duplicates()) / len(df_open_target), 4) if len(df_open_target)!=0 else None,
            'sent_kpi_conv': round( len(df_event.drop_duplicates()) / len(df_campaign), 4) if len(df_open_target)!=0 else None
        }
        df_fin_output = pd.DataFrame.from_dict(dict_output, orient='index')

        return df_fin_output, df_event
    
    def evaluate_campaign_result(df_campaign_conv, col_campaign_nm, col_open, col_open_kpi_conv, col_sent_kpi_conv, 
                             dict_weights = {'open':0.6, 'conv':0.2}, conf_level = 0.95):
        dict_conf = {0.95: 1.96}
        df_campaign_conv['total_performance'] = df_campaign_conv[col_open]*dict_weights['open'] +\
                df_campaign_conv[col_open_kpi_conv]*dict_weights['conv'] +\
                df_campaign_conv[col_sent_kpi_conv]*dict_weights['conv']
        
        _avg = df_campaign_conv['total_performance'].mean()
        _std = np.std(df_campaign_conv['total_performance'])
        _conf_int_min =_avg - (dict_conf[conf_level] * (_std / np.sqrt(df_campaign_conv[col_campaign_nm].nunique())))
        _conf_int_max =_avg + (dict_conf[conf_level] * (_std / np.sqrt(df_campaign_conv[col_campaign_nm].nunique())))

        df_campaign_conv['evaluate'] = df_campaign_conv['total_performance'].apply(
                lambda x: 'high' if x >= _conf_int_max else(
                    'mid' if (x >= _conf_int_min) & (x < _conf_int_max) else 'low')
                )
        
        return df_campaign_conv
    

    def vis_open_rate(df_campaign_open, xlabel, ylabel, color_palette, figsize = (6, 4)):
        df_tidy = pd.melt(df_campaign_open, ['index'], var_name='kpi', value_name='rate')
        _ = sns.color_palette(color_palette)
        _ = plt.figure(figsize=figsize)
        _ = sns.barplot(x = 'index', y= 'rate', hue= 'kpi',data = df_tidy)
        _ = plt.xlabel(xlabel)
        _ = plt.axhline(y=df_campaign_open['open_rate'].mean(), color='orange', linestyle='--', linewidth=1.75)
        _ = plt.ylabel(ylabel)
        plt.show();
