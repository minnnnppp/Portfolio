# 0. import package & library
# basic
import os
import sys
import warnings
warnings.filterwarnings(action='ignore') 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pprint

#--------------------#
# handling
#--------------------#
# import math
import time
import importlib
import numpy as np
import pandas as pd
from urllib import parse
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta

#--------------------#
# Vis
#--------------------#
import matplotlib.pyplot as plt
import seaborn as sns

#--------------------#
# sphere package
#--------------------#
sys.path.append("/home/das_share/sphere_class/")
import SpherePackage
from SpherePackage import *




#================================#
# Attribution Analysis
## calc attribution kpi for attribution analysis
#================================#
class AttributionAnalysis():

    def __init__(self, df_app_log, DEPTH_1_COL, DEPTH_2_COL = None,
                 KEY_ID = 'uid', get_fraud_keyid = 'v1', min_duration = 5,
                 lst_default_event = ['#firstVisit', '#appInstall', '#appUpdate', 'sapHome', '#sessionStart']):
        """
        [input]
            - df_app_log: 로그데이터
            - DEPTH_1_COL : ex) REFERR_COL
            - DEPTH_2_COL : (default = None) ex) REFERR_CAMPAIGN_COL
            - get_fraud_device: (default = 'v1) 
                - v1: 체류시간이 min_time 미만인 기기를 fraud 기기로 간주
                - v2: FraudAnalysis.fraud_main()를 사용해 fraud 기기 필터링
            - min_duration: 최소 체류시간
            - lst_default_event: 기본 default 이벤트 리스트
                - (default: ['#firstVisit', '#appInstall', '#appUpdate', 'sapHome', '#sessionStart'])
        """
        self.key_id = KEY_ID
        self.df_app_log = df_app_log
        self.dates = (max(df_app_log['day']) - min(df_app_log['day'])).days + 1
        self.DEPTH_1_COL = DEPTH_1_COL
        self.DEPTH_2_COL = DEPTH_2_COL
        self.dict_lst_keyid_from_referrs = {}
        self.dict_period_score_output = {}
        self.lst_default_event = lst_default_event
        self.lst_keyid_total = list(df_app_log[KEY_ID].unique())

        if get_fraud_keyid.lower() == 'v1':
            df_duration_per = self.df_app_log.groupby([self.key_id ])['duration'].sum()
            self.lst_keyid_fraud = list(df_duration_per[df_duration_per < min_duration].index)
        else:
            fraud_obj = FraudAnalysis(self.df_app_log, self.key_id)
            self.lst_keyid_fraud = fraud_obj.fraud_main(
                lst_default_event = self.lst_default_event, 
                min_duration = min_duration,
                vis = False, verbose = False)
        self.lst_keyid_fraud_x = list( set(self.lst_keyid_total) - set(self.lst_keyid_fraud))




    #--------------------------------#
    ## get user/device list by referr
    @staticmethod
    def get_user_list_by_referr(df, KEY_ID, DEPTH_1_COL, DEPTH_2_COL = None) :
        """
        : 채널별 유저 출력
        [input]
            - df
            - DEPTH_1_COL : ex) REFERR_COL
            - DEPTH_2_COL : ex) REFERR_CAMPAIGN_COL
        [output]
            - dict_lst_user_from_referrs
                ex) {
                        campaign1: {
                            'GDN': [가, 나, 다, 마, 사, ....],
                            'NAVER': [나, 라, 차, 하]
                        }
                }
        * 일시적 미맵핑으로 referr값이 None으로 발생할 수 있으며, 추후 다른 referr값이 붙는 경우가 있어 organic으로 변환하지 않고 제외함.
        """ 
        # 0. 채널별 앱설치 유저 Segment --> = 매칭유저 (설치자수[매핑])
        ## 1) lst_depth_1
        lst_depth_1 = list(df[DEPTH_1_COL].unique())
        try :
            lst_depth_1.remove(None)
        except :
            pass

        ## 2) lst_user from _depth1
        dict_lst_user_from_referrs = {}
        ### (1) 캠페인 미구분(매체만 구분)
        if DEPTH_2_COL == None :
            for _depth1 in lst_depth_1 :
                dict_lst_user_from_referrs[_depth1] =\
                    list(df[df[DEPTH_1_COL] == _depth1][KEY_ID].unique())
        else : 
        ### (2) 캠페인  & 매체 모두 구분
            for _depth1 in lst_depth_1 :
                ### prep
                df_referr = df[df[DEPTH_1_COL] == _depth1]
                df_referr[DEPTH_2_COL] = df_referr[DEPTH_2_COL].fillna('')
                lst_depth_2 = df_referr[DEPTH_2_COL].unique()
                
                ### extract each _depth2 users
                dict_lst_user_from_referrs[_depth1] = {}
                for _depth2 in lst_depth_2 :
                    dict_lst_user_from_referrs[_depth1][_depth2] =\
                        list(df_referr[df_referr[DEPTH_2_COL] == _depth2][KEY_ID].unique())
                    
        return dict_lst_user_from_referrs
    
    
    
    
    #--------------------------------#
    ## standardization
    @staticmethod
    def prep_scaling(df, lst_col_reverse = None):
        lst_target_col = [x for x in df.columns if (df[x].dtype==int)|(df[x].dtype==float)]
        scaler = StandardScaler()
        array_scaled = scaler.fit_transform(df[lst_target_col])
        df_fin_scaled = pd.DataFrame(array_scaled, index = df.index, columns = lst_target_col)
        if lst_col_reverse!=None:
            df_fin_scaled[lst_col_reverse] = df_fin_scaled[lst_col_reverse] * -1
        
        return df_fin_scaled




    #--------------------------------#
    # inflow: count total/daily inflow
    def get_inflow_kpi(self, period_referr_session, period = 'day', dict_costs = None):
        """
        : 매체별 기간별/전체 유입 및 평균 유입 집계
        [input]
            - dict_costs_by_referr: (default = None) 매체별 광고 비용 딕셔너리 -> 주어지지 않는 경우에는 None
            - period_referr_session: 일별 유입수 집계 방식(str) --> 'referr' or 'session'
                - 'referr': 매체(카카오, 네이버 등)별로 유입한 기기가 특정 기간동안 얼만큼 유입되었는지를 집계하는 방법
                - 'session': 각 세션마다 달고 들어오는 referr 값에서 어떤 채널인지를 확인하여 매체별로 유입된 기기 수를 집계
            - period: 집계 기준이 되는 기간 컬럼명      
                - ex) 'day', 'week', 'month'
        [output]
            - df_total_output: 매체별 전체/평균 유입 수를 담는 딕셔너리
                - ex)
                                        |   total_inflow
                            GDN         |       99389
                            NAVER_GFA   |       49381
            - df_period_inflow: 매체별 일별 유입 수를 집계한 dataframe 딕셔너리
                - ex) 
                                날짜     |   total  |   GDN
                            2022-10-22  |   2939   |   243
                            2022-10-23  |   3984   |   138
        """
        dict_total_output= {}
        dict_average_costs = {}

        # 0. prep about period
        ## period == 'week'인 경우 week number -> 주차의 시작일로 변경
        if period.lower() == 'week':
            if not ('week' in self.df_app_log.columns) : 
                self.df_app_log['week'] = SpherePrep.BasicPrep.convert_week_n_to_day(self.df_app_log)
        ## period = 'month'인 경우 month 컬럼 생성
        if period.lower() == 'month':
            if not ('month' in self.df_app_log.columns) : 
                self.df_app_log['month'] = self.df_app_log['date'].dt.month

        # 1. total inflow
        df_period_inflow = self.df_app_log.groupby(period)[self.key_id].nunique().to_frame(name = 'total_inflow')
        df_period_score_costs = pd.DataFrame(index = list(df_period_inflow.index))

        # 2. total & average inflow/inflow costs by referrs
        for _referr, _segment in self.dict_lst_keyid_from_referrs.items():
            dict_total_output[_referr] = {}
            dict_average_costs[_referr] = {}

            ## 1) total inflow & average inflow
            _average_inflow = round(len(set(_segment)) / self.dates, 4)
            dict_total_output[_referr] = [len(set(_segment)), _average_inflow]

            ## 2) daily/weekly/monthly inflow
            ### A. referr
            if period_referr_session.lower() == 'referr':
                _df_segment = self.df_app_log[self.df_app_log[self.key_id].isin(_segment)]
            ### B. session
            else:
                _df_segment = self.df_app_log[self.df_app_log['utm_source']==_referr]
            df_period_inflow[f'{_referr}_inflow'] = _df_segment.groupby(period)[self.key_id].nunique().to_frame()

            ## 3) average inflow costs
            if dict_costs != None:
                _average_costs = dict_costs[_referr] / self.dates
                dict_average_costs[_referr] = round( _average_costs / _average_inflow, 2)
                ### ++ additional: avg inflow costs per period
                df_period_score_costs[f'{_referr}'] = round(_average_costs / df_period_inflow[f'{_referr}_inflow'], 2)

        # 3. make dataframe
        ## 1) total_output
        df_total_output = pd.DataFrame.from_dict(
            dict_total_output, orient='index', 
            columns=['total_inflow', 'daily_avg_inflow']).sort_index(ascending=True)
        if dict_costs != None:        
            df_total_output['avg_inflow_costs'] = pd.DataFrame.from_dict(
                dict_average_costs, orient='index', 
                columns = ['avg_inflow_costs']).sort_index(ascending = True)
        ## ++ 추가 2) period output for using at the part of score
        self.dict_period_score_output['inflow'] = pd.melt(
            df_period_inflow.iloc[:,1:4].T.reset_index(), 
            id_vars = 'index', value_vars = list(df_period_inflow.iloc[:,1:4].T.columns),
            var_name = 'day', value_name = 'inflow')

        # 4. prep
        df_total_output = df_total_output.fillna(0)          
        df_period_inflow = df_period_inflow.fillna(0)      
        df_period_inflow = df_period_inflow.replace(np.inf, np.nan)
        
        return df_total_output, df_period_inflow




    #--------------------------------#
    # revisit: revisit rate
    def get_revisit_rate(self, revisit_over_n = 3):
        """
        : 재방문율 집계
        [input]
            - revisit_over_n: (default = 3) n회 이상 재방문한 기기 추출하기 위한 파라미터, n > 2 이상일 것
        [output]
            - df_revisit_output: 1회 방문/2회 이상 방문/n회 이상 방문 기기 수 dataframe
        """
        dict_output = {}
        dict_df_visit_day = {}

        for _referr, _segment in self.dict_lst_keyid_from_referrs.items():
            dict_output[_referr] = {}
            _df_target = self.df_app_log[self.df_app_log[self.key_id].isin(_segment)]
            dict_df_visit_day[_referr] = _df_target.groupby(self.key_id)['day'].count().to_frame(name = 'count')

            # device by times of visit: once/more than once/over three times
            _visit_1_cnt = dict_df_visit_day[_referr].value_counts()[1].values[0]
            _only_1_visit_rate = round(_visit_1_cnt / len(_segment), 4)
            _re_visit_rate = round(1 - _only_1_visit_rate, 4)
            _re_visit_over_n_rate =\
                round(1 - dict_df_visit_day[_referr].value_counts()[:revisit_over_n-1].sum() / len(_segment), 4)
            
            dict_output[_referr]['revisit_only_1_rate'] = _only_1_visit_rate
            dict_output[_referr]['revisit_over_2_rate'] = _re_visit_rate
            dict_output[_referr][f'revisit_over_{revisit_over_n}_rate'] = _re_visit_over_n_rate

        # convert to dataframe
        df_revisit_output = pd.DataFrame.from_dict(dict_output, orient = 'index')
        
        return df_revisit_output




    #--------------------------------#
    # fraud: count fraud user/device
    def get_fraud_rate(self, dict_costs = None):
        """
        : 매체별 이상 유저 수 집계
        [output]
            - df_fraud_output: 매체별 bouce/정착 기기 수, bouce/정착 rate
        """
        dict_fraud = {}

        for _referr, _lst_segment in self.dict_lst_keyid_from_referrs.items():
            ## 1) total fraud
            _lst_fraud = list(set(_lst_segment) & set(self.lst_keyid_fraud))       ## 수정된 부분
            fraud_cnt = len(_lst_fraud)             # = bounce rate
            settle_cnt = len(_lst_segment) - fraud_cnt
            fraud_rate = fraud_cnt / len(_lst_segment)
            settle_rate = 1 - fraud_rate            
            dict_fraud[f'{_referr}'] = [fraud_cnt, settle_cnt, fraud_rate, settle_rate]

            ## ++ average costs of settle user
            if dict_costs!=None:
                _avg_costs = dict_costs[_referr] / self.dates
                _avg_settle_costs = round(_avg_costs / settle_cnt, 2)
                dict_fraud[f'{_referr}'] = [fraud_cnt, settle_cnt, fraud_rate, settle_rate, _avg_settle_costs]
                lst_col_output = ['bounce_cnt', 'settle_cnt', 'bounce_rate', 'settle_rate', 'avg_settle_costs']
            else:
                dict_fraud[f'{_referr}'] = [fraud_cnt, settle_cnt, fraud_rate, settle_rate]
                lst_col_output=['bounce_cnt', 'settle_cnt', 'bounce_rate', 'settle_rate']

        df_fraud_output = pd.DataFrame.from_dict(dict_fraud, orient='index',  columns=lst_col_output)
        df_fraud_output = df_fraud_output.sort_values(by = 'bounce_rate', ascending = False)
                    
        return df_fraud_output




    #--------------------------------#
    # conversion: calculate conversion
    def get_kpi_conversion(self, dates, event_conv, period = 'day', dict_param = None,dict_costs = None):
        """
        : 광고를 통한 랜딩에서 목표 kpi로 얼만큼 전환되는지 전환율 집계
        [inpupt]
            - event_conv: 목표 KPI, 이벤트명
            - period: 집계 기준이 되는 기간 컬럼명            
                - ex) 'day', 'week', 'month'
            - dict_costs: (default = None) 매체별 비용이 들어있는 딕셔너리
        [output]
            - df_conv_output: 매체별 랜딩/전환 기기 수 + 전환율
            - df_period_conv_output: 매체별 일별 전환 이벤트 기기 수
        """
        # 0. setting
        ## 0) make df_conv & df_output
        if dict_param == None:
            self.df_conv = self.df_app_log[self.df_app_log['abs_events'].apply(lambda x : True if event_conv in x else False)]
        else:
            param_nm = list(dict_param.keys())[0]
            _, df_param_conv = DataImport.json_to_dataframe_nodeN(             
                self.df_app_log, ['uid', 'user_id'], type = 'manual', lst_manual_events = [event_conv])
            df_param_conv['date'] = df_param_conv['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            df_param_conv['day'] = df_param_conv['date'].apply(lambda x: x.date())
            self.df_conv = df_param_conv[df_param_conv[param_nm].isin(dict_param[param_nm])]
        ## 1) prep about period
        ### A. period == 'week'인 경우 week number -> 주차의 시작일로 변경
        if period.lower() == 'week':
            if not ('week' in self.df_conv.columns) : 
                self.df_conv['week'] = SpherePrep.BasicPrep.convert_week_n_to_day(self.df_conv)
        ### B. period = 'month'인 경우 month 컬럼 생성
        if period.lower() == 'month':
            if not ('month' in self.df_conv.columns) : 
                self.df_conv['month'] = self.df_conv['date'].dt.month
        ## 2) pre-setting output & segment
        df_output = pd.DataFrame(index=['conversion_cnt', 'conversion_rate'])
        set_conv_device = set(self.df_conv[self.key_id].unique())
        ## 3) make dictionary if dict_costs != None
        if dict_costs != None: dict_average_costs = {}

        # 1. calc device converted daily/weekly/monthly
        df_period_conv_output = self.df_conv.groupby(period)[self.key_id].nunique().to_frame(name = 'total_conv')

        # 2. calc indicators
        for _referr, _segment in self.dict_lst_keyid_from_referrs.items():
            ## (1) landing & conv cnt & conv rate
            _landing_cnt = len(set(_segment))
            _conv_cnt = len(set_conv_device & set(_segment))
            _conv_rate = round(_conv_cnt/_landing_cnt, 4)
            df_output[f'{_referr}'] = [_conv_cnt, _conv_rate]
            
            ## (2) device converted daily/weekly/monthly by each referr
            _lst_target = list(set_conv_device &  set(_segment))
            _df_segment = self.df_conv[self.df_conv[self.key_id].isin(_lst_target)]
            df_period_conv_output[f'{_referr}_conv'] = \
                _df_segment.groupby(period)[self.key_id].nunique().to_frame()
            
            ## (3) average daily conversion costs by referr
            if dict_costs != None:
                dict_average_costs[_referr] = {}
                try :
                    dict_average_costs[_referr] = round((dict_costs[_referr]/dates) / (_conv_cnt/dates), 2)
                except ZeroDivisionError:
                    dict_average_costs[_referr] = np.NaN
                ### ++ 추가: 기간별 평균 전환비용 집계
                df_period_conv_output[f'{_referr}_conv_costs'] = round(
                    (dict_costs[_referr]/dates) / df_period_conv_output[f'{_referr}_conv'], 2)

        # 4. df_output
        df_period_conv_output = df_period_conv_output.fillna(0)
        df_period_conv_output = df_period_conv_output.replace(np.inf, np.nan)
        df_conv_output = df_output.T   
        if dict_costs != None:
            df_conv_output['avg_conversion_costs'] = pd.DataFrame.from_dict(
                dict_average_costs, orient='index')

        return df_conv_output, df_period_conv_output




    #--------------------------------#
    # activation: calculate average duration & event cnt
    def get_activation_kpi(self, remove_fraud = True):
        """
        : 매체별 활성 지표 집계
        [inpupt]
            - remove_fraud: (default = True) 이상 유저 제거 유무 --> True: 제거 / False: 제거 x
        [output]
            - df_active_output: 매체별 인당 체류시간, 인당 컨텐츠 조회수  
        """
        dict_duration = {}
        dict_event_cnt = {}
        df_period_duration = pd.DataFrame()
        
        # 1. get result of activation
        for _referr, _segment in self.dict_lst_keyid_from_referrs.items():
            ## 0) prep: whether get rid of bounce device
            if remove_fraud == True: 
                _segment = list(set(self.lst_keyid_fraud_x) & set(_segment))
            _df_target_total = self.df_app_log[self.df_app_log[self.key_id].isin(_segment)]
            ## 1) duration
            _df_duration_sum =  _df_target_total.groupby(self.key_id)['duration'].sum()
            ## 2) event
            _df_target_total['events_cnt'] = _df_target_total['abs_events'].apply(lambda x : len(x))
            _df_events_sum = _df_target_total.groupby(self.key_id).agg({'events_cnt':'sum'})
            ## 3) average duration & event_cnt
            dict_duration[f'{_referr}']= round(_df_duration_sum.mean() / self.dates, 4)
            dict_event_cnt[f'{_referr}'] = round(_df_events_sum.mean().values[0], 4)

        # 2. convert to dataframe
        df_active_output =\
            pd.DataFrame.from_dict(dict_duration, orient='index', columns=['daily_avg_duration'])
        df_active_output ['avg_event_cnt'] =\
            pd.DataFrame.from_dict(dict_event_cnt, orient='index',  columns=['avg_event_cnt'])

        return df_active_output
    



    #--------------------------------#
    ## merge all kpi output in a dataframe
    def merge_all_output(self, dict_fin_total_output):
        """
        : 모든 output을 하나의 dataframe으로 merge하는 함수
        [output]
            - df_fin_output: 모든 지표 하나로 merge한 dataframe
        """
        self.df_fin_output = pd.DataFrame()
        for _df in dict_fin_total_output.values():
            _df = _df.sort_index(ascending=True) 
            self.df_fin_output = pd.merge(self.df_fin_output, _df, left_index=True, right_index=True, how='outer')
        
        return self.df_fin_output




    #--------------------------------#
    ## average: calc average of each kpi
    def get_average_kpi(self, event_conv, index_nm, remove_fraud = True):
        """
        : 주요 kpi 평균값 반환하는 함수
        [input]
            - event_conv: 전환 이벤트
            - index_nm: (default: 'total_referr') index명
            - remove_fraud: (default: True) 이상 유저 제거 유무 (True: 제거 / False: 제거 x)
        [output]
            - df_average_kpi: kpi별 평균값 dataframe
        """
        # df_average_kpi = pd.DataFrame()
        lst_keyid_total_referr = []
        for _segment in self.dict_lst_keyid_from_referrs.values():
            lst_keyid_total_referr = list(set(lst_keyid_total_referr + _segment))

        # 1. get average value of kpi ranked
        ## (1) average daily inflow cnt
        df_app_log_target= self.df_app_log[self.df_app_log[self.key_id].isin(lst_keyid_total_referr)]
        cnt_keyid_total_referr = df_app_log_target[self.key_id].nunique()
        
        ## (2) bounce rate
        lst_fraud_total = list(set(lst_keyid_total_referr) & set(self.lst_keyid_fraud))
        average_fraud_rate = round(len(lst_fraud_total) / cnt_keyid_total_referr, 4)
        
        ## (3) conversion
        df_conv= self.df_app_log[self.df_app_log['abs_events'].apply(lambda x: True if event_conv in x else False)]
        lst_keyid_conv = list(df_conv[self.key_id].unique())
        cnt_keyid_landing_referr = len(lst_keyid_total_referr)
        cnt_keyid_conv_referr = len(set(lst_keyid_conv) & set(lst_keyid_total_referr))
        
        ## (4) duration
        if remove_fraud == True:
            df_target = df_app_log_target[df_app_log_target[self.key_id].isin(self.lst_keyid_fraud_x)]
        else: 
            df_target = df_app_log_target
        avg_duration_per = df_target.groupby(self.key_id).agg({'duration':'sum'}).mean().values[0]

        # 2. add avg values to dataframe
        self.df_fin_output.loc[index_nm,'daily_avg_inflow'] = round(cnt_keyid_total_referr / self.dates, 4)
        self.df_fin_output.loc[index_nm,'bounce_rate']= average_fraud_rate
        self.df_fin_output.loc[index_nm,'conversion_rate']= round(cnt_keyid_conv_referr/ cnt_keyid_landing_referr, 4)
        self.df_fin_output.loc[index_nm,'daily_avg_duration'] = round(avg_duration_per / self.dates, 2)

        return self.df_fin_output



    #--------------------------------#
    ## make total/period statistics
    def attribution_analysis_stat(
            self, period_referr_session, event_conv, 
            campaign_nm = None, period = 'day', revisit_over_n = 3, avg_idx_nm = 'total_referr', 
            remove_fraud = True, dict_conv_param = None, dict_cost_by_referr = None):
        """
        [input]
            - period_referr_session: 기간별 유입수 집계 방식(str) --> 'referr' or 'session'
                - 'referr': 매체(카카오, 네이버 등)별로 유입한 기기가 특정 기간동안 얼만큼 유입되었는지를 집계하는 방법
                - 'session': 각 세션마다 달고 들어오는 referr 값에서 어떤 채널인지를 확인하여 매체별로 유입된 기기 수를 집계
            - event_conv: 전환 이벤트명
            - dict_conv_param: (default = None) 전환을 파라미터까지 살펴볼 경우 해당 파라미터명과 타겟 파라미터 값을 리스트로 받음
                - ex) {PARAM_CONTENT_NAME: ['무릉도원']}
            - campaign_nm: (default = None) 광고 utm_campaign명
            - period: 집계 기준이 되는 기간 컬럼명      ex) 'day', 'week', 'month'
            - revisit_over_n: (default = 3) 재방문 횟수가 n회 이상인 keyid를 집계하기 위한 n값
            - avg_idx_nm: (default = 'total_referr) 평균값의 index 값, str
            - remove_fraud: (default = True) 활성 지표 집계 시 fraud 기기 포함 여부, True = fraud 기기 제외 후 집계
            - dict_costs_by_referr: (default = None) 매체별 광고 비용 딕셔너리 -> 주어지지 않는 경우에는 None
        [output]
            - df_total_output: 모든 kpi 분석 지표를 하나로 합친 dataframe
                - ex) 
                            매체   |  total_inflow    | conversion_rate  |
                        FB_INSTA  |      87324       |      0.0252      |
            - df_period_output: 기간별 유입수&전환수를 보여주는 dataframe
                - ex) 
                            day    | total_inflow | GDN_inflow | ... | total_conv |
                        2023-01-01 |    38270     |     387    | ... |     3762
        """
        dict_output = {}
        dict_period_output = {}

        # 0. prep
        dict_user_lst_referr_depth2 = self.get_user_list_by_referr(
            df = self.df_app_log, KEY_ID = self.key_id, DEPTH_1_COL = self.DEPTH_1_COL, DEPTH_2_COL = self.DEPTH_2_COL)
        self.dict_lst_keyid_from_referrs = dict_user_lst_referr_depth2[campaign_nm]

        # 1. get kpi result
        ## 1) inflow
        dict_output['inflow'], dict_period_output['inflow'] = self.get_inflow_kpi(
            period_referr_session = period_referr_session, period = period,
            dict_costs = dict_cost_by_referr)
        ## 2) revisit
        dict_output['revisit'] = self.get_revisit_rate(
            revisit_over_n= revisit_over_n)
        ## 3) fraud
        dict_output['fraud'] = self.get_fraud_rate(dict_costs=dict_cost_by_referr)
        ## 4) conversion
        dict_output['conversion'], dict_period_output['conversion'] = self.get_kpi_conversion(
            dates = self.dates, period = period, event_conv = event_conv, 
            dict_param =dict_conv_param, dict_costs=dict_cost_by_referr)
        ## 5) conversion
        dict_output['activation']= self.get_activation_kpi(remove_fraud=remove_fraud)
        
        # 2. merge
        df_total_output = self.merge_all_output(
            dict_fin_total_output = dict_output)
        df_period_output = pd.concat([dict_period_output['inflow'],
                                      dict_period_output['conversion']], axis = 1)
        
        # 3. get average output
        df_total_output = self.get_average_kpi(
            index_nm = avg_idx_nm, event_conv = event_conv, remove_fraud = remove_fraud)
        
        return df_total_output, df_period_output




    #--------------------------------#
    ## make period data for PCA
    def get_period_pca_output(self, period_referr_session, period = 'day', dict_costs = None):
        # 0. prep about period
        lst_df_period_res = []
        ## prep: period
        ### A. 'week'인 경우 week number -> 주차의 시작일로 변경
        if period.lower() == 'week':
            if not ('week' in self.df_app_log.columns) : 
                self.df_app_log['week'] = SpherePrep.BasicPrep.convert_week_n_to_day(self.df_app_log)
            if not ('week' in self.df_conv.columns) : 
                self.df_conv['week'] = SpherePrep.BasicPrep.convert_week_n_to_day(self.df_conv)
        ### B. period = 'month'인 경우 month 컬럼 생성
        if period.lower() == 'month':
            if not ('month' in self.df_app_log.columns) : 
                self.df_app_log['month'] = self.df_app_log['date'].dt.month
            if not ('month' in self.df_conv.columns) : 
                self.df_conv['month'] = self.df_conv['date'].dt.month

        # 1. total & average inflow/inflow costs by referrs
        for _referr, _segment in self.dict_lst_keyid_from_referrs.items():
            ## 1) daily/weekly/monthly inflow
            ### A. referr
            if period_referr_session.lower() == 'referr':
                _df_segment = self.df_app_log[self.df_app_log[self.key_id].isin(_segment)]
            ### B. session
            else:
                _df_segment = self.df_app_log[self.df_app_log['utm_source']==_referr]
            _df_kpi_res = _df_segment.groupby(period)[self.key_id].nunique().to_frame(name = 'total_inflow')
            _df_kpi_res['bounce_cnt'] = _df_segment[
                    _df_segment[self.key_id].isin(self.lst_keyid_fraud)
                ].groupby(period)[self.key_id].nunique().to_frame()
            _df_kpi_res['referr'] = _referr
            _df_kpi_res = _df_kpi_res[['referr', 'total_inflow', 'bounce_cnt']]
        
            ## 2) period conv/landing
            _df_conv_target = self.df_conv[self.df_conv[self.key_id].isin(_segment)]
            _df_kpi_res['conversion_cnt'] = _df_conv_target.groupby(period)[self.key_id].nunique()
            _df_kpi_res['conversion_cnt'] = _df_kpi_res['conversion_cnt'].fillna(0)

            ## 3) average duration & event cnt
            _df_segment['events_cnt'] = _df_segment['abs_events'].apply(lambda x : len(x))
            _df_res = _df_segment.groupby(period).agg({'duration':'sum', self.key_id:'nunique'})
            _df_res2 = _df_segment.groupby(period).agg({'events_cnt':'sum', self.key_id:'nunique'})
            _df_kpi_res['daily_avg_duration'] = round(_df_res['duration']/_df_res[self.key_id], 2)
            _df_kpi_res['avg_event_cnt'] = round(_df_res2['events_cnt']/_df_res2[self.key_id], 4)

            ## 4) additioanl: period costs inflow/conv
            if dict_costs != None:
                _average_costs = dict_costs[_referr] / self.dates
                _df_kpi_res['avg_inflow_costs'] = round(_average_costs / _df_kpi_res['total_inflow'], 2)

            lst_df_period_res.append(_df_kpi_res)

        df_period_res = pd.concat(lst_df_period_res, axis = 0)
        df_period_res['bounce_rate'] = round(df_period_res['bounce_cnt']/df_period_res['total_inflow'], 4)
        df_period_res['conversion_rate'] = round(df_period_res['conversion_cnt']/df_period_res['total_inflow'], 4)

        return df_period_res



    #--------------------------------#
    ## final score: calc score by kpi
    def get_score_by_referr(self, df_total, dict_kpi_category, dict_kpi_weight, 
                            lst_referr_drop, lst_index_drop = [], df_period = None):
        ## 0. setting
        if type(lst_referr_drop) != list: lst_referr_drop = [lst_referr_drop]
        if type(lst_index_drop) != list: lst_index_drop = [lst_index_drop]

        ## 1. standardization
        lst_col_reverse_total = [x for x in df_total.columns if ('bounce' in x)|('costs' in x)]
        df_total_prep = df_total.loc[[x for x in df_total.index if x not in lst_index_drop + lst_referr_drop]]
        df_scaled_total = self.prep_scaling(df = df_total_prep, lst_col_reverse = lst_col_reverse_total)

        ### ++ if df_period is given, run pca
        #### 1) standardization
        if df_period.empty == False:
            lst_col_reverse_period = [x for x in df_period.columns if ('bounce' in x)|('costs' in x)]
            df_period_res_prep = df_period[~df_period['referr'].isin(lst_referr_drop)]
            df_scaled_period = self.prep_scaling(df = df_period_res_prep, lst_col_reverse = lst_col_reverse_period)

            #### 2) PCA
            df_pca_output = pd.DataFrame()
            pca = PCA(n_components=1, random_state=10)
            for _k, _lst_col in dict_kpi_category.items():
                _ = pca.fit(df_scaled_period[_lst_col])
                _df_pca = pd.DataFrame(pca.transform(df_scaled_total[_lst_col]),
                                    index = df_scaled_total.index, columns= [f'{_k}'])
                df_pca_output = pd.concat([df_pca_output, _df_pca], axis=1)
                df_res = df_pca_output
            
            ## 2. consider weight of kpi
            df_res_cp = df_res.copy()
            for _k in dict_kpi_weight.keys():
                    df_res_cp[_k] = df_res[_k].apply(lambda x : x*dict_kpi_weight[_k])
                    
        else: 
            df_res = df_scaled_total

            ## 2. consider weight of kpi
            df_res_cp = df_res.copy()
            for _k in dict_kpi_weight.keys():
                for _c in dict_kpi_category[_k]:
                    df_res_cp[_c] = df_res[_c].apply(lambda x : x*dict_kpi_weight[_k])

        ## 3. scoring: if the num of referr is more than 2, give score
        if len(df_res_cp.index)>2:
            lst_score = [0.5, 1, 1.5, 2, 2.5 , 3, 3.5, 4, 4.5, 5]
            df_fin_score = pd.DataFrame(index = df_res_cp.index, columns = df_res_cp.columns)
            for _kpi in df_res_cp.columns:
                df_fin_score[_kpi] = pd.cut(df_res_cp[_kpi], bins = 10, 
                    labels=[0.5, 1, 1.5, 2, 2.5 , 3, 3.5, 4, 4.5, 5], right=True) #라벨 설정
                df_fin_score[_kpi] = df_fin_score[_kpi].astype(float)
            for _referr in df_fin_score.index:
                df_fin_score.loc[_referr, 'score'] = round(df_fin_score.loc[_referr].mean(), 4)
        ## if the num of referr is 2, just return pca output
        else: 
            df_fin_score = df_res_cp

        return df_fin_score




    #--------------------------------#
    ## get score by referrs
    def attribution_analysis_score(
            self, df_total_output, period_referr_session, 
            period = 'day', avg_idx_nm = 'total_referr', pca_boolean = False, 
            lst_referr_drop = [], dict_cost_by_referr = None, 
            dict_kpi_weight = {'inflow': 0.4, 'conversion': 0.3, 'settle': 0.2, 'activation': 0.1}):
        """
        [input]
            - df_total_output: gen_attribution_analysis_stat() output 중 전체 기간 output 
            - period: 집계 기준이 되는 기간 컬럼명      ex) 'day', 'week', 'month'
            - period_referr_session: 기간별 유입수 집계 방식(str) --> 'referr' or 'session'
                - 'referr': 매체(카카오, 네이버 등)별로 유입한 기기가 특정 기간동안 얼만큼 유입되었는지를 집계하는 방법
                - 'session': 각 세션마다 달고 들어오는 referr 값에서 어떤 채널인지를 확인하여 매체별로 유입된 기기 수를 집계
            - event_conv: 전환 이벤트명
            - period: 집계 기준이 되는 기간 컬럼명      ex) 'day', 'week', 'month'
            - avg_idx_nm: (default = 'total_referr) 평균값의 index 값, str
            - lst_referr_drop: (default = []) 타겟 캠페인의 utm_source로 들어오지만 광고 성과에서 제외할 매체 이름 리스트
            - dict_costs_by_referr: (default = None) 매체별 광고 비용 딕셔너리 -> 주어지지 않는 경우에는 None
            - dict_kpi_weights: key = kpi명, value = 가중치
                - default = {'inflow': 0.4, 'conversion': 0.3, 'settle': 0.2, 'activation': 0.1}
        [output]
            - df_fin_score: kpi별 성과 점수와 종합 점수를 집계한 output으로, kpi별 점수 구간은 0~4점
                - 종합 점수 = score: kpi별 점수의 가중평균값
                - ex) 
                          매체   |  inflow  |   settle   |  ....   |  score
                       FB_INSTA |     2    |     3      |  ....   |  3.124
            - dict_kpi_category: kpi별 지표 리스트를 담은 딕셔너리로, key = kpi, value = 지표 리스트
        """
        # 1. get pca data(period)
        if pca_boolean==True:
            df_period_pca = self.get_period_pca_output(
                period=period, period_referr_session = period_referr_session, dict_costs=dict_cost_by_referr)
            df_period_pca = df_period_pca.reset_index()

            ## get dictionary of category of kpi
            dict_kpi_category = {'inflow':[], 'settle':[], 'conversion':[], 'activation':[]}
            for _col in df_period_pca.columns:
                if ('settle' in _col) or ('bounce' in _col): 
                    _cate = 'settle'
                elif 'conv' in _col:
                    _cate = 'conversion'
                elif ('duration' in _col) | ('event' in _col):
                    _cate = 'activation'
                elif 'inflow' in _col:
                    _cate = 'inflow'
                else: 
                    _cate = None
                ## add columns based on kpi
                if _cate != None: 
                    dict_kpi_category[_cate].append(_col)
        else:
            lst_col_drop = [x for x in df_total_output.columns if ('bounce' in x)|('only_1' in x)]
            df_total_output = df_total_output.drop(columns = lst_col_drop)
            
            dict_kpi_category = {'inflow':[], 'settle':[], 'conversion':[], 'activation':[]}
            for _col in df_total_output.columns:
                ## reset kpi(=key) for dictionary
                if ('revisit' in _col) or ('bounce' in _col) or ('settle' in _col): 
                    _cate = 'settle'
                elif 'landing' in _col:
                    _cate = 'conversion'
                elif ('duration' in _col) | ('event' in _col):
                    _cate = 'activation'
                else:
                    _cate = list( set(_col.split('_')) & set((dict_kpi_category.keys())) )[0]
                ## add columns based on kpi
                dict_kpi_category[_cate].append(_col)

                ## setting df_pca = None
                df_period_pca = pd.DataFrame()

        # 5. calc total score of kpi
        df_fin_score = self.get_score_by_referr(
            df_period = df_period_pca, df_total = df_total_output, 
            dict_kpi_category = dict_kpi_category, dict_kpi_weight = dict_kpi_weight, 
            lst_referr_drop = lst_referr_drop, lst_index_drop = [avg_idx_nm])

        return df_fin_score



