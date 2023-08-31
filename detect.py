# detect.py

import pymysql 
import pandas as pd
from orion import Orion
import joblib
from tqdm import tqdm
import sys


class Tadgan():
    """
    orion-ml 라이브러리를 이용해서 TadGAN 모델로 이상치를 탐지하고, 탐지된 이상치를 기반으로 생산량을 예측합니다.
    이상치 탐지는 xinsunadd, xintemp1, xsthum, xco2 4가지 컬럼에 대한 모델이 구현되어있습니다.
    생산량 예측은 각 이상치의 총합을 feature로 하여 한 작기의 10a당 생산량을 ridge 회귀모델을 이용하여 예측합니다.

    Authorizer : 골든플래닛 DX엔지니어링 데이터 분석 이성민 매니저
    Release
    ver1.1 : 2023. 8. 31.

    모델 디렉토리 설명
    ---------------
    모델은 2가지로 구성되며, 각 모델별 활용 시점이 다릅니다.
    detect_model : orion-ml 이상치 탐지 모델이 들어있습니다. 작물별, 컬럼별 학습한 모델 총 12개가 들어있습니다. json파일
                    모델 로딩 시간이 다소 소요되어 load_model 함수를 사용하여 loading이 필요합니다.
    predict_model : sklearn 생산량 예측 모델이 들어있습니다. 작물별 모델 총 3개 및 작물별 scaler 총 3개가 들어있습니다. joblib 파일

    `__init__` :
    argument로 zone 정보를 받습니다.
    zone은 분석 db에 저장되어있는 region, area의 값을 합친 것으로 {region}_{area}형식입니다.
    ex) 'mysb4_1' : WHERE region='mysb4' AND area='1'


    Parameters
    ---------------
    zone : 위 `__init__`함수 설명과 같이 zone을 입력받습니다.


    Attributes
    ---------------
    zone : 입력한 zone의 정보를 반환합니다.
    crop : 해당 zone의 농작물 정보를 반환합니다.
    data : sql 쿼리를 이용해서 추출한 해당 zone의 데이터(xdatetime, xinsunadd, xintemp1, xsthum, xco2, region, area)를
                  pd.DataFrame으로 반환합니다.
    season_beginning : 작물별 작기 시작일을 dictionary로 보관합니다. 해당 시작일 이후 데이터만 불러옵니다.


    Example
    ---------------
    ```
    farm = Tadgan('mysb4_1')  # mysb4_1 농가에 대한 Tadgan 인스턴스 생성합니다.
    farm.load_data()          # sql query를 이용하여 해당 농가의 데이터를 불러옵니다.
    farm.load_model()         # 해당 농가의 작물에 해당하는 이상치 탐지 모델(4개)을 불러옵니다.
    farm.detect('xinsunadd')  # parameter로 입력한 xinsunadd의 이상치를 탐지하여 dataframe형태로 반환합니다.
    farm.predict()            # 전체 컬럼 이상치 탐지하여 합을 구하고, 해당 합을 feature로 하여 생산량 예측합니다.
    ```
    """

    zone_dict = {'strawberry': ['mysb6_6', 'mysb6_1', 'mysb4_4', 'mysb6_5'],
                 'tomato': ['mysb4_1', 'mysb4_2', 'mysb4_3', 'mysb2_2', 'mysb6_2'],
                 'paprica': ['mysb2_1']}
    zone_list = ['mysb6_6', 'mysb6_1', 'mysb4_4', 'mysb6_5', 'mysb4_1', 'mysb4_2', 'mysb4_3', 'mysb2_2', 'mysb6_2', 'mysb2_1']
    crop_list = ['strawberry', 'tomato', 'paprica']
    season_beginning = {'stawberry': '2022-09-02', # 작물별 이번 작기 시작일을 입력 -> db에서 추출할 때 참고
                        'tomato': '2022-09-02',
                        'paprica': '2022-09-02'}

    detect_model_path = './detect_model/'      # detecting anomaly model path
    predict_model_path = './predict_model/'      # predict yield output model path
    
    
    def __init__(self, zone=None):
        zone = zone.lower()
        if zone not in Tadgan.zone_list:
            raise Exception(f'Not valid zone. \n Please input one of mysb6_6, mysb6_1, mysb4_4, mysb6_5, mysb4_1, mysb4_2, mysb4_3, mysb2_2, mysb6_2, mysb2_1')
        self.zone = zone
        self.crop = [key for key, value in Tadgan.zone_dict.items() if zone in value][0]
        self.data = pd.DataFrame()
        self.model_dict = dict()
        self.anomalies_sum_df = 0


    def load_data(self):
        """
        db에서 pymysql을 이용해 데이터를 불러옵니다.
        이상치 탐지에 사용되는 컬럼 xdatetime, xinsunadd, xintemp1, xsthum, xco2, region, area만 불러옵니다.
        또한, 클래스 변수 season_beginning의 crop별 작기 시작기간을 기준으로 해당 데이터 이후 일자만 가져옵니다.

        conn변수에 mysql 연결 정보를 입력하여 db를 연결, 데이터를 추출하면 됩니다.

        Returns
        -------
        data : pd.DataFrame, query에 따라 load한 데이터를 pd.DataFrame형식으로 반환하며, 클래스 변수 self.data에 저장합니다.
        """

        query= ''
        region, area = self.zone.split('_')
        query =  f'''
            SELECT xdatetime, xinsunadd, xintemp1, xsthum, xco2, region, area FROM MIRYANG_SENSING_DATA 
            WHERE (region="{region}" AND area="{area}" ) 
            AND xdatetime > {Tadgan.season_beginning[self.crop]}
            '''

        conn = pymysql.connect(host='127.0.0.1', user='root', db='smartfarm', charset='utf8')

        cur = conn.cursor()

        data = pd.read_sql_query(query, conn)
        data = data.sort_values(by='xdatetime')
        data = data.reset_index(drop=True)
        data['zone'] = data['region'] + '_' + data['area']
        data = data.drop(['region', 'area'], axis=1)
        self.data = data
        return data
    
    
    def load_model(self):
        """
        이상치 탐지 모델을 불러오는 함수입니다.
        작물에 해당하는 컬럼별 탐지모델 4개를 불러옵니다.
        인스턴스 변수 self.model_dict에 불러온 컬럼별 모델을 저장합니다.
        """
        model_path = Tadgan.detect_model_path + self.crop
        
        self.model_dict['xinsunadd'] = Orion.load(model_path + '_xinsunadd.json')
        self.model_dict['xsthum'] = Orion.load(model_path + '_xsthum.json')
        self.model_dict['xintemp1'] = Orion.load(model_path + '_xintemp1.json')
        self.model_dict['xco2'] = Orion.load(model_path + '_xco2.json')
        

    def make_train_df(self, df, column):
        """
        orion-ml 모델 인풋 형식에 맞춰 data를 변경합니다.
        모델 인풋 형식은 timestamp와 value로만 구성된 2개 컬럼을 가진 데이터프레임입니다.

        Attribute
        ---------
        df : pd.DataFrame, dataframe을 입력받습니다. 인스턴스의 load_data로 불러온 self.data를 활용하면 됩니다.
        column : str, 이상치를 탐지하고자 하는 컬럼명을 입력합니다.

        Return
        ------
        train_df : pd.DataFrame, timestamp와 value로만 구성된 데이터프레임을 반환합니다.

        """
        train_df = df[['xdatetime', column]].rename(columns={'xdatetime': 'timestamp', column: 'value'})

        # 저장일시 column 을 모델 input 양식(timestamp)으로 변경
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
        train_df['timestamp'] = train_df['timestamp'].apply(lambda x : x.timestamp())

        # input type : integer or float
        train_df['timestamp'] = train_df['timestamp'].astype('int')
        return train_df

    def detect(self, column, period=None):
        """
        이상치를 탐지하여 반환하는 메서드입니다.
        컬럼만 입력하는 경우 load_data로 불러온 전체 데이터에 대해 이상치를 탐지하며,
        period를 argument로 주는 경우 period 내의 이상치만 탐지합니다.

        Parameters
        ---------
        column : str, 이상치를 탐지하고자 하는 컬럼을 입력(xinsunadd, xco2, xintemp1, xsthum 중 1)
        period : str, list or tuple, 탐지하고자 하는 기간을 입력, list나 tuple로 입력 시 첫번째 인자를 시작일, 두번째 인자를 마지막일로 입력합니다.
                 마지막일의 경우 포함되지 않습니다.(00:00까지만 포함됨)
                 string으로 입력 시 해당일을 시작일로 지정하고, 현재일을 마지막일로 자동 지정하여 detect

        Returns
        -------
        column_anomalies : 탐지된 이상치들을 전부 반환하며, start, end, severity 의 컬럼을 가진 데이터 프레임을 반환합니다.
                            이상치는 point anomaly가 아닌 context anomaly로 기간을 포함한 이상치를 반환합니다.
                            start -> 탐지된 이상치의 시작 기간
                            end -> 탐지된 이상치의 끝 기간
                            severity -> 탐지된 이상치의 심각도 (높을 수록 심각한 이상치)
        """
        data = self.data

        if isinstance(period, list) or isinstance(period, tuple):
            start = pd.to_datetime(period[0])
            end = pd.to_datetime(period[1])
            condition = (data['xdatetime'] >= start) & (data['xdatetime'] <= end)
            data = data.loc[condition].reset_index(drop=True)
        elif isinstance(period, str):
            start = pd.to_datetime(period)
            end = pd.to_datetime(datetime.now().date())
            condition = (data['xdatetime'] >= start) & (data['xdatetime'] <= end)
            data = data.loc[condition].reset_index(drop=True)

        model = self.model_dict[column]
        train_df = self.make_train_df(data, column)
        column_anomalies = model.detect(train_df)
        column_anomalies['start'] = pd.to_datetime(column_anomalies['start'], unit='s')
        column_anomalies['end'] = pd.to_datetime(column_anomalies['end'], unit='s')

        print('=====\t', column, 'detected', '\t=====')
        return column_anomalies
    
    
    def detect_all(self):
        """
        모든 컬럼의 이상치를 탐지해 해당 zone의 이상치별 총합을 반환합니다.

        Returns
        -------
        anomaly_sum_df : pd.DataFrame, 인스턴스의 zone을 인덱스로, 이상치 환경 컬럼 4개를 컬럼으로 가지는 데이터프레임입니다.
                        인스턴스의 zone만을 row로 가지는 튜플 하나짜리 데이터 프레임입니다.
                        필드값은 탐지된 이상치의 severity를 전부 더한 값입니다.
                        생산량 predict시 사용되는 데이터셋입니다.

        """
        
        columns_list = ['xinsunadd', 'xintemp1', 'xsthum', 'xco2']
        anomaly_sum_df = pd.DataFrame(index=[self.zone])
        for column in columns_list:
            column_anomaly = self.detect(column)
            anomaly_sum_df.loc[self.zone, column] = column_anomaly['severity'].sum()     
            
        self.anomaly_sum_df = anomaly_sum_df
        
        return anomaly_sum_df
    
    
    def predict(self, anomalies_df=None):
        """
        탐지된 이상치들을 이용하여 생산량을 예측합니다.

        학습 데이터셋 : 농가별 탐지된 이상치의 총 합 값
        학습 모델 : Ridge(alpha = 2) # 데이터가 적어 일반화 성능이 부족하므로 alpha값을 높게 주었는데, 추후 작기가 늘어나면 하이퍼파라미터 조정으로 성능 향상 가능

        Process
        -------
        1. predict
            anomalies_sum_df를 이용해 학습 모델로 생산량을 예측합니다.
        2. scaler
            위 1번으로 예측된 값은 scaling된 값이므로, 학습시 사용한 scaler객체를 불러와 inverse_transform으로 원래 값으로 변환합니다.
        3. median sum
            학습시 사용한 값은 median으로부터의 편차이므로, 예측값 또한 편차입니다. 작물별 median과 더하여 최종예측값을 만들어 냅니다.

        Parameters
        ----------
        anomalies_df : pd.DataFrame, 탐지된 이상치의 총 합이 담긴 데이터 프레임.
                    최초 1회 실행 시에만 detect_all을 실행하고, self.anomalies_sum_df에 할당된 경우 해당 변수를 불러와 사용합니다.

        Returns
        -------
        predicted_value : np.array, 2차원 array형태로 반환됩니다. 값만 출력하고 싶은 경우 인덱싱으로 출력가능합니다.


        Notice
        ------
        paprica 작물의 경우 예측이 불가능합니다.(농가가 1개이므로 학습이 제한되어 coefficient가 전부 0으로 어떤 값을 입력해도 작년 작기 생산량 5705.79만 나옴)

        """

        if anomalies_df == None:
            if not isinstance(self.anomalies_sum_df, pd.DataFrame):
                self.anomalies_sum_df = self.detect_all()
            anomalies_df = self.anomalies_sum_df
            
        model_path = Tadgan.predict_model_path
        
        median_dict = {'tomato': 8069.44,       # median dict : 2022~2023년 작기 생산량의 작물별 median값을 저장한 것이며
                      'paprica': 5705.79,                     # 다음 작기 이후 학습 시 변경될 수 있습니다.
                      'strawberry': 1469.57}
        FEATURE = ['xco2', 'xinsunadd', 'xintemp1', 'xsthum']
        train_df = anomalies_df[FEATURE]
        
        model = joblib.load(model_path + f'{self.crop}_predict_model.joblib')
        scaler = joblib.load(model_path + f'{self.crop}_scaler.joblib')

        pred = model.predict(anomalies_df)    
        if pred.ndim == 1:
            pred = [pred]

        deviation = scaler.inverse_transform(pred)
        predicted_value = median_dict[self.crop] + deviation

        try:
            print('예측되는 생산량은 {}입니다.'.format(predicted_value[0][0]))
        except:
            print(predicted_value)
        return predicted_value
        
    
    
    def __repr__(self):
        return f'''TadGAN Object 
                   crop : {self.crop}
                   zone : {self.zone}
                   season_beginning_day : {Tadgan.season_beginning.get(self.crop)}
                '''
    
    

if __name__ == '__main__':
    print(str(sys.argv[1]))
#    farm = Tadgan(str(sys.argv[1]))
#    farm.load_data()
#    farm.load_model()
#    farm.detect('xinsunadd')
#    farm.predict()
#     print(farm)



