
# TadGAN 알고리즘을 이용한 스마트팜 이상치 탐지 및 생산량 예측
- orion-ml 라이브러리를 이용해서 TadGAN 모델로 이상치를 탐지하고, 탐지된 이상치를 기반으로 생산량을 예측합니다.  
- 이상치 탐지는 xinsunadd, xintemp1, xsthum, xco2 4가지 컬럼에 대한 모델이 구현되어있습니다.  
- 생산량 예측은 각 이상치의 총합을 feature로 하여 한 작기의 10a당 생산량을 ridge 회귀모델을 이용하여 예측합니다.

# 데이터
- 경남TP 스마트팜 데이터 이용 2022~2023년 1개 작기
- 딸기(4개 농가), 토마토(5개 농가), 파프리카(1개 농가)
- 분석 DB

# 모델
## 1. 이상치 탐지 모델
- orion-ml 라이브러리의 Tadgan 모델을 사용하여 이상치 탐지
- 시간 집계는 1시간 (3600초)을 기준으로 집계
- 5 epoch 학습
### 데이터셋
- 시계열 데이터셋
- 필수 컬럼 중 xintemp는 xintemp1만 사용 후 나머지는 제, xco2set은 분석에서 제외함
	1. xintemp2의 경우 xintemp1과 매우 높은 상관관계(0.99)가 보이며, 데이터의 스케일이 같으므로 모델링에서 제외함 (Correlation Heatmap.jpeg 참고)<br>
![Correlation Heatmap](https://github.com/SeongminLee812/TadGAN/assets/105956513/94d430dd-ab45-4c36-82fa-800058bee0e7)
		
	2. xco2set의 경우 1/4이상이 0.0으로 기록되어 있으며, 일부 농가에서는 3/4 가량이 0.0이므로 결측치가 너무 많다고 판단되어 모델링에서 제외 (Boxplot.jpeg 참고)<br>
![Boxplot](https://github.com/SeongminLee812/TadGAN/assets/105956513/5b6c7518-b1c0-497e-b458-4e54cd7c9a88)
	
- datetime과 탐지하고자하는 환경데이터로 구성된 데이터셋을 사용<br>
![Pasted image 20230831120026](https://github.com/SeongminLee812/TadGAN/assets/105956513/ec06bb6c-1a34-4b8a-8443-fc23364dec06)

### Returns
- Tadgan 모델은 point anomaly를 탐지하는 것이 아니라, context anomaly를 탐지하므로, 기간과 severity가 출력
- 출력값 형태는 아래와 같음<br>
![returns](https://github.com/SeongminLee812/TadGAN/assets/105956513/e35dae92-5b46-4394-8012-90998376a298)
- TadGAN이 탐지한 이상치 예시
	- 파란 점 : 관측 값
	- 빨간 색칠 부분 : 이상치로 탐지된 기간<br>
![Pasted image 20230831121300](https://github.com/SeongminLee812/TadGAN/assets/105956513/f9769b80-25bc-47a2-a270-d61b44765d94)

## 2. 생산량 예측 모델
- 선형회귀 모델 사용 (Ridge 회귀)
- 학습 데이터셋의 부족 문제로 일반화 성능이 부족하여 alpha값을 많이 주어 예측값이 민감하게 반응하지 않음
### 학습 데이터셋
- 농가별 총 탐지된 이상치의 총 합을 feature로, 10a당 생산량 중 medain으로부터의 편차를 target으로 하여 데이터셋 생성<br>
![Pasted image 20230831115544](https://github.com/SeongminLee812/TadGAN/assets/105956513/a8f417a6-0cb0-4db8-a75c-c1198721296a)
### 학습 프로세스
- 농가별 이상치의 총 합을 feature로 함
- scaling : RobustScaler를 이용하여 median과 IQR을 이용해 robust한 스케일링을 진행
- target : scaled된 median으로부터의 편차를 target으로 하여 학습을 진행
- __paprica의 경우 생산량데이터가 1개로, 회귀식 생성이 불가능해 예측이 불가능__(predict 함수 사용 시 저번 작기 생산량 값을 출력)
### Returns
- 2차원 np.array형태안에 값이 들어있는 형태<br>
![Pasted image 20230831120618](https://github.com/SeongminLee812/TadGAN/assets/105956513/9dbbbd46-cdbc-45a7-a50e-8b56500fe58e)


# 사용법 예시
```python
if __name__ == '__main__':
	farm = Tadgan(str(sys.argv[1]))  
	farm.load_data()  
	farm.load_model()  
	farm.predict()  
```
```bash
python detect.py 'mysb2_1'
```
- main 함수는 사용 형태에 맞게 변경해서 사용
<br><br>

![Pasted image 20230831120659](https://github.com/SeongminLee812/TadGAN/assets/105956513/5390ce97-a7bb-469c-9dc6-e2c1362c24cc)<br>
![Pasted image 20230831120710](https://github.com/SeongminLee812/TadGAN/assets/105956513/88118eb8-b6e7-4ec1-ac81-8693b75a11a5)<br>
![Pasted image 20230831120719](https://github.com/SeongminLee812/TadGAN/assets/105956513/24546b96-ace2-41fd-9159-a1195d751334)<br>


## Release Note
ver1.1 : 2023. 8. 31.
