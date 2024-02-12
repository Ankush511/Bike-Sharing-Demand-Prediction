import pickle
import os
from datetime import datetime
import pandas as pd
import numpy as np

class Inference:
	def __init__(self, model_path, sc_path):
		self.model_path = model_path
		self.sc_path = sc_path
			
		if os.path.exists(self.model_path) and os.path.exists(self.sc_path):
			self.model = pickle.load(open(self.model_path, 'rb'))
			self.sc = pickle.load(open(self.sc_path, 'rb'))
		else:
			print('Model or Standard Scaler path is not correct!')
					
				
	def get_string_to_datetime(self, date):
		dt = datetime.strptime(date, '%d/%m/%Y')		
		return {'day': dt.day, 'month': dt.month, 'year': dt.year, 'week_day': dt.strftime('%A')}
		
		
	def seasons_to_df(self, Seasons):
		seasons_cols = ['Spring', 'Summer', 'Winter']
		seasons_data = np.zeros((1, len(seasons_cols)), dtype = 'int')

		df_seasons = pd.DataFrame(seasons_data, columns=seasons_cols)

		if Seasons in seasons_cols:
			df_seasons[Seasons] = 1
		return df_seasons
		
		
	def days_to_df(self, week_day):
		days_name = ['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
		days_name_data = np.zeros((1,len(days_name)), dtype='int')
				
		df_days = pd.DataFrame(days_name_data, columns=days_name)
				
		if week_day in days_name:
			df_days[week_day] = 1
		return df_days
			
		
				
	def users_input(self,):
		print('Enter correct information to predict Rented Bike Count for a day w.r.t time :')
				
		Date = input('Enter Date (DD/MM/YYYY) :')
		Hour = int(input('Enter Hours (24hrs format) :'))
		Temperature = float(input('Enter Temperature (°C) :'))
		Humidity = float(input('Enter Humidity (%) :'))
		Wind_Speed = float(input('Enter Wind Speed (m/s) :'))
		Visibility = float(input('Enter Visibility (%) :'))
		Solar_Radiation = float(input('Enter Solar Radiation :'))
		Rainfall = float(input('Enter Rainfall :'))
		Snowfall = float(input('Enter Snowfall :'))
		Seasons = input('Enter Season (Spring,Summer,Autumn,Winter) :')
		Holiday = input('Enter Holiday/No Holiday : ')
		Functioning_Day = input('Enter Functioning Day (Yes/No) :')
				
		holiday_dict = {'No Holiday': 0 ,'Holiday': 1}
		functioning_day_dict = {'No': 0, 'Yes': 1}
				
		str_to_date = self.get_string_to_datetime(Date)
				
		u_input_list = [Hour,Temperature,Humidity,Wind_Speed,Visibility,Solar_Radiation,Rainfall,Snowfall,
						holiday_dict[Holiday], functioning_day_dict[Functioning_Day], str_to_date['day'], str_to_date['month'],str_to_date['year']]
			
		features_name = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 
                		'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day', 'Day', 'Month', 'Year']

		df_u_input = pd.DataFrame([u_input_list], columns=features_name)
		seasons_df = self.seasons_to_df(Seasons)
		days_df = self.days_to_df(str_to_date['week_day'])
				
		df_for_pred = pd.concat([df_u_input, seasons_df, days_df], axis=1)
				
		return df_for_pred
		
		
	def prediction(self):
		df = self.users_input()
		scaled_data = self.sc.transform(df)
		prediction = self.model.predict(scaled_data)
		return prediction

if __name__ == '__main__':

	ml_model_path = r'/Users/Ankush/Desktop/Data Science/Seoul Bike Sharing Prediction/models/xgboost_regressor_r2_0_95_v1.pkl'
	standard_scaler_path = r'/Users/Ankush/Desktop/Data Science/Seoul Bike Sharing Prediction/models/sc.pkl'
		
	inference = Inference(ml_model_path, standard_scaler_path)
		
	pred = inference.prediction()

	print(f'Rented Bike Count prediction w.r.t date and time : {round(pred.tolist()[0])}')
