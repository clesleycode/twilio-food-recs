import foursquare
import pandas as pd
from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
import random 
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)

class BobaRecs:

	def __init__(self):
		self.boba_places = pd.read_csv('./data/boba_final.csv')
		self.client = foursquare.Foursquare(client_id='key-here', client_secret='key-here')
		self.test_data = pd.read_csv("./data/Reviews.csv")
		self.log_model = self.train_model()
		self.vectorizer = None


	def get_boba_places(self):
		'''
		    This function converts latitude and longitudes to coordinate string
		    returns param_list, which is a numpy array of coordinates and the boba place name
		'''
		self.boba_places["li"] = self.boba_places['Lat'].map(str) + ", " + self.boba_places['Longitude'].map(str)
		boba_params = self.boba_places[['li', 'Name']]
		
		params_list = boba_params.T.to_dict().values()
		return params_list


	def get_tips(self, params_list):
		'''
		    This function extracts tips for each boba place using FourSquare API
		    returns tips: list of actual tips
		            ids: list of business ids
			    tips_ids: list of ids for each tip
		'''
		queries = []
		for i in params_list:
		    queries.append(self.client.venues.search(params={'ll': i['li'], 'query': i['Name']}))

		ids = []
		tips_ids = {}
		for i in queries:
		    ids.append(i['venues'][0]['id'])
		    tips_ids[i['venues'][0]['id']] = []


		tips = []
		for i in ids:
		    tips.append(self.client.venues.tips(VENUE_ID=i))

		return tips, ids, tips_ids


	def get_formatted_tips(self, tips, ids, tips_ids):
		'''
		    This function TBD
		    returns format_tips:
		            review_ids: 
		'''
		format_tips = []
		ind = 0 
		review_ids = []
		del tips[0]
		for i in tips:
		    for j in i['tips']['items']:
		        format_tips.append(j['text'])
		        tips_ids[ids[ind]].append(j['text'])
		        review_ids.append(ids[ind])
		    ind = ind + 1
		return format_tips, review_ids



	def train_model(self):
		'''
		    This function trains the logistic regression model to classify the rating
		    returns log_model
		'''
		data = list(self.test_data['Text'])
		data_labels = list(self.test_data['Score'])

		self.vectorizer = TfidfVectorizer(
		    analyzer = 'word',
		    lowercase = True,
		)
		features = self.vectorizer.fit_transform(
		    data
		)

		features_nd = features.toarray()

		X_train, X_test, y_train, y_test  = train_test_split(
		        features_nd, 
		        data_labels,
		        train_size=0.90, 
		        random_state=1234)

		log_model = LogisticRegression()
		print(log_model)
		log_model = log_model.fit(X=X_train, y=y_train)

		return log_model


	def get_recs(self, format_tips):
		X_test = self.vectorizer.transform(format_tips)
		predictions = self.log_model.predict(X_test) 
		self.boba_places['id'] = pd.Series(ids) # this adds the business id to the dataframe
		return predictions


	def get_flavors(self, format_tips, review_ids, predictions):

		tip = pd.Series(format_tips)
		rating = pd.Series(predictions)
		bus_ids = pd.Series(review_ids)
		reviews = pd.DataFrame(dict(tip=tip, rating=rating, bus_ids=bus_ids))



		boba_flavors = ['almond', 'apple', 'black', 'caramel', 'chai', 'classic', 'coconut', 
		                'coffee', 'chocolate', 'earl', 'french', 'ginger', 'grapefruit', 'green', 
		                'hazelnut', 'horchata', 'honey', 'honeydew', 'jasmine', 'lavender', 
		                'lemon', 'lychee', 'mango', 'matcha', 'oolong', 'passion', 'peach', 
		                'regular', 'rose', 'sesame', 'strawberry', 'taro', 'thai', 'vanilla', 'watermelon']

		reviews['flavor'] = np.nan

		return reviews


	def get_reviews(self, reviews):

		for i in range(len(reviews)):
		    buff = reviews['tip'][i].lower().split()
		    for j in boba_flavors:
		        if j in buff:
		            reviews['flavor'][i] = j

		reviews.dropna(axis = 0, inplace = True)


		full_ratings = reviews.groupby(['bus_ids', 'flavor']).rating.mean()

		full_ratings_df = pd.DataFrame(full_ratings).groupby(['flavor', 'bus_ids']).rating.mean()

		full_ratings_df = full_ratings_df.reset_index()

		recs = full_ratings_df.groupby('flavor', as_index=False).apply(self.func).reset_index(drop=True)
		return recs


	def func(self, group):
	    return group.loc[group['rating'] == group['rating'].max()]



@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():

	boba_recs = BobaRecs()

	params = boba_recs.get_boba_places()
	tips = boba_recs.get_tips(params)

	ftips, freviews = boba_recs.get_formatted_tips(tips)
	predictions = boba_recs.get_recs(ftips) 

	flavors = boba_recs.get_flavors(ftips, freviews, predictions)

	recs = boba_recs.get_reviews(flavors)

	body = request.values.get('Body', None)

	recs_df = recs[recs['flavor'] == body]

	resp = MessagingResponse()

	resp.message(recs_df)

	return str(resp)


if __name__ == "__main__":
	#app.run(debug=True)
	boba_recs = BobaRecs()

	params = boba_recs.get_boba_places()
	print("HELLO1")
	tips, ids, tips_ids = boba_recs.get_tips(params)
	print("HELLO2")


	ftips, freviews = boba_recs.get_formatted_tips(tips, ids, tips_ids)
	print("HELLO3")
	predictions = boba_recs.get_recs(ftips) 
	print("HELLO4")

	flavors = boba_recs.get_flavors(ftips, freviews, predictions)
	print("HELLO5")

	recs = boba_recs.get_reviews(flavors)
	print(recs)
