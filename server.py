import os
import numpy as np
import pandas as pd
import random
from flask import Flask,request,send_file,jsonify
from recommender import Recommender 

app = Flask(__name__)

@app.route('/randomimages', methods = ['GET'])
def randomimages():
	content = request.get_json()
	ids = []
	for i in range(20):
		ids.append(random.randint(0,df.shape[0]))
	return df[['image_name']].iloc[ids].to_json(orient='records')
	#recomimages
@app.route('/recomimages', methods = ['GET'])
def recomimages():
	return jsonify(rec.getSet())
	
@app.route('/rate', methods = ['POST'])
def rate():
	content = request.get_json()
	global df_r
	global count
	#print(df_r)
	for r in content:
		print(r['image_name'])
		rec.updateUsed(r['image_name'])
		df_r = df_r.append(pd.DataFrame([[r['image_name'],r['target']]],columns = ['image_name','target']))
	count += 1
	
	if count>=15:
		df_r.to_csv('likes.csv', index=False)
		#rec.train(df_r)
		count = 0
	#print(df_r)
	
	return "OK"
	
@app.route('/images/<string:pid>', methods = ['GET'])
def imageById(pid):
	return send_file('Selfie-dataset/images/'+pid+'.jpg', mimetype='image/jpg')

	
if __name__ == "__main__":
	rec = Recommender()
	rec.train(pd.read_csv('likes.csv'))
	df = pd.read_csv('Selfie-dataset/selfie_dataset.csv')
	df = df[(df.baby == 0) & (df.child == 0)]
	df_r = pd.DataFrame()
	count = 0
	print('df_r init ', df_r.shape[0])
	app.run(host = "0.0.0.0", debug=True)