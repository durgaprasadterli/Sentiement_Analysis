from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config["DEBUG"] = True
tfidf=TfidfVectorizer(max_features=10000,ngram_range=(1,2))
#tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)

loaded_model = pickle.load(open('sentiment_model_SVC.pkl', 'rb'))
df=pd.read_csv(r'C:\Users\durgaprasad.terli\python_class\twitt30k.csv')
#dataframe = pd.read_csv(r'C:\Users\pavan\Desktop\MP\DEMO\data\train.csv')
#x = dataframe['text']
x=df['twitts']
x=tfidf.fit_transform(x)
y=df['sentiment']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=53)

def fake_news_det(news):
    input_data = news
    vectorized_input_data = tfidf.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Statement = [str(request.form['Statement'])]
        #Processed_data = fake_news_det(Statement)
        
        vectorized_input_data = tfidf.transform(Statement)
        prediction = loaded_model.predict(vectorized_input_data)
        #prediction=model.predict(Processed_data)
        output = int(prediction)
        if output == 0:
            return render_template('index.html',prediction_text="This sentence has a Negative Sentiment.")
        elif output == 1:
            return render_template('index.html',prediction_text="This sentence has a Positive Sentiment.")
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)