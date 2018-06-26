from flask import Flask, render_template, request
import pickle
import time
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

app = Flask(__name__)


print "Preparing classifier"
start_time = time.time()

with open('Vectorizer.pkl', 'rb') as f:
	from lemmatokenizer import LemmaTokenizer
	vectorizer = pickle.load(f) 	
with open('LogitR.pkl', 'rb') as f:
	clf = pickle.load(f) 

print "Classifier is ready"
print time.time() - start_time, "seconds"

@app.route("/sentiment-demo", methods=["POST", "GET"])
def index_page(text="", prediction_message=""):
    if request.method == "POST":
        text = request.form["text"]
        logfile = open("ydf_demo_logs.txt", "a")
        print text
        logfile.write('<response> ')
        logfile.write(text)
        text_vect = vectorizer.transform([text])
        prediction = clf.predict(text_vect)
        if prediction == 1:
        	prediction_message = "Positive"
        if prediction == 0:
            prediction_message = "Negative"
        logfile.write(' ' + str(prediction[0]))
        logfile.write(' </response> \n')
        logfile.close()
    return render_template('hello.html', text=text, prediction_message=prediction_message)


if __name__ == "__main__":
    app.run(port=1080, debug=False)
