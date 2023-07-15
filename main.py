# Prepare libraries
from flask import Flask, render_template, request,json
import functions as func
import pickle
import warnings
import arabic_reshaper
from bidi.algorithm import get_display

# Stop not important warnings and define the main flask application
warnings.filterwarnings("ignore")
main_application = Flask(__name__)
main_application.debug = True

# model = pickle.load(open('models/ar_kmeans.pkl', 'rb'))

def formatArabicSentences(sentences):
   formatedSentences = arabic_reshaper.reshape(sentences)
   return get_display(formatedSentences)
   
# Application home page
@main_application.route("/")
def index():
    return render_template("index.html", page_title="Text Summarizer & Categorical")

                 
# Analyze text page
@main_application.route("/analyze", methods=['POST'])
def analyze_text2():


    input_text = request.form['text_input_text']
    classifier_model_name = request.form['text_classifier']
    
    sentences_number = int(request.form['text_sentences_number'])
    if classifier_model_name == 'kmeans':
        model = pickle.load(open('models/ar_'+classifier_model_name+'.pkl', 'rb'))
        text_summary, text_category  = func.summerize_category_text_kmeans(input_text,sentences_number,model)
        
    else:
        classifier_model = pickle.load(open('models/ar_' + classifier_model_name + '.pkl', 'rb'))
        text_summary, text_category = func.summarize_category_supervised(input_text, sentences_number, classifier_model)

    text_summary1= formatArabicSentences(text_summary)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print(str(text_summary1))
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    return "النوع: "+text_category+"<br/>"+"التلخيص : "+  text_summary +"<br/> "+ "model : "+ classifier_model_name +"<br/>"
    #return json.dumps({'status':'OK','text_summary':text_summary,'Category':text_category})

    
    # return render_template("index.html", page_title="Text Summarizer & Categorical", input_text=input_text, text_summary=text_summary, text_category=text_category)
                          
# Start the application on local server
if __name__ == "__main__":
    main_application.run(host = 'localhost', debug = True)