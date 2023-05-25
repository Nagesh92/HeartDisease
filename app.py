from flask import Flask,render_template,request,url_for
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route("/")
def man():
    return render_template('home.html')


@app.route("/predict", methods = ['POST'])
def home():
    # age = float(request.form['age'])
    # Sex = float(request.form['Sex'])
    # ChestPainType = float(request.form['ChestPainType'])
    # RestingBP = float(request.form['RestingBP'])
    # Cholestrol = float(request.form['Cholestrol'])
    # FastingBS = float(request.form['FastingBS'])
    # RestingECG = float(request.form['RestingECG'])
    # MaxHR = float(request.form['MaxHR'])
    # ExerciseAngina = float(request.form['ExerciseAngina'])
    # OldPeak = float(request.form['OldPeak'])
    # ST_Slope = float(request.form['ST_Slope'])
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    predict = model.predict(final_features)[0]


    # result = model.predict([['age','Sex','ChestingPainType','RestingBP','Cholestrol','FastingBS','RestingECG','MaxHR','ExerciseAngina','OldPeak','ST_Slope']])[0]
    return render_template('new.html',data =predict)


if __name__ == "__main__":
    app.run(debug=True)

