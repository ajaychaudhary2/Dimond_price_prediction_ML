

from DimondPricePrediction.pipelines.prediction_pipeline import customdata,predictpipeline

from flask import Flask,render_template,request, jsonify



app = Flask(__name__)


@app.route('/',methods=["GET", "POST"])
def predict_datapoint():
    if request.method=="GET":
        
        return render_template("form.html")
    
    
    else:
        
        data = customdata(
            
            carat=float(request.form.get("carat")),
            depth=float(request.form.get("depth")),
            table=float(request.form.get("table")),
            x=float(request.form.get("x")),
            y=float(request.form.get("y")),
            z=float(request.form.get("z")),
            cut=(request.form.get("cut")),
            color=(request.form.get("color")),
            clarity=(request.form.get("clarity"))
            
            
            
            
        )
        
        final_data = data.get_data_as_df()
        
        predict_pipeline = predictpipeline()
        
        pred=predict_pipeline.predict(final_data)
        
        result =round(pred[0],2)
        
        return render_template("result.html",final_result=result)    
    

if __name__ == "__main__":
    app.run(debug=True)