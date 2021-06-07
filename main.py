import uvicorn
from fastapi import FastAPI, Response, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from prediction_Validation_Insertion import pred_validation  # need to fix
from trainmodel import trainModel
from training_Validation_Insertion import train_validation
from predictFromModel import prediction
from typing import Dict
import pandas as pd
import numpy as np
# For Cross Origin

# app = FastAPI(debug = True)

from fastapi import FastAPI
#from elasticapm.contrib.starlette import make_apm_client, ElasticAPM

'''
apm = make_apm_client(
    {
        'SERVICE_NAME': 'FDP_APM',
        'ELASTIC_APM_SERVER_URL': 'http://localhost:8200',
    }

)
'''
app = FastAPI(debug=True)
#app.add_middleware(ElasticAPM, client=apm)

templates = Jinja2Templates(directory="templates")


# route to dashboard(UI)
#@app.get("/")
#def dashboard(request: predictclient):
 #   """
 #   displays the financial distress prediction dashboard/homepage
 #   """
 #   return templates.TemplateResponse("dashboard.html", {
 #       "request": request
 #   })


@app.post("/training")  # Training batch file
async def training(json_data: Dict):
    try:
        # if val is not None:
        if json_data['json_data'] is not None:
            path = json_data['json_data']
            train_valObj = train_validation(path, 'Batch')  # object initialization
            train_valObj.train_validation()  # calling the training_validation function
            trainModelObj = trainModel()  # object initialization
            path = trainModelObj.trainingModel()  # training the model for the files in the table
            return {"message": path}

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return 'Error Occurred!'
    # return Response("Training successfull!!")


@app.post("/trainclient")  # training via upload
def trainclient(file: UploadFile = File(...)):
    """
    trains the financial distress prediction
    """
    contents = file.file.read()

    from io import BytesIO
    data = BytesIO(contents)
    data = pd.read_csv(data)
    print(data.head(5))
    try:
        train_valObj = train_validation(data, 'UI')  # object initialization
        train_valObj.train_validation()  # calling the training_validation function
        trainModelObj = trainModel()  # object initialization
        path = trainModelObj.trainingModel()  # training the model for the files in the table
        print("Training is completed")
        return {"message": path}

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return 'Error Occurred!'


@app.post("/predictFileupload")  # predicting via file upload
def predictFileupload(file: UploadFile = File(...)):
    """
    trains the financial distress prediction
    """
    try:

        contents = file.file.read()

        from io import BytesIO
        data = BytesIO(contents)
        data = pd.read_csv(data)
        print(data.head(5))
        data.replace('?', np.NaN, inplace=True)
        pred_val = pred_validation(data, 'UP')  # object initialization
        pred_val.prediction_validation()  # calling the prediction_validation function
        pred = prediction(data, 'UP')  # object initialization
        # predicting for dataset present in database
        path = pred.predictionFromModel()
        # return Response("Prediction File created at %s!!!" % Data)
        return path

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


# Oredictibg by UI
@app.post("/predictclient/{val}/{val1}/{val2}/{val3}/{val4}/{val5}/{val6}/{val7}/{val8}/{val9}/")
async def predictRouteClient(val: str, val1: str, val2: str, val3: str, val4: str, val5: str, val6: str, val7: str,
                             val8: str, val9: str):
    try:
        print("Start Predicting")
        list_data = [val, val1, val2, val3, val4, val5, val6, val7, val8, val9]
        data = pd.DataFrame(list_data)
        data = data.astype(float)
        data = data.T
        print(data.head(5))
        pred_val = pred_validation(data, 'UI')  # object initialization
        pred_val.prediction_validation()  # calling the prediction_validation function
        pred = prediction(data, 'UI')  # object initialization
        # predicting for dataset present in database
        path = pred.predictionFromModel()
        # return Response("Prediction File created at %s!!!" % Data)
        # return path

        return {"message": path}
        # return {"message": "Parameter1 "+val+" Parameter 2 "+val1 +" Parameter 3 "+val2 +" Parameter 4 "+val3 +" Parameter 5 "+val4 +" Parameter 6 "+val5+" Parameter 7 "+val6 +" Parameter 8 "+val7 +" Parameter 9 "+val8 +" Parameter 10 "+val9 }
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.post("/predict")  # Predicting via batch upload
async def predictRouteClient(json_data: Dict):
    try:
        print("Start Predicting")
        if json_data['json_data'] is not None:
            path = json_data['json_data']

            pred_val = pred_validation(path, 'Batch')  # object initialization

            pred_val.prediction_validation()  # calling the prediction_validation function

            pred = prediction(path, 'Batch')  # object initialization

            # predicting for dataset present in database
            path = pred.predictionFromModel()
            return {"message": path}

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


'''
@app.post("/predictclient")
async def predictRouteupload(json_data: Dict):
    try:
        print("Start Predicting")

        if json_data.get('Data') is not None:
            Data = json_data.get('Data')
            Df = pd.DataFrame.from_dict(Data)
            Df = Df.astype(float)
            Df = Df.T
            schema_path = 'schema_prediction_ui.json'
            with open(schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
            column_names = dic['ColName']
            Df.columns = column_names
            pred_val = pred_validation(Df,'UI')  # object initialization

            pred_val.prediction_validation()  # calling the prediction_validation function

            pred = prediction(Df,'UI')  # object initialization

            # predicting for dataset present in database
            path = pred.predictionFromModel()
            return Response("Prediction File created at %s!!!" % Data)

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
'''

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn app:app --port 5000
