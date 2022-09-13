from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from msrest.authentication import ApiKeyCredentials
import azure.functions as func
from datetime import datetime
import os
import logging
import jwt
import re
import pyodbc
import requests
import json

storage_name = os.environ["storage_name"]
ENDPOINT = os.environ["customvision_endpoint"]
training_key = os.environ["customvision_trainning_key"]
prediction_resource_id = os.environ["customvision_prediction_resource"]
headers = {'Training-key': training_key,}
credentials = ApiKeyCredentials(headers)
default_credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
blob_service = BlobServiceClient(
    "https://" + storage_name + ".blob.core.windows.net/", credential=default_credential)
header_response = {
    'Content-Type': "application/json",
    'charset': "utf-8",
}

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    """[function for publish a get trainning information]

    Args:
        req (func.HttpRequest): [receive hashes lob, project and model]

    Returns:
        func.HttpResponse: [returns a json with the information]
    """
    bearer_token = req.headers.get('Authorization').replace('Bearer ', '')
    bearer_token = jwt.decode(bearer_token, algorithms="RS256", options={
                              "verify_signature": False})
    regex_folder = r"[^a-zA-Z0-9_-]"
    customer = "" if len(re.findall(
        regex_folder, bearer_token["customer_id"])) != 0 else bearer_token["customer_id"]
    user_id = int(bearer_token["user_id"])
    lob = "" if len(re.findall(regex_folder, req.params.get(
        'hashIdentifierLob'))) != 0 else req.params.get('hashIdentifierLob').lower()
    project = "" if len(re.findall(regex_folder, req.params.get(
        'hashIdentifierProject'))) != 0 else req.params.get('hashIdentifierProject').lower()
    format_customer = customer.replace("-", "_")
    cnn_db = os.environ["ocr_connection_string_"+format_customer]
    model = "" if len(re.findall(regex_folder, req.params.get(
        'hashIdentifierModel'))) != 0 else req.params.get('hashIdentifierModel').lower()
    publish_iteration_name = model

    if not model:
        return func.HttpResponse("unspecified or not valid model", status_code=400)
    if not customer:
        return func.HttpResponse("unspecified or invalid customer", status_code=400)
    if not lob:
        return func.HttpResponse("unspecified or invalid lob", status_code=400)
    if not project:
        return func.HttpResponse("unspecified or invalid hashIdentifierProject", status_code=400)
    try:
        conn = pyodbc.connect(cnn_db)
        cursor = conn.cursor()
    except Exception:
        logging.error("Error, connecting to database ")
        return 'failed'
    try:
        cv_model = get_hash_cv_project(cursor, model)
        final_response = custom_vision_prediction(trainer, cv_model, publish_iteration_name, prediction_resource_id, cursor, model, project, user_id, lob)
        cursor.close()
        del cursor
        conn.close()
        return func.HttpResponse(body=json.dumps(final_response), status_code=200, headers=header_response)
    except Exception :
        cursor.close()
        del cursor
        conn.close()
        logging.error("error getting prediction")
        model_info = {"modelId": model, "status": 'failed', "createdDateTime":datetime.today().isoformat(), "lastUpdatedDateTime":datetime.today().isoformat()}
        train_result ={"errors": [{"code": "0","message":"error getting prediction"}]}
        failed_resp = dict(modelInfo = model_info, trainResult = train_result)
        return  func.HttpResponse(body=json.dumps(failed_resp), status_code=400, headers=header_response)

def custom_vision_prediction(trainer, cv_model,publish_iteration_name, prediction_resource_id, cursor, model, project, user_id, lob):
    try:
        iterations = get_iterations(cv_model)
        if iterations != []:
            iteration_id = iterations[-1]['id']
            result = get_train_status(cv_model, iteration_id)
            model_info = {"modelId": model, "status": result['status'], "createdDateTime":result['created'], "lastUpdatedDateTime":result['lastModified']}
            status = result['status']
            if status == 'Training':
                update_model(cursor, lob, project, model, cv_model, status, user_id, 0)
                return dict(modelInfo = model_info, trainResult = {"errors": []})
            elif status == 'Failed':
                update_model(cursor, lob, project, model, cv_model, status, user_id, 0)
                return dict(modelInfo = model_info, trainResult = {"errors": [{"code": "0", "message": "Trainning failed"}]})
            elif status == 'Completed':
                if result['publishName'] == None:
                    trainer.publish_iteration(cv_model, iteration_id, publish_iteration_name, prediction_resource_id, overwrite=True)                
                performance = get_iteration_performance(cv_model, iteration_id)
                result_from_prediction =  [{'tagName': tag['name'], 'precision':tag['precision'], 'recall':tag['recall'], 'ap':tag['averagePrecision']} for tag in performance['perTagPerformance']]            
                tags_from_db = get_labels(cursor, model, project)
                for x in result_from_prediction:
                    [update_trainning_fields_sp(cursor, dict(hashlabel=z['HashIdentifierLabel'], hashproject=z['HashIdentifierProject'], hashmodel=z['HashIdentifierModel'], tag_title=z['TagTitle'], field_type=z['FielType'], min_percentage=z['MinPercentage'], accuracy=x['ap'], negative=z['Negative'], precision=x['precision'], recall= x['recall'], image_count=z['ImagenCount'], color_tag=z['ColorTag'], user_id=user_id)) for z in tags_from_db if x['tagName']==z['HashIdentifierLabel'].lower()]
                model_info = {"modelId": model, "status": status, "createdDateTime":result['created'], "lastUpdatedDateTime":result['lastModified']}
                train_result = {"precision":performance['precision'],"recall":performance['recall'],"ap":performance['averagePrecision'], "tags":result_from_prediction, "errors": []} 
                update_model(cursor, lob, project, model, cv_model, 'trained', user_id, 0)
                return dict(modelInfo = model_info, trainResult = train_result)
        else:
            model_info = {"modelId": model, "status":"Error", "createdDateTime":None, "lastUpdatedDateTime":None}
            train_result ={"errors": [{"code": "0","message":"There are not iteration training"}]}
            return dict(modelInfo = model_info, trainResult = train_result)
    except Exception:
        error_log = "Error getting Custom Vision Trainning Status"
        logging.error(error_log)
        raise ValueError(error_log)

def get_hash_cv_project(cursor, model):
    try:
        sql = """[dbo].[GetModel_SEL] @HashIdentifierModel = ?"""
        val = (model)
        cursor.execute(sql, val)
        response_id = cursor.fetchone()
        cursor.commit()
        return response_id[6]
    except Exception:
        error_log = "error getting hash cv project"
        logging.error(error_log)
        raise ValueError(error_log)

def update_trainning_fields_sp(cursor, fields):
    try:
        sql = """
                Exec [dbo].[UpdateLabel_UPD]
                @HashIdentifierLabel  = ?,
                @HashIdentifierProject = ?,
                @HashIdentifierModel = ?,
                @TagTitle = ?,
                @FielType = ?,
                @MinPercentage = ?,
                @Accuracy = ?,
                @Negative = ?,
                @Precision = ?,
                @Recall = ?,
                @ImagenCount = ?,
                @ColorTag = ?,
                @UserModified = ?        
            """
        val = (fields['hashlabel'], fields['hashproject'], fields['hashmodel'], fields['tag_title'], fields['field_type'], fields['min_percentage'], fields['accuracy'], fields['negative'], fields['precision'], fields['recall'], fields['image_count'], fields['color_tag'], fields['user_id'])
        cursor.execute(sql, val)
        response = cursor.fetchone()
        cursor.commit()
        return response
    except Exception:
        error_log = "Error Updating cv tag in db"
        logging.error(error_log)
        raise ValueError(error_log)

def get_labels(cursor, model, project):
    try:
        sql = "Exec [dbo].[GetAllLabel_SEL] @HashIdentifierModel = ?, @HashIdentifierProject = ?"
        val = (model, project)
        cursor.execute(sql, val)
        data = cursor.fetchall()
        cursor.commit()
        result = build_json_labels(data)
        return result
    except Exception:
        error_log = "Connection get labels error"
        logging.error(error_log)
        raise ValueError(error_log)

def get_iteration_performance(cv_model, iteration_id):
    try:
        r = requests.get(
            f'{ENDPOINT}/customvision/v3.3/Training/projects/{cv_model}/iterations/{iteration_id}/performance?overlapThreshold=0.3&threshold=0.5', headers=headers)
        if r.status_code == 200:
            return json.loads(r.text)
        else:
            return "Failed"
    except Exception:
        error = 'Error get iteration performance'
        logging.error(error)
        return error

def get_iterations(cv_model):
    try:
        r = requests.get(
            f'{ENDPOINT}/customvision/v3.3/Training/projects/{cv_model}/iterations', headers=headers)
        return json.loads(r.text)
    except Exception:
        error = 'Cannot get iterations'
        logging.error(error)
        return str(error)

def build_json_labels(data):
    cont = []
    for x in data:
        assets={}
        assets["Id"]=x[0] 
        assets["HashIdentifierLabel"]=x[1]
        assets["HashIdentifierModel"]=x[2]
        assets["HashIdentifierProject"]=x[3]
        assets["TagTitle"]=x[4]
        assets["FielType"]=x[5]
        assets["MinPercentage"]=x[6]
        assets["Accuracy"]=x[7]
        assets["Negative"]=x[8]
        assets["ColorTag"]=x[9]
        assets["Precision"]=x[10]
        assets["Recall"]=x[11]
        assets["ImagenCount"]=x[12]
        assets["Activate"]=x[13]
        assets["UserCreated"]=x[14]
        assets["UserModified"]=x[15]
        assets["DateCreated"]=x[16]
        assets["DateModified"]=x[17]
        assets["Deleted"]=x[18]
        assets["IdCvTag"]=x[19]
        cont.append(assets)
    return cont

def get_train_status(cv_model, iteration_id):
    try:
        r = requests.get(
            f'{ENDPOINT}/customvision/v3.3/Training/projects/{cv_model}/iterations/{iteration_id}', headers=headers)
        return json.loads(r.text)
    except Exception:
        error = 'Error get trainning status'
        logging.error(error)
        return error

def update_model(cursor, lob, project, model, idmodel, statusmodel, usermodified, averagemodelaccuracy):
    try:
        sql = "Exec [dbo].[UpdateModelTraining_UDP] @HashIdentifieLob = ?, @HashIdentifierProject = ?, @HashIdentifierModel = ?, @IdModel = ?, @StatusModel = ?, @UserModified = ?, @AverageModelAccuracy = ?"
        val = (lob, project, model, idmodel, statusmodel,
               usermodified, averagemodelaccuracy)
        cursor.execute(sql, val)
        cursor.commit()
    except Exception:
        error_log = 'error updating database model'
        logging.error(error_log)
        raise ValueError(error_log)