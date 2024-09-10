import re
import json
import streamlit as st
import datetime
import os
from icecream import ic
import google.auth
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload

def clean_latex_formatting(text: str) -> str:
    # Entfernt alle LaTeX-Mathematik-Umgebungen
    cleaned_text = re.sub(r"\\\(", "", text)
    cleaned_text = re.sub(r"\\\)", "", cleaned_text)
    return cleaned_text

def save_state_json_local():
    uuid = st.session_state.get("user_uuid")
    session_state_dict = {}
    for key, value in st.session_state.items():
        if key not in ["assistant"]:
            session_state_dict[key] = str(value)
    session_state_dict["last_updated"] = datetime.datetime.now().isoformat()

    if not os.path.exists("state_data"):
        os.makedirs("state_data")
    with open(f"state_data\{uuid}.json", "w") as f:
        json.dump(session_state_dict, f)

def get_tipi_result_from_session_state(uuid):
    # TODO: maybe delete this function
    return st.session_state["tipi_scores"]

def authenticate_with_service_account():
    # Service Account Schlüssel aus Streamlit Secrets laden
    credentials_info = st.secrets["google_drive"]
    # import json
    # credentials_info = json.load("serviceacc.json")
    credentials = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=["https://www.googleapis.com/auth/drive.file"]  # Scope zum Hochladen von Dateien
    )
    
    drive_service = build('drive', 'v3', credentials=credentials)
    return drive_service
def find_file_in_drive(drive_service, file_name):
    query = f"name='{file_name}'"
    response = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = response.get('files', [])
    if files:
        return files[0]['id']  # Datei gefunden, gibt die Datei-ID zurück
    return None
# JSON-Datei zu Google Drive hochladen
def upload_to_gdrive(session_state_dict, file_name):
    drive_service = authenticate_with_service_account()

    # JSON-Daten in Bytes konvertieren
    json_data = json.dumps(session_state_dict)
    json_bytes = json_data.encode('utf-8')

    # Datei in Google Drive suchen
    file_id = find_file_in_drive(drive_service, file_name)

    media = MediaInMemoryUpload(json_bytes, mimetype='application/json')

    if file_id:
        # Datei existiert -> Neue Version hochladen
        file = drive_service.files().update(fileId=file_id, media_body=media).execute()

    else:
        # Datei existiert nicht -> Neu erstellen
        file_metadata = {'name': file_name}
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    

    return file.get('id')

def save_state_json():
    uuid = st.session_state.get("user_uuid")
    session_state_dict = {}
    
    # Session State sammeln
    for key, value in st.session_state.items():
        if key not in ["assistant"]:  # Exclude specific keys if necessary
            session_state_dict[key] = str(value)
    
    session_state_dict["last_updated"] = datetime.datetime.now().isoformat()

    # JSON-Datei direkt in Google Drive speichern
    file_name = f"{uuid}.json"
    file_id = upload_to_gdrive(session_state_dict, file_name)
    