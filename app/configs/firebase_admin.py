import json
import os
import firebase_admin
from firebase_admin import credentials, firestore

firebase_credentials = os.environ.get("FIREBASE_CREDENTIALS")

if not firebase_credentials:
    raise ValueError("FIREBASE_CREDENTIALS environment variable is not set.")

if os.path.isfile(firebase_credentials):
    cred = credentials.Certificate(firebase_credentials)
else:
    try:
        firebase_cred_json = json.loads(firebase_credentials)
        cred = credentials.Certificate(firebase_cred_json)
    except json.JSONDecodeError as e:
        raise ValueError("FIREBASE_CREDENTIALS is not a valid path or JSON.") from e

firebase_admin.initialize_app(cred)
db = firestore.client()
