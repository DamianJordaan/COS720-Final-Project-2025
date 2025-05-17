from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import List, Tuple
from uuid import uuid4
import secrets
import uvicorn
import re
import pandas as pd

from lime.lime_text import LimeTextExplainer
import numpy as np

from detector import PhishingEmailDetector
from sim_detector import format_check_phishing_email_simularity

# pip install python-multipart, fastapi, uvicorn

app = FastAPI()

API_KEYS = {}
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

##############################################################################################################
# SECURITY
def generate_api_key():
    return secrets.token_hex(16)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid or expired API key")
    return api_key

def rotate_api_key(old_key):
    new_key = generate_api_key()
    API_KEYS.pop(old_key, None)
    API_KEYS[new_key] = True
    return new_key

##############################################################################################################
# CORE LOGIC
class WordScore(BaseModel):
    word: str
    score: float

analyzer = PhishingEmailDetector()
analyzer.load_model("phishing_email_detector_v2.pkl")

email_database = pd.read_csv("phishing_dataset.csv")
email_database['id'] = email_database.index

##############################################################################################################
# ROUTES
@app.post("/check-email-form")
def check_email_form(
    sender: str = Form(...),
    receiver: str = Form(...),
    subject: str = Form(...),
    body: str = Form(...),
    detailed: bool = Form(False),
    api_key: str = Depends(verify_api_key)
):
    # print("s1")
    prediction, confidence_x = analyzer.predict_single(subject, body, sender, receiver)
    # print("s2")
    confidence = confidence_x.astype(float)
    # print("s3")
    # print(prediction)
    result = prediction.astype(int)
    # print("s4")
    explination = analyzer.explain_single_instance_as_list(subject, body, sender, receiver)
    # print("s5")

    # print("result:", result)
    # print("explanation:", explination)

    # convert to list of WordScore
    scores = []
    for wordx, scorex in explination:
        word = wordx
        score = round(scorex * 100, 2)
        # print("word:", word)
        # print("score:", score)
        scores.append(WordScore(word=word, score=score))

    # print("scores:", scores)
    # print("result:", result)

    print(f"API Key: {api_key}")

    result_txt = "No - " + str(confidence)
    res_label = 0
    if result == 1:
        print("Suspicious email detected!")
        result_txt = "Yes - " + str(confidence)
        res_label = 1
    else:
        print("Email is safe.")

    logs = ""
    if detailed:
        # logs += '<h2>Details</h2><div style="border: 1px solid #ccc; padding: 10px; white-space: pre-wrap;">'
        sim_logs, avg_similarity, sim_lable_decision = format_check_phishing_email_simularity(
            sender=sender,
            receiver=receiver,
            subject=subject,
            body=body,
            db_path="",
            df=email_database,
            ml_classification_lable=res_label,
        )

        logs += f"ML Classification: {result_txt}\n"
        logs += f"Similarity Check: {avg_similarity}\n"
        if avg_similarity == 0:
            logs += "No similar emails found.\n"
        else:
            logs += f"Similarity Label Decision: {sim_lable_decision}\n"
            if sim_lable_decision != res_label:
                logs += "Warning: Similarity check disagrees with ML classification. User should be cautious.\n"
        logs += f"Similarity Results:\n{sim_logs}\n\n"
        logs += f"Lime Explanation:\n {explination}\n\n"

        # logs += "</div>"

    new_key = rotate_api_key(api_key)
    return {"result": result_txt, "scores": scores, "logs": logs, "new_api_key": new_key}

@app.get("/", response_class=HTMLResponse)
def load_page():
    key = generate_api_key()
    API_KEYS[key] = True
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Email Checker</title>
        <style>
            textarea {{
                width: 98%;
            }}
        </style>
        <script>
        async function submitForm() {{
            document.getElementById('result').innerText = 'Checking...';
            document.getElementById('highlighted-body').innerHTML = '';
            document.getElementById('highlighted-subject').innerHTML = '';
            document.getElementById('result_logs').innerText = '';

            const formData = new FormData(document.getElementById('emailForm'));
            const token = localStorage.getItem('token');
            const response = await fetch('/check-email-form', {{
                method: 'POST',
                headers: {{ 'access_token': token }},
                body: formData
            }});
            const data = await response.json();
            localStorage.setItem('token', data.new_api_key);
            document.getElementById('result').innerText = 'Phishing: ' + data.result;
            highlightWords(data.scores);
            document.getElementById('result_logs').innerText = data.logs;
        }}

        
        
        function highlightWords(scores) {{
            let body = document.getElementById('body').value;
            let subject = document.getElementById('subject').value;

            for (const s of scores) {{
                console.log("s.score:" + s.score);
                console.log("s.word:" + s.word);
                const re = new RegExp(`\\b(${{s.word}})\\b`, 'gi');
                console.log("re:" + re);

                if (s.score < 0) {{
                    body = body.replace(s.word, "<span style='color:red;'>" + s.word + "</span>");
                    subject = subject.replace(s.word, "<span style='color:red;'>" + s.word + "</span>");
                }} else {{
                    body = body.replace(s.word, "<span style='color:blue;'>" + s.word + "</span>");
                    subject = subject.replace(s.word, "<span style='color:blue;'>" + s.word + "</span>");
                }}

                <!-- body = body.replace(s.word, "<span style='color:green;'>" + s.word + "</span>");>
                <!-- subject = subject.replace(s.word, "<span style='color:green;'>" + s.word + "</span>");>

                
            }}

            document.getElementById('highlighted-body').innerHTML = body;
            document.getElementById('highlighted-subject').innerHTML = subject;
        }}



        </script>
    </head>
    <body onload="localStorage.setItem('token', '{key}')">
        <h1>Email Checker</h1>
        
        
        <form id="emailForm" onsubmit="event.preventDefault(); submitForm();">
            <label>Sender: <input name="sender" type="text" required></label><br>
            <label>Receiver: <input name="receiver" type="text" required></label><br>
            <h3>Subject:</h3>
            <label><textarea id="subject" name="subject" rows="2" cols="80"></textarea></label><br>
            <h3>Body:</h3>
            <label><textarea id="body" name="body" rows="10" cols="80"></textarea></label><br>
            <label><input type="checkbox" name="detailed" value="true"> Detailed Results</label><br>
            <button type="submit">Check Email</button>
        </form>

        <br><hr><hr>
        <h2>Results</h2>

        <div id="result"></div>

        <h3>Highlighted Subject</h3>
        <div id="highlighted-subject" style="border: 1px solid #ccc; padding: 10px;"></div>

        <h3>Highlighted Body</h3>
        <div id="highlighted-body" style="border: 1px solid #ccc; padding: 10px; white-space: pre-wrap;"></div>

        <br><hr><hr>
        <h3>Details</h3>
        <div id="result_logs" style="border: 1px solid #ccc; padding: 10px; white-space: pre-wrap;"></div>
    </body>
    </html>
    """

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
    uvicorn.run(app, host="0.0.0.0", port=8000)
