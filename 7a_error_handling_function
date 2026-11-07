import requests
import json

def emotion_detector(text_to_analyze):
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    myobj = { "raw_document": { "text": text_to_analyze } }
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_standard"}
    
    # Manejo de entradas vacías
    if not text_to_analyze or text_to_analyze.strip() == "":
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    response = requests.post(url, json = myobj, headers = header)
    
    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    formatted_response = json.loads(response.text)
    embedding = formatted_response['emotionPredictions'][0]['emotion']
    
    # Obtener emoción dominante
    dominant_emotion = max(embedding, key=embedding.get)
    
    return {
        'anger': embedding['anger'],
        'disgust': embedding['disgust'],
        'fear': embedding['fear'],
        'joy': embedding['joy'],
        'sadness': embedding['sadness'],
        'dominant_emotion': dominant_emotion
    }
    
