from src.search import Searcher
from flask import Flask, render_template, request, jsonify
from io import BytesIO
import speech_recognition as sr

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

id_export = 0
model_list = [False, False, False]  # clip, blip, beit
searching_system = None

okresponse = {
    'status': 'ok'
}
recognizer = sr.Recognizer()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    global model_list, searching_system
    data = request.get_json()
    model_list = data['query']  # list bool [clip, blip, beit]
    if searching_system is None:
        searching_system = Searcher(
            use_clip=model_list[0],
            use_blip=model_list[1],
            use_beit=model_list[2]
        )
    return jsonify(okresponse)

@app.route('/search')
def search():
    return render_template('search.html', model_list=model_list)

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"})

    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        # Save the audio file to the static folder
        audio_data = BytesIO()
        file.save(audio_data)
        audio_data.seek(0)
        print("Processing audio...")
        
        # Use SpeechRecognition to process the audio
        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)
            # Convert speech to text
            text = recognizer.recognize_google(audio)
        print(text)
        return jsonify({"transcription": text})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)})

@app.route('/process_query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data['query']
    print("Processing query...")
    print(query)
    
    # search [topk, query_text, clip, blip, beit]
    searching_system.update_searching_mode(
        clip_engine=bool(query[2]),
        blip_engine=bool(query[3]),
        beit_engine=bool(query[4])
    )

    result_info = searching_system.text_search(
        query_text=query[1].lower(),
        top_k=100 if query[0] == '' else int(query[0])
    )

    print(result_info)
    result_with_index = {i: [key, float(value)] for i, (key, value) in enumerate(result_info.items())}  # add index to result_info {0: [key, value], 1: [key, value], ...}
    return jsonify(result_with_index)
