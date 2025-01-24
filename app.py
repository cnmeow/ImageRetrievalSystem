from src.search import Searcher
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

id_export = 0
model_list = [True, False, True]  # clip, blip, beit
searching_system = Searcher(
    use_clip=model_list[0],
    use_blip=model_list[1],
    use_beit=model_list[2]
)

okresponse = {
    'status': 'ok'
}

@app.route('/')
def index():
    return render_template('search.html', model_list=model_list)

@app.route('/text_search', methods=['POST'])
def text_search():
    data = request.get_json()
    query = data['query']
    print("Processing query...")
    print(query)
    
    # search [topk, clip, blip, beit, query_text]
    searching_system.update_searching_mode(
        clip_engine=bool(query[1]),
        blip_engine=bool(query[2]),
        beit_engine=bool(query[3])
    )

    result_info = searching_system.text_search(
        query_text=query[4].lower(),
        top_k=100 if query[0] == '' else int(query[0])
    )

    print(result_info)
    result_with_index = {i: [key, float(value)] for i, (key, value) in enumerate(result_info.items())}  # add index to result_info {0: [key, value], 1: [key, value], ...}
    return jsonify(result_with_index)

@app.route('/image_search', methods=['POST'])
def image_search():
    file = request.files['file']
    option = request.form.get('option').split(',')
    print("Processing image...")

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    searching_system.update_searching_mode(
        clip_engine=bool(option[1]),
        blip_engine=bool(option[2]),
        beit_engine=bool(option[3])
    )
    
    result_info = searching_system.image_search(
        image_path=file_path,
        top_k=100 if option[0] == '' else int(option[0])
    )

    print(result_info)
    result_with_index = {i: [key, float(value)] for i, (key, value) in enumerate(result_info.items())}  # add index to result_info {0: [key, value], 1: [key, value], ...}
    return jsonify(result_with_index)
    
