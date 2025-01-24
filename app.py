from src.search import Searcher
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

id_export = 0
model_list = [True, True, True]  # clip, blip, beit
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

@app.route('/process_query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data['query']
    print("Processing query...")
    print(query)

    if query[0] == 'image': # image search [type, path, top_k]
        result_info = searching_system.image_search(
            image_path=query[1],
            top_k=100 if query[2] == '' else int(query[2])
        )
    else: # text search [type, topk, query_text, clip, blip, beit]
        searching_system.update_searching_mode(
            clip_engine=bool(query[2]),
            blip_engine=bool(query[3]),
            beit_engine=bool(query[4])
        )

        result_info = searching_system.text_search(
            query_text=query[1].lower(),
            top_k=100 if query[0] == '' else int(query[0])
        )

    # searcher.image_search(image_path='./data/raw/images/flickr30k/0.jpg', top_k=10)
    print(result_info)
    result_with_index = {i: [key, float(value)] for i, (key, value) in enumerate(result_info.items())}  # add index to result_info {0: [key, value], 1: [key, value], ...}
    return jsonify(result_with_index)
