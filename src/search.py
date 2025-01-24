from src.models.clip import CLIPImageEncoder, CLIPSearchEngine
from src.models.blip import BLIPImageEncoder, BLIPSearchEngine
from src.models.beit import BEITImageEncoder, BEITSearchEngine
from src.utils import merge_searching_results_by_addition
import numpy as np

clip_config = {
    'bin_file': './index/flickr30k/clip.bin',
    'metadata': './index/flickr30k/metadata.json'
}

blip_config = {
    'bin_file': './index/flickr30k/blip.bin',
    'metadata': './index/flickr30k/metadata.json'
}

beit_config = {
    'bin_file': './index/flickr30k/beit.bin',
    'metadata': './index/flickr30k/metadata.json'
}

class Searcher:

    def __init__(self, use_clip: bool = True, use_blip: bool = True, use_beit: bool = True):
        self.clip_engine = CLIPSearchEngine(**clip_config) if use_clip else None
        self.blip_engine = BLIPSearchEngine(**blip_config) if use_blip else None
        self.beit_engine = BEITSearchEngine(**beit_config) if use_beit else None
        self.searching_mode = {
            'clip_engine': use_clip,
            'blip_engine': use_blip,
            'beit_engine': use_beit
        }

    def update_searching_mode(self, clip_engine: bool, blip_engine: bool, beit_engine: bool):
        self.searching_mode = {
            'clip_engine': clip_engine,
            'blip_engine': blip_engine,
            'beit_engine': beit_engine
        }
        
    def image_search(self, image_path: str, top_k: int):
        list_results = []
        if self.searching_mode['clip_engine']:
            image_encoder = CLIPImageEncoder()
            _, image_encoding = image_encoder.encode_image([image_path], batch_size=1)
            result = self.clip_engine.embedding_search(embedding=np.array(image_encoding), top_k=top_k)
            list_results.append(result)
            
        if self.searching_mode['blip_engine']:
            image_encoder = BLIPImageEncoder()
            _, image_encoding = image_encoder.encode_image([image_path])
            result = self.blip_engine.embedding_search(embedding=np.array(image_encoding), top_k=top_k)
            list_results.append(result)
        
        if self.searching_mode['beit_engine']:
            image_encoder = BEITImageEncoder()
            _, image_encoding = image_encoder.encode_image([image_path], batch_size=1)
            result = self.beit_engine.embedding_search(embedding=np.array(image_encoding), top_k=top_k)
            list_results.append(result)
        
        final_result = merge_searching_results_by_addition(list_results)
        top_k_final_result = dict(list(final_result.items())[:top_k])
        return top_k_final_result

    def text_search(self, query_text: str, top_k: int):
        list_results = []
        if self.searching_mode['clip_engine']:
            result = self.clip_engine.text_search(
                query_text=query_text,
                top_k=top_k
            )
            list_results.append(result)
        
        if self.searching_mode['blip_engine']:
            result = self.blip_engine.text_search(
                query_text=query_text,
                top_k=top_k
            )
            list_results.append(result)

        if self.searching_mode['beit_engine']:
            result = self.beit_engine.text_search(
                query_text=query_text,
                top_k=top_k
            )
            list_results.append(result)
        
        final_result = merge_searching_results_by_addition(list_results)
        top_k_final_result = dict(list(final_result.items())[:top_k])
        return top_k_final_result
