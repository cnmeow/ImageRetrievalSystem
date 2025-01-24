from src.models.clip import CLIPSearchEngine
#from src.models.blip import BLIPSearchEngine
from src.models.beit import BEITSearchEngine
from src.utils import merge_searching_results_by_addition

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
        self.blip_engine = None
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
