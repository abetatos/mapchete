from mapchete import RandChete, SeqChete, MaxChete

class MatcheteFarm():
    
    def get(crop_type):
        assert crop_type in {"random", "sequential", "maxchete"}
        
        if crop_type == "random": 
            return RandChete
        if crop_type == "sequential":
            return SeqChete
        if crop_type == "maxchete":
            return MaxChete
        