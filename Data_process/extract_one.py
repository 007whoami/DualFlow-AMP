import pandas as pd
import numpy as np
from pathlib import Path
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class Data_Process(object):
    def __init__(self, max_len, device="cpu", model_path=None, USE_FP16=False):
        self.max_len = max_len
        self.device = device
        self.model_path = model_path
        self.USE_FP16 = USE_FP16
        
        if self.model_path:
            print(f"Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            if self.USE_FP16:
                self.model = self.model.half()
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Model loaded.")
        else:
            self.tokenizer = None
            self.model = None
            print("Warning: No model path provided. Embedding features will not be available.")

    @staticmethod
    def extract_amp_physchem_features(seq: str) -> np.ndarray:
        """
        0. Sequence_Length
        1. Net_Charge
        2. Charge_Density
        3. GRAVY
        4. Hydro_Moment
        5. Aromatic_Ratio
        6. Proline_Ratio
        7. Glycine_Ratio
        """
        HYDROPATHY = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }  
        CHARGE = {
            'R': 1.0, 'K': 1.0, 'H': 0.1,
            'D': -1.0, 'E': -1.0,
            'A': 0.0, 'N': 0.0, 'C': 0.0, 'Q': 0.0, 'G': 0.0,
            'I': 0.0, 'L': 0.0, 'M': 0.0, 'F': 0.0, 'P': 0.0,
            'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0
        }
        seq = seq.upper()
        length = len(seq)
        if length == 0:
            return np.zeros(8, dtype=np.float32)
        counts = {aa: seq.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
        
        feat_len = float(length)
        net_charge = sum(CHARGE[aa] * counts[aa] for aa in counts)
        feat_charge_density = net_charge / feat_len if feat_len > 0 else 0.0

        hydro_vals = [HYDROPATHY.get(aa, 0.0) for aa in seq]
        feat_gravy = float(np.mean(hydro_vals))
        
        rad_angle = np.radians(100.0)
        sum_sin = sum_cos = 0.0
        for n, aa in enumerate(seq):
            h = HYDROPATHY.get(aa, 0.0)
            theta = n * rad_angle
            sum_sin += h * np.sin(theta)
            sum_cos += h * np.cos(theta)
        feat_hydro_moment = float(np.sqrt(sum_sin**2 + sum_cos**2) / feat_len) if feat_len > 0 else 0.0

        feat_aromatic = float((counts['F'] + counts['Y'] + counts['W']) / feat_len)
        feat_proline = float(counts['P'] / feat_len)
        feat_glycine = float(counts['G'] / feat_len)

        return np.array([
            feat_len,
            float(net_charge),
            feat_charge_density,
            feat_gravy,
            feat_hydro_moment,
            feat_aromatic,
            feat_proline,
            feat_glycine
        ], dtype=np.float32)
    
    def get_embedding(self, seq: str):
        if self.model is None:
            raise ValueError("Model not initialized. Please provide model_path in __init__.")
            
        # Tokenize
        encoded = self.tokenizer( #type:ignore
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        embeddings = outputs.last_hidden_state.float().cpu()[0] 
        attention_mask = encoded["attention_mask"].cpu()[0]
        
        valid_embeds = embeddings[attention_mask == 1]
        
        final_embed = torch.zeros((self.max_len, embeddings.shape[1]), dtype=torch.float32)
        actual_len = min(valid_embeds.shape[0], self.max_len)
        final_embed[:actual_len] = valid_embeds[:actual_len]
        
        return final_embed

    def run(self, seq):
        print(f"Processing sequence: {seq}")
        
        physchem_feat = self.extract_amp_physchem_features(seq)
        
        if self.model is not None:
            embedding_feat = self.get_embedding(seq)
        else:
            embedding_feat = None
            
        return physchem_feat, embedding_feat

if __name__ == "__main__":
    MAX_LEN = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "/home/tree/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c"
    
    data_processer = Data_Process(
        max_len=MAX_LEN,
        device=DEVICE,
        model_path=MODEL_PATH,
        USE_FP16=False
    )
    
    target_seq = "RGGRLCYCRRRFCVCVGR"
    physchem, embedding = data_processer.run(target_seq)
    np.save("/home/tree/Work_area/AMP_Design_Github/single/phychem", physchem)
    torch.save(embedding, "/home/tree/Work_area/AMP_Design_Github/single/embedding.pt")
    print(physchem)
    