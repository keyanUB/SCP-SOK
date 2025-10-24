import json
import numpy as np
from typing import List, Dict
import pandas as pd
import torch

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
import nltk

# Download required NLTK data
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

class SCPMetricsEvaluator:
    def __init__(self):
        """Initialize the evaluator with required models and scorers."""
        # Prefer GPU if available for big speedups
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        # Load SBERT on the selected device
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.torch_device)
        self.smoothing = SmoothingFunction()
        
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score."""
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()
        # BLEU-4 with smoothing
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=self.smoothing.method1)
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L F1 score."""
        scores = self.rouge_scorer.score(reference, candidate)
        return scores['rougeL'].fmeasure
    
    def calculate_meteor(self, reference: str, candidate: str) -> float:
        """Calculate METEOR score."""
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()
        return meteor_score([reference_tokens], candidate_tokens)
    
    def calculate_bertscore_batch(self, references: List[str], candidates: List[str]) -> np.ndarray:
        """Calculate BERTScore F1 for a batch."""
        # bert_score returns torch tensors; compute on GPU if available
        _, _, F1 = bert_score(
            candidates, references, lang="en", verbose=False, device=self.torch_device
        )
        return F1.detach().cpu().numpy()
    
    def calculate_sbert_cosine_batch(self, references: List[str], candidates: List[str]) -> np.ndarray:
        """Calculate SBERT cosine similarity for a batch using diagonal of pairwise matrix."""
        # Encode refs and cands in batches; keep as tensors on device
        ref_emb = self.sbert_model.encode(
            references, batch_size=64, convert_to_tensor=True, show_progress_bar=False
        )
        cand_emb = self.sbert_model.encode(
            candidates, batch_size=64, convert_to_tensor=True, show_progress_bar=False
        )
        # NxN matrix; take diagonal to match aligned pairs
        sim_matrix = util.cos_sim(ref_emb, cand_emb)  # (N, N)
        diag = sim_matrix.diagonal()                  # (N,)
        return diag.detach().cpu().numpy()
    
    def calculate_key_concept_coverage(
        self, 
        key_concepts: List[str], 
        candidate: str, 
        threshold: float = 0.7
    ) -> float:
        """Calculate semantic key concept coverage."""
        if not key_concepts:
            return 1.0  # If no key concepts defined, return perfect score
        
        # Encode key concepts
        concept_embeddings = self.sbert_model.encode(
            key_concepts, batch_size=64, convert_to_tensor=True, show_progress_bar=False
        )
        
        # Split candidate into sentences
        candidate_sentences = [s.strip() for s in candidate.split('.') if s.strip()]
        if not candidate_sentences:
            return 0.0
        
        sentence_embeddings = self.sbert_model.encode(
            candidate_sentences, batch_size=64, convert_to_tensor=True, show_progress_bar=False
        )
        
        # Check coverage for each concept
        concepts_covered = 0
        for concept_emb in concept_embeddings:
            # Similarity with all sentences; take max per concept
            similarities = util.cos_sim(concept_emb, sentence_embeddings)  # (1, num_sent)
            max_similarity = similarities.max().item()
            if max_similarity >= threshold:
                concepts_covered += 1
        
        coverage_score = concepts_covered / len(key_concepts)
        return coverage_score
    
    def evaluate_batch(self, data: List[Dict]) -> pd.DataFrame:
        """
        Evaluate multiple examples efficiently with batching for BERT-based metrics.
        
        Args:
            data: List of dicts with keys:
                - 'id', 'reference', 'candidate'
                - optional: 'key_concepts'
        
        Returns:
            DataFrame with all metrics (scores only; no metadata)
        """
        # Extract aligned lists for batchable metrics
        ids = [item.get('id', 'unknown') for item in data]
        references = [item['reference'] for item in data]
        candidates = [item['candidate'] for item in data]
        key_concepts_list = [item.get('key_concepts', None) for item in data]
        
        # --- Batch the heavy stuff first ---
        print("Computing BERTScore (batched)...")
        bert_f1 = self.calculate_bertscore_batch(references, candidates)  # (N,)
        
        print("Computing SBERT cosine similarities (batched)...")
        sbert_cos = self.calculate_sbert_cosine_batch(references, candidates)  # (N,)
        
        # --- Lightweight per-item metrics ---
        print("Computing BLEU / ROUGE-L / METEOR (loop)...")
        bleu_scores = []
        rouge_l_scores = []
        meteor_scores = []
        key_cov_scores = []
        
        for i, (ref, cand, key_concepts) in enumerate(zip(references, candidates, key_concepts_list), start=1):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(references)}...")
            bleu_scores.append(self.calculate_bleu(ref, cand))
            rouge_l_scores.append(self.calculate_rouge_l(ref, cand))
            meteor_scores.append(self.calculate_meteor(ref, cand))
            
            if key_concepts:
                key_cov_scores.append(self.calculate_key_concept_coverage(key_concepts, cand))
            else:
                key_cov_scores.append(None)
        
        # Build results DataFrame
        results = {
            'ID': ids,
            'BLEU': bleu_scores,
            'ROUGE-L': rouge_l_scores,
            'METEOR': meteor_scores,
            'BERTScore': bert_f1,
            'SBERT_Cosine': sbert_cos,
        }
        # Only include Key_Concept_Coverage if any entry has it
        if any(k is not None for k in key_concepts_list):
            results['Key_Concept_Coverage'] = key_cov_scores
        
        df = pd.DataFrame(results)
        # Reorder columns
        cols = ['ID', 'BLEU', 'ROUGE-L', 'METEOR', 'BERTScore', 'SBERT_Cosine']
        if 'Key_Concept_Coverage' in df.columns:
            cols.append('Key_Concept_Coverage')
        df = df[cols]
        return df
    
    @staticmethod
    def interpret_score(metric_name: str, score: float) -> str:
        """Interpret a single score based on metric type."""
        if metric_name == 'BLEU':
            if score < 0.3: return 'Poor'
            elif score < 0.5: return 'Fair'
            elif score < 0.7: return 'Good'
            else: return 'Excellent'
        
        elif metric_name in ['ROUGE-L', 'METEOR']:
            if score < 0.2: return 'Poor'
            elif score < 0.4: return 'Fair'
            elif score < 0.6: return 'Good'
            else: return 'Very Good'
        
        elif metric_name == 'BERTScore':
            if score < 0.7: return 'Poor'
            elif score < 0.8: return 'Fair'
            elif score < 0.9: return 'Good'
            else: return 'Very Good'
        
        elif metric_name == 'SBERT_Cosine':
            if score < 0.5: return 'Poor'
            elif score < 0.7: return 'Fair'
            elif score < 0.9: return 'Good'
            else: return 'Excellent'
        
        elif metric_name == 'Key_Concept_Coverage':
            if score < 0.4: return 'Poor'
            elif score < 0.6: return 'Fair'
            elif score < 0.8: return 'Good'
            else: return 'Excellent'
        
        return 'Unknown'

def load_data_from_json(json_file: str) -> List[Dict]:
    """Load evaluation data from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data = [row for row in data if row["question_type"] == "long_answer"]
    return data


def main():
    """Main execution function."""
    
    # Load your data
    json_file = 'evaluation_data.json'  # Replace with your file path
    data = load_data_from_json(json_file)
    
    # Initialize evaluator
    evaluator = SCPMetricsEvaluator()
    
    # Evaluate
    print("Starting evaluation...")
    results_df = evaluator.evaluate_batch(data)
    
    # Display results table (scores only)
    print("\n" + "="*80)
    print("EVALUATION RESULTS (SCORES)")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Calculate and display average scores
    print("\n" + "="*80)
    print("AVERAGE SCORES")
    print("="*80)
    numeric_cols = ['BLEU', 'ROUGE-L', 'METEOR', 'BERTScore', 'SBERT_Cosine']
    if 'Key_Concept_Coverage' in results_df.columns:
        numeric_cols.append('Key_Concept_Coverage')
    
    avg_scores = results_df[numeric_cols].mean()
    for metric, score in avg_scores.items():
        interpretation = evaluator.interpret_score(metric, score)
        print(f"{metric:25s}: {score:.4f} ({interpretation})")
    
    # --- Save combined results (metadata + scores) ---
    # Convert original items to metadata DataFrame (safe subset if keys are missing)
    meta_cols = ['id', 'question', 'reference', 'question_type', 'candidate']
    # Ensure missing keys don't break things
    safe_data = [{k: item.get(k, None) for k in meta_cols} for item in data]
    meta_df = pd.DataFrame(safe_data)
    
    # Align by order; results_df['ID'] corresponds to meta_df['id']
    results_full_df = pd.concat([meta_df, results_df], axis=1)
    results_full_df.to_csv('scp_evaluation_results.csv', index=False)
    
    print(f"\n{'='*80}")
    print("Results saved to: scp_evaluation_results.csv (includes metadata columns)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
