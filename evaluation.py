import json
import numpy as np
from typing import List, Dict
import pandas as pd

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
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.smoothing = SmoothingFunction()
        
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score."""
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()
        
        # BLEU-4 with smoothing
        score = sentence_bleu(
            [reference_tokens], 
            candidate_tokens,
            smoothing_function=self.smoothing.method1
        )
        return score
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L F1 score."""
        scores = self.rouge_scorer.score(reference, candidate)
        return scores['rougeL'].fmeasure
    
    def calculate_meteor(self, reference: str, candidate: str) -> float:
        """Calculate METEOR score."""
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()
        score = meteor_score([reference_tokens], candidate_tokens)
        return score
    
    def calculate_bertscore(self, reference: str, candidate: str) -> float:
        """Calculate BERTScore F1."""
        P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
        return F1.item()
    
    def calculate_sbert_cosine(self, reference: str, candidate: str) -> float:
        """Calculate SBERT cosine similarity."""
        embeddings = self.sbert_model.encode([reference, candidate])
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return similarity.item()
    
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
        concept_embeddings = self.sbert_model.encode(key_concepts)
        
        # Split candidate into sentences
        candidate_sentences = [s.strip() for s in candidate.split('.') if s.strip()]
        if not candidate_sentences:
            return 0.0
        
        sentence_embeddings = self.sbert_model.encode(candidate_sentences)
        
        # Check coverage for each concept
        concepts_covered = 0
        for concept_emb in concept_embeddings:
            # Calculate similarity with all sentences
            similarities = util.cos_sim(concept_emb, sentence_embeddings)
            max_similarity = similarities.max().item()
            
            if max_similarity >= threshold:
                concepts_covered += 1
        
        coverage_score = concepts_covered / len(key_concepts)
        return coverage_score
    
    def evaluate_single(
        self, 
        reference: str, 
        candidate: str, 
        key_concepts: List[str] = None
    ) -> Dict[str, float]:
        """Evaluate a single reference-candidate pair."""
        results = {
            'BLEU': self.calculate_bleu(reference, candidate),
            'ROUGE-L': self.calculate_rouge_l(reference, candidate),
            'METEOR': self.calculate_meteor(reference, candidate),
            'BERTScore': self.calculate_bertscore(reference, candidate),
            'SBERT_Cosine': self.calculate_sbert_cosine(reference, candidate)
        }
        
        if key_concepts:
            results['Key_Concept_Coverage'] = self.calculate_key_concept_coverage(
                key_concepts, candidate
            )
        
        return results
    
    def evaluate_batch(self, data: List[Dict]) -> pd.DataFrame:
        """
        Evaluate multiple examples.
        
        Args:
            data: List of dictionaries with keys:
                - 'id': identifier for the example
                - 'reference': reference text
                - 'candidate': LLM output text
                - 'key_concepts': (optional) list of key concepts
        
        Returns:
            DataFrame with all metrics
        """
        results = []
        
        for item in data:
            item_id = item.get('id', 'unknown')
            reference = item['reference']
            candidate = item['candidate']
            key_concepts = item.get('key_concepts', None)
            
            print(f"Evaluating: {item_id}")
            
            scores = self.evaluate_single(reference, candidate, key_concepts)
            scores['ID'] = item_id
            results.append(scores)
        
        # Create DataFrame and reorder columns
        df = pd.DataFrame(results)
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
    
    # Save to CSV (scores only)
    results_df.to_csv('scp_evaluation_results.csv', index=False)
    print(f"\n{'='*80}")
    print("Results saved to: scp_evaluation_results.csv")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()