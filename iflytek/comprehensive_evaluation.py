#!/usr/bin/env python3
"""
Comprehensive Transcription Quality Evaluation

This script provides multiple evaluation metrics to better assess transcription quality:
1. Traditional WER/CER
2. Normalized WER (removing filler words, standardizing)
3. Manual sample review with detailed diff visualization
"""

import pandas as pd
import numpy as np
import re
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import difflib
import json
from datetime import datetime
import logging

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available. Semantic similarity analysis will be skipped.")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  NLTK not available. BLEU score analysis will be skipped.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTextNormalizer:
    """Advanced text normalization for more fair comparison"""
    
    # Common filler words and variations
    FILLER_WORDS = {
        'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 'sort of', 'kind of',
        'actually', 'basically', 'literally', 'obviously', 'definitely', 'probably',
        'maybe', 'perhaps', 'anyway', 'so', 'well', 'right', 'okay', 'ok', 'alright',
        'yeah', 'yes', 'yep', 'yup', 'no', 'nope', 'hmm', 'mhm', 'aha', 'oh'
    }
    
    # Word equivalencies
    WORD_EQUIVALENCIES = {
        'ok': 'okay',
        'alright': 'all right',
        'gonna': 'going to',
        'wanna': 'want to',
        'gotta': 'got to',
        'kinda': 'kind of',
        'sorta': 'sort of',
        'dunno': "don't know",
        'yeah': 'yes',
        'yep': 'yes',
        'yup': 'yes',
        'nope': 'no',
        'uh-huh': 'yes',
        'mm-hmm': 'yes',
        'uh-uh': 'no'
    }
    
    @staticmethod
    def basic_normalize(text: str) -> str:
        """Basic normalization"""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text).strip().lower()
        
        # Remove speaker labels
        text = re.sub(r'^(?:Speaker\s*)?([A-Za-z]+(?:\s*Agent)?|MysteryShop(?:per)?):[\s]*', '', text, flags=re.IGNORECASE)
        
        # Remove timestamps
        text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?', '', text)
        
        # Remove brackets and parentheses content
        text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
        
        # Normalize punctuation
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def advanced_normalize(text: str, remove_fillers: bool = True, apply_equivalencies: bool = True) -> str:
        """Advanced normalization with filler removal and word equivalencies"""
        text = AdvancedTextNormalizer.basic_normalize(text)
        
        if not text:
            return ""
        
        words = text.split()
        
        # Apply word equivalencies
        if apply_equivalencies:
            words = [AdvancedTextNormalizer.WORD_EQUIVALENCIES.get(word, word) for word in words]
        
        # Remove filler words
        if remove_fillers:
            words = [word for word in words if word not in AdvancedTextNormalizer.FILLER_WORDS]
        
        return ' '.join(words)
    
    @staticmethod
    def tokenize_mixed_language(text: str) -> List[str]:
        """Tokenize mixed English-Chinese text"""
        normalized = AdvancedTextNormalizer.basic_normalize(text)
        if not normalized:
            return []
        
        tokens = []
        for word in normalized.split():
            if re.search(r'[\u4e00-\u9fff]', word):
                # Split Chinese characters individually
                current_token = ""
                for char in word:
                    if re.match(r'[\u4e00-\u9fff]', char):
                        if current_token:
                            tokens.append(current_token)
                            current_token = ""
                        tokens.append(char)
                    else:
                        current_token += char
                if current_token:
                    tokens.append(current_token)
            else:
                tokens.append(word)
        
        return tokens

class ComprehensiveEvaluator:
    """Comprehensive evaluation with multiple metrics"""
    
    def __init__(self):
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer model for semantic similarity")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.sentence_model = None
    
    def calculate_traditional_wer(self, reference: List[str], hypothesis: List[str]) -> Dict:
        """Traditional WER calculation"""
        if not reference:
            return {'wer': 0.0 if not hypothesis else float('inf'), 'details': 'Empty reference'}
        
        ref_len = len(reference)
        hyp_len = len(hypothesis)
        
        # DP matrix for edit distance
        dp = np.zeros((ref_len + 1, hyp_len + 1), dtype=int)
        
        for i in range(ref_len + 1):
            dp[i][0] = i
        for j in range(hyp_len + 1):
            dp[0][j] = j
        
        for i in range(1, ref_len + 1):
            for j in range(1, hyp_len + 1):
                if reference[i-1].lower() == hypothesis[j-1].lower():
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        total_errors = dp[ref_len][hyp_len]
        wer = total_errors / ref_len
        
        return {
            'wer': wer,
            'accuracy': 1 - wer,
            'total_errors': total_errors,
            'reference_length': ref_len,
            'hypothesis_length': hyp_len
        }
    
    def calculate_semantic_similarity(self, reference: str, hypothesis: str) -> Optional[float]:
        """Calculate semantic similarity using sentence embeddings"""
        if not self.sentence_model or not reference.strip() or not hypothesis.strip():
            return None
        
        try:
            ref_embedding = self.sentence_model.encode([reference])
            hyp_embedding = self.sentence_model.encode([hypothesis])
            
            # Cosine similarity
            similarity = np.dot(ref_embedding[0], hyp_embedding[0]) / (
                np.linalg.norm(ref_embedding[0]) * np.linalg.norm(hyp_embedding[0])
            )
            
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return None
    
    def calculate_bleu_score(self, reference: str, hypothesis: str) -> Optional[float]:
        """Calculate BLEU score"""
        if not NLTK_AVAILABLE or not reference.strip() or not hypothesis.strip():
            return None
        
        try:
            ref_tokens = reference.split()
            hyp_tokens = hypothesis.split()
            
            if not ref_tokens or not hyp_tokens:
                return 0.0
            
            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
            return float(bleu)
        except Exception as e:
            logger.warning(f"Error calculating BLEU score: {e}")
            return None
    
    def generate_detailed_diff(self, reference: str, hypothesis: str, max_length: int = 200) -> str:
        """Generate detailed diff visualization"""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        diff = list(difflib.unified_diff(
            ref_words, hyp_words,
            fromfile='Ground Truth',
            tofile='Transcription',
            lineterm='',
            n=3
        ))
        
        if len(diff) <= 4:  # No differences
            return "✅ Perfect match"
        
        # Truncate if too long
        diff_text = '\n'.join(diff[4:])  # Skip header lines
        if len(diff_text) > max_length:
            diff_text = diff_text[:max_length] + "... [truncated]"
        
        return diff_text
    
    def evaluate_transcription_pair(self, reference: str, hypothesis: str) -> Dict:
        """Comprehensive evaluation of a reference-hypothesis pair"""
        results = {
            'reference_length': len(reference),
            'hypothesis_length': len(hypothesis),
            'metrics': {}
        }
        
        # Basic normalization
        ref_basic = AdvancedTextNormalizer.basic_normalize(reference)
        hyp_basic = AdvancedTextNormalizer.basic_normalize(hypothesis)
        
        # Advanced normalization
        ref_advanced = AdvancedTextNormalizer.advanced_normalize(reference)
        hyp_advanced = AdvancedTextNormalizer.advanced_normalize(hypothesis)
        
        # Tokenize for WER calculation
        ref_tokens_basic = AdvancedTextNormalizer.tokenize_mixed_language(ref_basic)
        hyp_tokens_basic = AdvancedTextNormalizer.tokenize_mixed_language(hyp_basic)
        
        ref_tokens_advanced = AdvancedTextNormalizer.tokenize_mixed_language(ref_advanced)
        hyp_tokens_advanced = AdvancedTextNormalizer.tokenize_mixed_language(hyp_advanced)
        
        # Traditional WER
        wer_basic = self.calculate_traditional_wer(ref_tokens_basic, hyp_tokens_basic)
        results['metrics']['wer_basic'] = wer_basic
        
        # Normalized WER (with filler removal and equivalencies)
        wer_normalized = self.calculate_traditional_wer(ref_tokens_advanced, hyp_tokens_advanced)
        results['metrics']['wer_normalized'] = wer_normalized
        
        # Character Error Rate (CER)
        ref_chars = list(ref_basic.replace(' ', ''))
        hyp_chars = list(hyp_basic.replace(' ', ''))
        cer_result = self.calculate_traditional_wer(ref_chars, hyp_chars)
        results['metrics']['cer'] = cer_result
        
        # Semantic similarity
        semantic_sim = self.calculate_semantic_similarity(ref_basic, hyp_basic)
        if semantic_sim is not None:
            results['metrics']['semantic_similarity'] = semantic_sim
        
        # BLEU score
        bleu_score = self.calculate_bleu_score(ref_basic, hyp_basic)
        if bleu_score is not None:
            results['metrics']['bleu_score'] = bleu_score
        
        # Detailed diff
        results['diff'] = self.generate_detailed_diff(ref_basic, hyp_basic)
        
        return results

class DatasetEvaluator:
    """Evaluate entire datasets"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.evaluator = ComprehensiveEvaluator()
    
    def load_dataset_files(self, dataset_name: str) -> Dict[str, Tuple[pd.DataFrame, str]]:
        """Load all files for a dataset"""
        dataset_dir = self.base_dir / dataset_name
        files = {}
        
        if not dataset_dir.exists():
            return files
        
        # Find files
        for file in dataset_dir.iterdir():
            if file.suffix in ['.xlsx', '.xls']:
                if 'groundtruth' in file.name.lower() and 'unchunked' in file.name.lower():
                    df = pd.read_excel(file)
                    text_col = self._find_text_column(df)
                    if text_col:
                        full_text = " ".join([str(row[text_col]) for _, row in df.iterrows() 
                                            if pd.notna(row[text_col]) and str(row[text_col]).strip()])
                        files['ground_truth'] = (df, full_text)
                elif '11lab' in file.name.lower():
                    df = pd.read_excel(file)
                    text_col = self._find_text_column(df)
                    if text_col:
                        full_text = " ".join([str(row[text_col]) for _, row in df.iterrows() 
                                            if pd.notna(row[text_col]) and str(row[text_col]).strip()])
                        files['11labs'] = (df, full_text)
            elif file.suffix == '.csv' and 'iflytek' in file.name.lower():
                df = pd.read_csv(file)
                text_col = self._find_text_column(df)
                if text_col:
                    full_text = " ".join([str(row[text_col]) for _, row in df.iterrows() 
                                        if pd.notna(row[text_col]) and str(row[text_col]).strip()])
                    files['iflytek'] = (df, full_text)
        
        return files
    
    def _find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the text column in a dataframe"""
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['text', 'transcript', 'content', 'utterance']):
                return col
        return None
    
    def evaluate_dataset(self, dataset_name: str) -> Dict:
        """Evaluate a complete dataset"""
        logger.info(f"Evaluating dataset: {dataset_name}")
        
        files = self.load_dataset_files(dataset_name)
        
        if 'ground_truth' not in files:
            logger.error(f"No ground truth found for {dataset_name}")
            return {}
        
        ground_truth_df, ground_truth_text = files['ground_truth']
        
        results = {
            'dataset': dataset_name,
            'ground_truth_segments': len(ground_truth_df),
            'ground_truth_length': len(ground_truth_text),
            'services': {}
        }
        
        # Evaluate each service
        for service_name in ['iflytek', '11labs']:
            if service_name in files:
                service_df, service_text = files[service_name]
                
                logger.info(f"Evaluating {service_name} for {dataset_name}")
                
                evaluation = self.evaluator.evaluate_transcription_pair(ground_truth_text, service_text)
                
                results['services'][service_name] = {
                    'segments': len(service_df),
                    'transcript_length': len(service_text),
                    'evaluation': evaluation
                }
        
        return results
    
    def generate_comprehensive_report(self, results: Dict, output_file: str):
        """Generate comprehensive evaluation report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Transcription Quality Evaluation\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Evaluation Methodology\n\n")
            f.write("This evaluation uses multiple metrics to provide a more complete picture:\n\n")
            f.write("1. **Traditional WER**: Standard word error rate with basic normalization\n")
            f.write("2. **Normalized WER**: WER after removing filler words and applying word equivalencies\n")
            f.write("3. **Character Error Rate (CER)**: Character-level comparison\n")
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                f.write("4. **Semantic Similarity**: Measures meaning preservation using sentence embeddings\n")
            if NLTK_AVAILABLE:
                f.write("5. **BLEU Score**: Measures fluency and n-gram overlap\n")
            f.write("\n")
            
            for dataset_name, dataset_results in results.items():
                f.write(f"## Dataset: {dataset_name}\n\n")
                f.write(f"- Ground truth segments: {dataset_results.get('ground_truth_segments', 0)}\n")
                f.write(f"- Ground truth length: {dataset_results.get('ground_truth_length', 0)} characters\n\n")
                
                for service_name, service_results in dataset_results.get('services', {}).items():
                    f.write(f"### {service_name.upper()}\n\n")
                    f.write(f"- Segments: {service_results['segments']}\n")
                    f.write(f"- Length: {service_results['transcript_length']} characters\n\n")
                    
                    eval_results = service_results['evaluation']
                    metrics = eval_results['metrics']
                    
                    f.write("#### Metrics Summary\n\n")
                    
                    # Traditional WER
                    wer_basic = metrics['wer_basic']
                    f.write(f"**Traditional WER**: {wer_basic['wer']:.4f} ({wer_basic['wer']*100:.2f}%)\n")
                    f.write(f"- Accuracy: {wer_basic['accuracy']:.4f} ({wer_basic['accuracy']*100:.2f}%)\n")
                    f.write(f"- Errors: {wer_basic['total_errors']}\n\n")
                    
                    # Normalized WER
                    wer_norm = metrics['wer_normalized']
                    f.write(f"**Normalized WER** (fillers removed): {wer_norm['wer']:.4f} ({wer_norm['wer']*100:.2f}%)\n")
                    f.write(f"- Accuracy: {wer_norm['accuracy']:.4f} ({wer_norm['accuracy']*100:.2f}%)\n")
                    f.write(f"- Errors: {wer_norm['total_errors']}\n\n")
                    
                    # CER
                    cer = metrics['cer']
                    f.write(f"**Character Error Rate**: {cer['wer']:.4f} ({cer['wer']*100:.2f}%)\n\n")
                    
                    # Semantic similarity
                    if 'semantic_similarity' in metrics:
                        sem_sim = metrics['semantic_similarity']
                        f.write(f"**Semantic Similarity**: {sem_sim:.4f} ({sem_sim*100:.2f}%)\n\n")
                    
                    # BLEU score
                    if 'bleu_score' in metrics:
                        bleu = metrics['bleu_score']
                        f.write(f"**BLEU Score**: {bleu:.4f} ({bleu*100:.2f}%)\n\n")
                    
                    f.write("#### Sample Differences\n\n")
                    f.write("```diff\n")
                    f.write(eval_results['diff'])
                    f.write("\n```\n\n")
                    
                    f.write("---\n\n")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive transcription quality evaluation')
    parser.add_argument('base_dir', help='Base directory containing transcription results')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to analyze')
    parser.add_argument('--output-report', '-r', default='comprehensive_evaluation_report.md', 
                       help='Output report file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    evaluator = DatasetEvaluator(args.base_dir)
    
    # Determine datasets
    base_path = Path(args.base_dir)
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = [d.name for d in base_path.iterdir() if d.is_dir()]
    
    logger.info(f"Evaluating datasets: {datasets}")
    
    # Evaluate datasets
    all_results = {}
    for dataset in datasets:
        results = evaluator.evaluate_dataset(dataset)
        if results:
            all_results[dataset] = results
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE TRANSCRIPTION QUALITY EVALUATION")
    print("="*80)
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n📊 Dataset: {dataset_name}")
        
        for service_name, service_results in dataset_results.get('services', {}).items():
            eval_results = service_results['evaluation']['metrics']
            
            print(f"\n   🔍 {service_name.upper()}:")
            print(f"      Traditional WER: {eval_results['wer_basic']['wer']:.4f} ({eval_results['wer_basic']['wer']*100:.2f}%)")
            print(f"      Normalized WER:  {eval_results['wer_normalized']['wer']:.4f} ({eval_results['wer_normalized']['wer']*100:.2f}%)")
            print(f"      CER:            {eval_results['cer']['wer']:.4f} ({eval_results['cer']['wer']*100:.2f}%)")
            
            if 'semantic_similarity' in eval_results:
                print(f"      Semantic Sim:   {eval_results['semantic_similarity']:.4f} ({eval_results['semantic_similarity']*100:.2f}%)")
            
            if 'bleu_score' in eval_results:
                print(f"      BLEU Score:     {eval_results['bleu_score']:.4f} ({eval_results['bleu_score']*100:.2f}%)")
    
    # Generate report
    evaluator.generate_comprehensive_report(all_results, args.output_report)
    print(f"\n📄 Comprehensive report saved to: {args.output_report}")
    print("\n✅ Comprehensive evaluation complete!")

if __name__ == "__main__":
    main()
