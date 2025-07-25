#!/usr/bin/env python3
"""
Transcription Comparison Report Generator

Creates a clean, presentable comparison between iFlytek and 11Labs transcription quality.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List, Dict
import argparse
from datetime import datetime
import tabulate

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).strip().lower()
    # Remove speaker labels
    text = re.sub(r'^(?:Speaker\s*)?([A-Za-z]+(?:\s*Agent)?|MysteryShop(?:per)?):[\s]*', '', text, flags=re.IGNORECASE)
    # Remove punctuation
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_aggressive(text: str) -> str:
    """Aggressive cleaning - remove fillers and normalize equivalents"""
    text = clean_text(text)
    
    # Remove common filler words
    fillers = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 'okay', 'ok', 'yeah', 'yes', 'no', 'well', 'so', 'right']
    words = text.split()
    words = [w for w in words if w not in fillers]
    
    # Apply equivalencies
    equivalents = {'ok': 'okay', 'yeah': 'yes', 'yep': 'yes', 'nope': 'no'}
    words = [equivalents.get(w, w) for w in words]
    
    return ' '.join(words)

def tokenize_mixed(text: str) -> List[str]:
    """Tokenize mixed English-Chinese text"""
    if not text:
        return []
    
    tokens = []
    for word in text.split():
        if re.search(r'[\u4e00-\u9fff]', word):
            # Split Chinese characters individually
            current = ""
            for char in word:
                if re.match(r'[\u4e00-\u9fff]', char):
                    if current:
                        tokens.append(current)
                        current = ""
                    tokens.append(char)
                else:
                    current += char
            if current:
                tokens.append(current)
        else:
            tokens.append(word)
    return tokens

def calculate_wer(reference: List[str], hypothesis: List[str]) -> Dict:
    """Calculate WER with detailed breakdown"""
    if not reference:
        return {'wer': 0.0, 'accuracy': 1.0, 'errors': 0, 'ref_len': 0}
    
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    
    # Dynamic programming for edit distance
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
    
    errors = dp[ref_len][hyp_len]
    wer = errors / ref_len
    
    return {
        'wer': wer,
        'accuracy': 1 - wer,
        'errors': errors,
        'ref_len': ref_len,
        'hyp_len': hyp_len
    }

def analyze_dataset(dataset_path: str):
    """Analyze a single dataset"""
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name
    
    # Find files
    ground_truth_file = None
    iflytek_file = None
    labs_11_file = None
    
    for file in dataset_path.iterdir():
        if file.suffix in ['.xlsx', '.xls']:
            if 'groundtruth' in file.name.lower() and 'unchunked' in file.name.lower():
                ground_truth_file = file
            elif '11lab' in file.name.lower():
                labs_11_file = file
        elif file.suffix == '.csv' and 'iflytek' in file.name.lower():
            iflytek_file = file
    
    if not ground_truth_file:
        return None
    
    # Load ground truth
    gt_df = pd.read_excel(ground_truth_file)
    gt_text_col = None
    for col in gt_df.columns:
        if any(keyword in col.lower() for keyword in ['text', 'transcript', 'content']):
            gt_text_col = col
            break
    
    if not gt_text_col:
        return None
    
    gt_full_text = " ".join([str(row[gt_text_col]) for _, row in gt_df.iterrows() 
                            if pd.notna(row[gt_text_col]) and str(row[gt_text_col]).strip()])
    
    results = {
        'name': dataset_name,
        'ground_truth': {
            'segments': len(gt_df),
            'characters': len(gt_full_text),
            'text': gt_full_text
        },
        'services': {}
    }
    
    # Analyze each service
    services = []
    if iflytek_file:
        services.append(('iflytek', iflytek_file, 'csv'))
    if labs_11_file:
        services.append(('11labs', labs_11_file, 'excel'))
    
    for service_name, service_file, file_type in services:
        # Load transcription
        if file_type == 'csv':
            trans_df = pd.read_csv(service_file)
        else:
            trans_df = pd.read_excel(service_file)
        
        trans_text_col = None
        for col in trans_df.columns:
            if any(keyword in col.lower() for keyword in ['text', 'transcript', 'content']):
                trans_text_col = col
                break
        
        if not trans_text_col:
            continue
        
        trans_full_text = " ".join([str(row[trans_text_col]) for _, row in trans_df.iterrows() 
                                   if pd.notna(row[trans_text_col]) and str(row[trans_text_col]).strip()])
        
        service_results = {
            'segments': len(trans_df),
            'characters': len(trans_full_text),
            'metrics': {}
        }
        
        # Different levels of analysis
        gt_clean = clean_text(gt_full_text)
        trans_clean = clean_text(trans_full_text)
        
        gt_aggressive = clean_text_aggressive(gt_full_text)
        trans_aggressive = clean_text_aggressive(trans_full_text)
        
        # Tokenize
        gt_tokens = tokenize_mixed(gt_clean)
        trans_tokens = tokenize_mixed(trans_clean)
        
        gt_tokens_aggressive = tokenize_mixed(gt_aggressive)
        trans_tokens_aggressive = tokenize_mixed(trans_aggressive)
        
        # Calculate WERs
        basic_wer = calculate_wer(gt_tokens, trans_tokens)
        aggressive_wer = calculate_wer(gt_tokens_aggressive, trans_tokens_aggressive)
        
        # Character-level (CER)
        gt_chars = list(gt_clean.replace(' ', ''))
        trans_chars = list(trans_clean.replace(' ', ''))
        cer = calculate_wer(gt_chars, trans_chars)
        
        service_results['metrics'] = {
            'basic': basic_wer,
            'aggressive': aggressive_wer,
            'cer': cer
        }
        
        results['services'][service_name] = service_results
    
    return results

def generate_comparison_report(results_list, output_path):
    """Generate comparison report between services"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Transcription Quality Comparison: iFlytek vs 11Labs\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall table
        f.write("## Overall Performance Summary\n\n")
        
        overall_table = []
        for results in results_list:
            dataset_name = results['name']
            
            for service_name, service_results in results['services'].items():
                basic_wer = service_results['metrics']['basic']['wer']
                basic_acc = service_results['metrics']['basic']['accuracy']
                aggressive_wer = service_results['metrics']['aggressive']['wer']
                aggressive_acc = service_results['metrics']['aggressive']['accuracy']
                cer = service_results['metrics']['cer']['wer']
                
                overall_table.append([
                    dataset_name, 
                    service_name.upper(), 
                    f"{basic_wer:.2%}", 
                    f"{basic_acc:.2%}",
                    f"{aggressive_wer:.2%}", 
                    f"{aggressive_acc:.2%}",
                    f"{cer:.2%}"
                ])
        
        headers = [
            "Dataset", 
            "Service", 
            "WER", 
            "Accuracy",
            "WER (Aggressive)", 
            "Accuracy (Aggressive)",
            "CER"
        ]
        
        f.write(tabulate.tabulate(overall_table, headers=headers, tablefmt="pipe"))
        f.write("\n\n")
        
        # Detailed dataset comparisons
        for results in results_list:
            dataset_name = results['name']
            f.write(f"## Dataset: {dataset_name}\n\n")
            
            gt_info = results['ground_truth']
            f.write(f"- Ground truth: {gt_info['segments']} segments, {gt_info['characters']} characters\n\n")
            
            # Side-by-side metrics
            if 'iflytek' in results['services'] and '11labs' in results['services']:
                iflytek = results['services']['iflytek']
                labs_11 = results['services']['11labs']
                
                comparison_table = []
                
                # Basic info
                comparison_table.append(["Segments", iflytek['segments'], labs_11['segments']])
                comparison_table.append(["Characters", iflytek['characters'], labs_11['characters']])
                
                # WER metrics
                comparison_table.append(["WER", f"{iflytek['metrics']['basic']['wer']:.2%}", f"{labs_11['metrics']['basic']['wer']:.2%}"])
                comparison_table.append(["Accuracy", f"{iflytek['metrics']['basic']['accuracy']:.2%}", f"{labs_11['metrics']['basic']['accuracy']:.2%}"])
                
                # Aggressive normalization
                comparison_table.append(["WER (Aggressive)", f"{iflytek['metrics']['aggressive']['wer']:.2%}", f"{labs_11['metrics']['aggressive']['wer']:.2%}"])
                comparison_table.append(["Accuracy (Aggressive)", f"{iflytek['metrics']['aggressive']['accuracy']:.2%}", f"{labs_11['metrics']['aggressive']['accuracy']:.2%}"])
                
                # Character-level
                comparison_table.append(["Character Error Rate", f"{iflytek['metrics']['cer']['wer']:.2%}", f"{labs_11['metrics']['cer']['wer']:.2%}"])
                
                f.write(tabulate.tabulate(comparison_table, headers=["Metric", "iFlytek", "11Labs"], tablefmt="pipe"))
                f.write("\n\n")
                
                # Winner indicators
                f.write("### Performance Winner\n\n")
                
                if iflytek['metrics']['basic']['accuracy'] > labs_11['metrics']['basic']['accuracy']:
                    winner = "iFlytek"
                    accuracy_diff = iflytek['metrics']['basic']['accuracy'] - labs_11['metrics']['basic']['accuracy']
                else:
                    winner = "11Labs"
                    accuracy_diff = labs_11['metrics']['basic']['accuracy'] - iflytek['metrics']['basic']['accuracy']
                
                f.write(f"- **{winner}** performs better on this dataset\n")
                f.write(f"- Accuracy advantage: **{accuracy_diff:.2%}**\n\n")
            
            f.write("---\n\n")
        
        # Methodology explanation
        f.write("## Methodology\n\n")
        f.write("### Metrics Explained\n\n")
        f.write("- **WER (Word Error Rate)**: Percentage of words that differ from ground truth\n")
        f.write("- **Accuracy**: Percentage of words correctly transcribed (1 - WER)\n")
        f.write("- **WER (Aggressive)**: WER after removing filler words and normalizing equivalent terms\n") 
        f.write("- **CER (Character Error Rate)**: Percentage of characters that differ from ground truth\n\n")
        
        f.write("### Normalization Levels\n\n")
        f.write("1. **Basic**: Lowercase, remove punctuation, normalize whitespace\n")
        f.write("2. **Aggressive**: Remove filler words (um, uh, like), normalize equivalent terms (ok→okay, yeah→yes)\n\n")
        
        f.write("### Why Accuracy Matters More Than WER\n\n")
        f.write("WER can be misleadingly high due to:\n")
        f.write("- Different handling of filler words\n")
        f.write("- Formatting and punctuation differences\n") 
        f.write("- Equivalent terms counted as errors\n")
        f.write("- Mixed language tokenization challenges\n\n")
        
        f.write("Higher accuracy percentage = better transcription quality\n")
        
def generate_console_report(results_list):
    """Generate a console-friendly comparison report"""
    print("\n" + "="*80)
    print(" TRANSCRIPTION QUALITY COMPARISON: IFLYTEK VS 11LABS ")
    print("="*80 + "\n")
    
    for results in results_list:
        dataset_name = results['name']
        print(f"\n📊 Dataset: {dataset_name}")
        print("-" * 60)
        
        gt_info = results['ground_truth']
        print(f"Ground Truth: {gt_info['segments']} segments, {gt_info['characters']} characters")
        
        # Side-by-side metrics
        if 'iflytek' in results['services'] and '11labs' in results['services']:
            iflytek = results['services']['iflytek']
            labs_11 = results['services']['11labs']
            
            print("\n" + "-" * 60)
            print(f"{'Metric':<25} {'iFlytek':<15} {'11Labs':<15}")
            print("-" * 60)
            
            # Basic info
            print(f"{'Segments':<25} {iflytek['segments']:<15} {labs_11['segments']:<15}")
            print(f"{'Characters':<25} {iflytek['characters']:<15} {labs_11['characters']:<15}")
            
            # WER metrics
            print(f"{'WER':<25} {iflytek['metrics']['basic']['wer']:.2%:<15} {labs_11['metrics']['basic']['wer']:.2%:<15}")
            print(f"{'Accuracy':<25} {iflytek['metrics']['basic']['accuracy']:.2%:<15} {labs_11['metrics']['basic']['accuracy']:.2%:<15}")
            
            # Aggressive normalization
            print(f"{'WER (Aggressive)':<25} {iflytek['metrics']['aggressive']['wer']:.2%:<15} {labs_11['metrics']['aggressive']['wer']:.2%:<15}")
            print(f"{'Accuracy (Aggressive)':<25} {iflytek['metrics']['aggressive']['accuracy']:.2%:<15} {labs_11['metrics']['aggressive']['accuracy']:.2%:<15}")
            
            # Character-level
            print(f"{'Character Error Rate':<25} {iflytek['metrics']['cer']['wer']:.2%:<15} {labs_11['metrics']['cer']['wer']:.2%:<15}")
            
            # Winner indicators
            print("\n🏆 Performance Winner:")
            
            if iflytek['metrics']['basic']['accuracy'] > labs_11['metrics']['basic']['accuracy']:
                winner = "iFlytek"
                accuracy_diff = iflytek['metrics']['basic']['accuracy'] - labs_11['metrics']['basic']['accuracy']
            else:
                winner = "11Labs"
                accuracy_diff = labs_11['metrics']['basic']['accuracy'] - iflytek['metrics']['basic']['accuracy']
            
            print(f"- {winner} performs better (accuracy advantage: {accuracy_diff:.2%})")
    
    print("\n" + "="*80)
    print(" METHODOLOGY ")
    print("="*80)
    print("\n📏 Metrics Explained:")
    print("- WER: Percentage of words that differ from ground truth")
    print("- Accuracy: Percentage of words correctly transcribed (1 - WER)")
    print("- WER (Aggressive): WER after removing fillers and normalizing equivalents") 
    print("- CER: Percentage of characters that differ from ground truth")
    
    print("\n💡 Why Accuracy May Appear Low:")
    print("- WER counts any word difference as a full error")
    print("- Mixed language content increases complexity")
    print("- Filler words and formatting inflate error rates")
    print("- Transcription may preserve meaning despite word differences")

def main():
    parser = argparse.ArgumentParser(description='Generate transcription comparison report')
    parser.add_argument('base_dir', help='Base directory containing transcription results')
    parser.add_argument('--output', '-o', default='comparison_report.md', 
                        help='Output report file (markdown format)')
    
    args = parser.parse_args()
    
    base_path = Path(args.base_dir)
    results_list = []
    
    # Analyze each dataset
    for dataset_dir in base_path.iterdir():
        if dataset_dir.is_dir():
            results = analyze_dataset(dataset_dir)
            if results:
                results_list.append(results)
    
    # Generate reports
    generate_comparison_report(results_list, args.output)
    generate_console_report(results_list)
    
    print(f"\n✅ Comparison report saved to: {args.output}")

if __name__ == "__main__":
    main()
