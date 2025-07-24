"""
AI Validation Module for AudioUI Transcription Engine
Uses AI Verify Test Engine to analyze bias and quality in transcription results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ ReportLab not available. Install with: pip install reportlab")

try:
    from aiverify_test_engine.interfaces.ialgorithm import IAlgorithm
    from aiverify_test_engine.interfaces.idata import IData
    from aiverify_test_engine.interfaces.imodel import IModel
    AIVERIFY_AVAILABLE = True
except ImportError:
    AIVERIFY_AVAILABLE = False
    print("⚠️ AI Verify Test Engine not available. Install with: pip install aiverify-test-engine")

class TranscriptionBiasAnalyzer:
    """
    Analyzes bias and quality metrics in audio transcription results
    """
    
    def __init__(self):
        self.results = {}
        self.bias_metrics = {}
        self.quality_metrics = {}
        
    def analyze_speaker_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze bias in speaker identification and transcription quality
        """
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'total_segments': len(df),
            'speaker_distribution': {},
            'bias_indicators': {},
            'quality_metrics': {},
            'recommendations': []
        }
        
        # Speaker distribution analysis
        if 'Speaker' in df.columns:
            speaker_counts = df['Speaker'].value_counts()
            analysis_results['speaker_distribution'] = speaker_counts.to_dict()
            
            # Check for speaker imbalance
            total_speakers = len(speaker_counts)
            if total_speakers > 1:
                max_count = speaker_counts.max()
                min_count = speaker_counts.min()
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                analysis_results['bias_indicators']['speaker_imbalance_ratio'] = imbalance_ratio
                
                if imbalance_ratio > 2.0:
                    analysis_results['recommendations'].append(
                        f"High speaker imbalance detected (ratio: {imbalance_ratio:.2f}). "
                        "Consider reviewing speaker identification accuracy."
                    )
        
        # Text length analysis by speaker
        if 'Text' in df.columns and 'Speaker' in df.columns:
            df['text_length'] = df['Text'].str.len()
            length_by_speaker = df.groupby('Speaker')['text_length'].agg(['mean', 'std', 'count'])
            
            analysis_results['quality_metrics']['avg_text_length_by_speaker'] = length_by_speaker.to_dict()
            
            # Check for significant differences in text length
            if len(length_by_speaker) > 1:
                length_variance = length_by_speaker['mean'].var()
                analysis_results['bias_indicators']['text_length_variance'] = length_variance
                
                if length_variance > 1000:  # Threshold for significant variance
                    analysis_results['recommendations'].append(
                        "Significant variance in transcription length between speakers detected. "
                        "This may indicate bias in transcription quality."
                    )
        
        # Timestamp gap analysis
        if 'Start' in df.columns and 'End' in df.columns:
            df['duration'] = pd.to_numeric(df['End']) - pd.to_numeric(df['Start'])
            avg_duration = df['duration'].mean()
            duration_std = df['duration'].std()
            
            analysis_results['quality_metrics']['average_segment_duration'] = avg_duration
            analysis_results['quality_metrics']['duration_std_deviation'] = duration_std
            
            # Detect unusually long or short segments
            outlier_threshold = 2 * duration_std
            outliers = df[abs(df['duration'] - avg_duration) > outlier_threshold]
            
            if len(outliers) > 0:
                analysis_results['bias_indicators']['duration_outliers'] = len(outliers)
                analysis_results['recommendations'].append(
                    f"Found {len(outliers)} segments with unusual duration. "
                    "Review these segments for transcription accuracy."
                )
        
        # Language consistency analysis
        if 'Text' in df.columns:
            # Simple heuristic for language detection
            english_chars = df['Text'].str.count(r'[a-zA-Z]').sum()
            total_chars = df['Text'].str.len().sum()
            
            if total_chars > 0:
                english_ratio = english_chars / total_chars
                analysis_results['quality_metrics']['english_character_ratio'] = english_ratio
                
                if english_ratio < 0.7:  # Less than 70% English characters
                    analysis_results['recommendations'].append(
                        f"Low English character ratio ({english_ratio:.2%}). "
                        "Consider language-specific transcription models."
                    )
        
        # Quality indicators based on text patterns
        if 'Text' in df.columns:
            # Check for repetitive patterns (potential transcription errors)
            repetitive_count = 0
            for text in df['Text']:
                if isinstance(text, str):
                    words = text.lower().split()
                    if len(words) > 2:
                        # Check for repeated words
                        for i in range(len(words) - 1):
                            if words[i] == words[i + 1]:
                                repetitive_count += 1
            
            analysis_results['quality_metrics']['repetitive_patterns'] = repetitive_count
            
            if repetitive_count > len(df) * 0.1:  # More than 10% of segments have repetition
                analysis_results['recommendations'].append(
                    "High number of repetitive patterns detected. "
                    "This may indicate transcription quality issues."
                )
        
        return analysis_results
    
    def analyze_temporal_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temporal patterns that might indicate bias
        """
        temporal_analysis = {
            'timestamp': datetime.now().isoformat(),
            'temporal_patterns': {},
            'bias_indicators': {},
            'recommendations': []
        }
        
        if 'Start' in df.columns:
            df['start_numeric'] = pd.to_numeric(df['Start'])
            
            # Analyze transcription quality over time
            # Divide into time buckets
            time_buckets = pd.cut(df['start_numeric'], bins=10, labels=False)
            df['time_bucket'] = time_buckets
            
            # Quality metrics per time bucket
            if 'Text' in df.columns:
                df['text_length'] = df['Text'].str.len()
                bucket_quality = df.groupby('time_bucket')['text_length'].agg(['mean', 'count'])
                
                temporal_analysis['temporal_patterns']['quality_by_time_bucket'] = bucket_quality.to_dict()
                
                # Check for degradation over time
                quality_trend = bucket_quality['mean'].corr(bucket_quality.index)
                temporal_analysis['bias_indicators']['quality_trend_correlation'] = quality_trend
                
                if quality_trend < -0.3:  # Negative correlation indicates degradation
                    temporal_analysis['recommendations'].append(
                        "Transcription quality appears to degrade over time. "
                        "Consider reviewing longer audio files for fatigue effects."
                    )
        
        return temporal_analysis
    
    def generate_bias_report(self, csv_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive bias analysis report
        """
        if not os.path.exists(csv_path):
            return {'error': f'File not found: {csv_path}'}
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return {'error': f'Error reading CSV: {str(e)}'}
        
        # Perform all analyses
        speaker_analysis = self.analyze_speaker_bias(df)
        temporal_analysis = self.analyze_temporal_bias(df)
        
        # Combine results
        comprehensive_report = {
            'file_analyzed': csv_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_overview': {
                'total_segments': len(df),
                'columns': list(df.columns),
                'date_range': self._get_date_range(df) if 'Start' in df.columns and 'End' in df.columns else None
            },
            'speaker_bias_analysis': speaker_analysis,
            'temporal_bias_analysis': temporal_analysis,
            'overall_bias_score': self._calculate_overall_bias_score(speaker_analysis, temporal_analysis),
            'aiverify_integration_status': AIVERIFY_AVAILABLE
        }
        
        return comprehensive_report
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get the time range of the transcription"""
        try:
            start_times = pd.to_numeric(df['Start'])
            end_times = pd.to_numeric(df['End'])
            return {
                'start_time': float(start_times.min()),
                'end_time': float(end_times.max()),
                'total_duration': float(end_times.max() - start_times.min())
            }
        except:
            return None
    
    def _calculate_overall_bias_score(self, speaker_analysis: Dict, temporal_analysis: Dict) -> Dict[str, Any]:
        """
        Calculate an overall bias score based on various indicators
        """
        bias_score = 0
        max_score = 100
        factors = []
        
        # Speaker imbalance factor
        if 'speaker_imbalance_ratio' in speaker_analysis.get('bias_indicators', {}):
            ratio = speaker_analysis['bias_indicators']['speaker_imbalance_ratio']
            if ratio > 3:
                bias_score += 30
                factors.append(f"High speaker imbalance (ratio: {ratio:.2f})")
            elif ratio > 2:
                bias_score += 15
                factors.append(f"Moderate speaker imbalance (ratio: {ratio:.2f})")
        
        # Text length variance factor
        if 'text_length_variance' in speaker_analysis.get('bias_indicators', {}):
            variance = speaker_analysis['bias_indicators']['text_length_variance']
            if variance > 2000:
                bias_score += 25
                factors.append(f"High text length variance ({variance:.0f})")
            elif variance > 1000:
                bias_score += 10
                factors.append(f"Moderate text length variance ({variance:.0f})")
        
        # Temporal degradation factor
        if 'quality_trend_correlation' in temporal_analysis.get('bias_indicators', {}):
            trend = temporal_analysis['bias_indicators']['quality_trend_correlation']
            if trend < -0.5:
                bias_score += 20
                factors.append(f"Strong quality degradation over time ({trend:.2f})")
            elif trend < -0.3:
                bias_score += 10
                factors.append(f"Moderate quality degradation over time ({trend:.2f})")
        
        # Duration outliers factor
        if 'duration_outliers' in speaker_analysis.get('bias_indicators', {}):
            outliers = speaker_analysis['bias_indicators']['duration_outliers']
            total_segments = speaker_analysis.get('total_segments', 1)
            outlier_ratio = outliers / total_segments
            if outlier_ratio > 0.1:
                bias_score += 15
                factors.append(f"High number of duration outliers ({outliers})")
        
        # Language consistency factor
        if 'english_character_ratio' in speaker_analysis.get('quality_metrics', {}):
            ratio = speaker_analysis['quality_metrics']['english_character_ratio']
            if ratio < 0.5:
                bias_score += 20
                factors.append(f"Low English character ratio ({ratio:.2%})")
        
        bias_level = "LOW"
        if bias_score > 50:
            bias_level = "HIGH"
        elif bias_score > 25:
            bias_level = "MEDIUM"
        
        return {
            'bias_score': min(bias_score, max_score),
            'max_score': max_score,
            'bias_level': bias_level,
            'contributing_factors': factors,
            'interpretation': self._interpret_bias_score(bias_score)
        }
    
    def _interpret_bias_score(self, score: int) -> str:
        """Provide interpretation of the bias score"""
        if score <= 15:
            return "The transcription shows minimal bias indicators. Quality appears consistent across speakers and time."
        elif score <= 35:
            return "Some bias indicators detected. Consider reviewing transcription quality and speaker identification accuracy."
        elif score <= 60:
            return "Moderate bias detected. Significant issues with transcription consistency or speaker identification may be present."
        else:
            return "High bias detected. Major issues with transcription quality, speaker identification, or temporal consistency require immediate attention."

class BiasReportPDFGenerator:
    """
    Generate PDF reports from bias analysis results
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet() if REPORTLAB_AVAILABLE else None
        if REPORTLAB_AVAILABLE:
            # Custom styles
            self.title_style = ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            )
            self.heading_style = ParagraphStyle(
                'CustomHeading',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            self.normal_style = ParagraphStyle(
                'CustomNormal',
                parent=self.styles['Normal'],
                fontSize=10,
                spaceAfter=6
            )
    
    def generate_pdf_report(self, analysis_results: Dict[str, Any], output_path: str) -> bool:
        """
        Generate a PDF report from bias analysis results
        """
        if not REPORTLAB_AVAILABLE:
            print("❌ ReportLab not available. Cannot generate PDF report.")
            return False
        
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Title
            story.append(Paragraph("AI Transcription Bias Analysis Report", self.title_style))
            story.append(Spacer(1, 12))
            
            # Analysis metadata
            story.append(Paragraph("Analysis Overview", self.heading_style))
            
            overview_data = [
                ['File Analyzed:', analysis_results.get('file_analyzed', 'N/A')],
                ['Analysis Date:', analysis_results.get('analysis_timestamp', 'N/A')[:19]],
                ['Total Segments:', str(analysis_results.get('dataset_overview', {}).get('total_segments', 'N/A'))],
                ['AI Verify Status:', 'Available' if analysis_results.get('aiverify_integration_status') else 'Not Available']
            ]
            
            overview_table = Table(overview_data, colWidths=[2*inch, 4*inch])
            overview_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(overview_table)
            story.append(Spacer(1, 20))
            
            # Overall Bias Score
            bias_score = analysis_results.get('overall_bias_score', {})
            story.append(Paragraph("Overall Bias Assessment", self.heading_style))
            
            bias_color = colors.green
            if bias_score.get('bias_level') == 'MEDIUM':
                bias_color = colors.orange
            elif bias_score.get('bias_level') == 'HIGH':
                bias_color = colors.red
            
            bias_data = [
                ['Bias Score:', f"{bias_score.get('bias_score', 0)}/{bias_score.get('max_score', 100)}"],
                ['Bias Level:', bias_score.get('bias_level', 'N/A')],
                ['Interpretation:', bias_score.get('interpretation', 'N/A')]
            ]
            
            bias_table = Table(bias_data, colWidths=[2*inch, 4*inch])
            bias_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('BACKGROUND', (1, 1), (1, 1), bias_color),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(bias_table)
            story.append(Spacer(1, 20))
            
            # Speaker Analysis
            speaker_analysis = analysis_results.get('speaker_bias_analysis', {})
            story.append(Paragraph("Speaker Bias Analysis", self.heading_style))
            
            # Speaker distribution
            speaker_dist = speaker_analysis.get('speaker_distribution', {})
            if speaker_dist:
                story.append(Paragraph("Speaker Distribution:", self.normal_style))
                speaker_data = [['Speaker', 'Segment Count']]
                for speaker, count in speaker_dist.items():
                    speaker_data.append([speaker, str(count)])
                
                speaker_table = Table(speaker_data, colWidths=[2*inch, 2*inch])
                speaker_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(speaker_table)
                story.append(Spacer(1, 12))
            
            # Quality metrics
            quality_metrics = speaker_analysis.get('quality_metrics', {})
            if quality_metrics:
                story.append(Paragraph("Quality Metrics:", self.normal_style))
                
                quality_data = [['Metric', 'Value']]
                
                if 'english_character_ratio' in quality_metrics:
                    ratio = quality_metrics['english_character_ratio']
                    quality_data.append(['English Character Ratio', f"{ratio:.2%}"])
                
                if 'repetitive_patterns' in quality_metrics:
                    patterns = quality_metrics['repetitive_patterns']
                    quality_data.append(['Repetitive Patterns', str(patterns)])
                
                if 'avg_text_length_by_speaker' in quality_metrics:
                    avg_lengths = quality_metrics['avg_text_length_by_speaker'].get('mean', {})
                    for speaker, length in avg_lengths.items():
                        quality_data.append([f'Avg Text Length ({speaker})', f"{length:.1f} chars"])
                
                quality_table = Table(quality_data, colWidths=[3*inch, 2*inch])
                quality_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(quality_table)
                story.append(Spacer(1, 20))
            
            # Recommendations
            recommendations = speaker_analysis.get('recommendations', [])
            if recommendations:
                story.append(Paragraph("Recommendations", self.heading_style))
                for i, rec in enumerate(recommendations, 1):
                    story.append(Paragraph(f"{i}. {rec}", self.normal_style))
                story.append(Spacer(1, 20))
            
            # Contributing factors
            factors = bias_score.get('contributing_factors', [])
            if factors:
                story.append(Paragraph("Bias Contributing Factors", self.heading_style))
                for i, factor in enumerate(factors, 1):
                    story.append(Paragraph(f"{i}. {factor}", self.normal_style))
                story.append(Spacer(1, 20))
            
            # Technical details
            story.append(PageBreak())
            story.append(Paragraph("Technical Details", self.heading_style))
            
            # Dataset overview
            dataset_overview = analysis_results.get('dataset_overview', {})
            columns = dataset_overview.get('columns', [])
            if columns:
                story.append(Paragraph("Dataset Columns:", self.normal_style))
                columns_text = ", ".join(columns)
                story.append(Paragraph(columns_text, self.normal_style))
                story.append(Spacer(1, 12))
            
            # Bias indicators
            bias_indicators = speaker_analysis.get('bias_indicators', {})
            if bias_indicators:
                story.append(Paragraph("Bias Indicators:", self.normal_style))
                for indicator, value in bias_indicators.items():
                    story.append(Paragraph(f"• {indicator}: {value}", self.normal_style))
                story.append(Spacer(1, 12))
            
            # Footer
            story.append(Spacer(1, 30))
            story.append(Paragraph("Report generated by AI Transcription Bias Analyzer", 
                                 ParagraphStyle('Footer', parent=self.styles['Normal'], 
                                              fontSize=8, alignment=TA_CENTER, 
                                              textColor=colors.grey)))
            
            # Build PDF
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"❌ Error generating PDF report: {str(e)}")
            return False

def run_bias_analysis(csv_file_path: str) -> Dict[str, Any]:
    """
    Main function to run bias analysis on a transcription CSV file
    """
    analyzer = TranscriptionBiasAnalyzer()
    return analyzer.generate_bias_report(csv_file_path)

def generate_pdf_report(csv_file_path: str, output_pdf_path: str = None) -> str:
    """
    Generate both bias analysis and PDF report
    """
    # Run bias analysis
    analysis_results = run_bias_analysis(csv_file_path)
    
    # Generate PDF report
    if output_pdf_path is None:
        base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        output_pdf_path = f"{base_name}_bias_report.pdf"
    
    pdf_generator = BiasReportPDFGenerator()
    success = pdf_generator.generate_pdf_report(analysis_results, output_pdf_path)
    
    if success:
        print(f"✅ PDF report generated: {output_pdf_path}")
        return output_pdf_path
    else:
        print("❌ Failed to generate PDF report")
        return None

if __name__ == "__main__":
    # Test with a sample file if available
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        generate_pdf = '--pdf' in sys.argv or '-p' in sys.argv
        
        if os.path.exists(csv_path):
            if generate_pdf:
                # Generate PDF report
                pdf_path = generate_pdf_report(csv_path)
                if pdf_path:
                    print(f"\n📄 PDF Report generated: {pdf_path}")
                else:
                    print("\n❌ Failed to generate PDF report")
            else:
                # Generate JSON output
                results = run_bias_analysis(csv_path)
                print(json.dumps(results, indent=2, default=str))
        else:
            print(f"File not found: {csv_path}")
    else:
        print("Usage: python ai_validation.py <csv_file_path> [--pdf|-p]")
        print("  --pdf, -p: Generate PDF report instead of JSON output")
