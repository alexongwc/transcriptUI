"""
Consolidated Report Generator for 11Labs Transcription Analysis
Generates a single PDF report analyzing all transcription files in a directory
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from ai_validation import TranscriptionBiasAnalyzer, BiasReportPDFGenerator
import warnings
warnings.filterwarnings('ignore')

# Check for ReportLab
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ ReportLab not available. Install with: pip install reportlab")

class ConsolidatedReportGenerator:
    """
    Generate a consolidated report for multiple transcription files
    """
    
    def __init__(self):
        self.analyzer = TranscriptionBiasAnalyzer()
        self.pdf_generator = BiasReportPDFGenerator() if REPORTLAB_AVAILABLE else None
        self.styles = getSampleStyleSheet() if REPORTLAB_AVAILABLE else None
        
        # Common speaker prefixes/patterns to detect in text
        self.speaker_patterns = [
            r'^([A-Za-z]+):', 
            r'^(SPEAKER_[a-z_0-9]+):', 
            r'^([A-Z_]+):', 
            r'^(Speaker\s*\d+):',
            r'^(Person\s*[A-Za-z0-9]+):'
        ]
        
        if REPORTLAB_AVAILABLE:
            self.title_style = ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            )
            self.heading_style = ParagraphStyle(
                'CustomHeading',
                parent=self.styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            self.subheading_style = ParagraphStyle(
                'CustomSubHeading',
                parent=self.styles['Heading3'],
                fontSize=14,
                spaceAfter=10,
                textColor=colors.navy
            )
            self.normal_style = ParagraphStyle(
                'CustomNormal',
                parent=self.styles['Normal'],
                fontSize=10,
                spaceAfter=6
            )
    
    def check_for_embedded_speakers(self, text_series):
        """
        Check if the text column contains embedded speaker information
        """
        # Sample the first 10 entries or all if fewer
        sample_size = min(10, len(text_series))
        sample = text_series.head(sample_size)
        
        # Check for common speaker patterns in the sample
        for pattern in self.speaker_patterns:
            matches = sample.str.extract(pattern, expand=False).dropna()
            if len(matches) > 0:
                return True
        
        return False
    
    def extract_speakers_from_text(self, text_series):
        """
        Extract speaker information from text
        Returns two lists: speakers and cleaned texts
        """
        speakers = []
        cleaned_texts = []
        
        for text in text_series:
            text = str(text).strip()
            speaker = "Unknown"
            cleaned_text = text
            
            # Try each pattern to extract speaker
            for pattern in self.speaker_patterns:
                import re
                match = re.match(pattern, text)
                if match:
                    speaker = match.group(1)
                    # Remove the speaker prefix from the text
                    cleaned_text = re.sub(f"^{re.escape(speaker)}:\\s*", "", text).strip()
                    break
            
            speakers.append(speaker)
            cleaned_texts.append(cleaned_text)
        
        return speakers, cleaned_texts
    
    def process_excel_file(self, file_path):
        """
        Process an Excel file and extract transcript data
        """
        try:
            df = pd.read_excel(file_path)
            
            # Check for required columns - looking for common patterns in transcription files
            if 'Combined Text' in df.columns:
                # This seems to be a chunked file
                text_column = 'Combined Text'
                # Check if speaker info is embedded in the text
                has_embedded_speakers = self.check_for_embedded_speakers(df[text_column])
                has_speaker_info = has_embedded_speakers
            elif 'Text' in df.columns:
                # This might be a raw transcription file with speaker info
                text_column = 'Text'
                has_explicit_speaker = 'Speaker' in df.columns
                has_embedded_speakers = self.check_for_embedded_speakers(df[text_column])
                has_speaker_info = has_explicit_speaker or has_embedded_speakers
            else:
                # Try to find any column that might contain text
                text_columns = [col for col in df.columns if any(s in col.lower() for s in ['text', 'transcript'])]
                if text_columns:
                    text_column = text_columns[0]
                    has_explicit_speaker = any(s in str(col).lower() for col in df.columns for s in ['speaker', 'person'])
                    has_embedded_speakers = self.check_for_embedded_speakers(df[text_column])
                    has_speaker_info = has_explicit_speaker or has_embedded_speakers
                else:
                    print(f"⚠️ Warning: Could not identify text column in {os.path.basename(file_path)}")
                    return None
            
            # Determine basic metrics
            file_metrics = {
                'filename': os.path.basename(file_path),
                'total_segments': len(df),
                'has_speaker_info': has_speaker_info,
                'columns': list(df.columns),
                'word_count': sum(len(str(text).split()) for text in df[text_column]) if text_column in df.columns else 0
            }
            
            # Run bias analysis
            # Create a modified DataFrame for bias analysis if needed
            if has_speaker_info:
                # If we have explicit speaker column, use it directly
                if 'Speaker' in df.columns:
                    analysis_df = df.copy()
                # Otherwise, extract speaker info from text
                elif has_embedded_speakers:
                    analysis_df = df.copy()
                    speakers, cleaned_texts = self.extract_speakers_from_text(df[text_column])
                    analysis_df['Speaker'] = speakers
                    analysis_df['Text'] = cleaned_texts
                
                # Convert to a temp CSV for bias analysis
                temp_csv = os.path.join(os.path.dirname(file_path), f"temp_{os.path.basename(file_path)}.csv")
                analysis_df.to_csv(temp_csv, index=False)
                
                try:
                    # Run bias analysis
                    bias_results = self.analyzer.generate_bias_report(temp_csv)
                    file_metrics['bias_analysis'] = bias_results
                except Exception as e:
                    print(f"⚠️ Error analyzing bias in {os.path.basename(file_path)}: {str(e)}")
                    file_metrics['bias_analysis'] = {"error": str(e)}
                
                # Clean up temp file
                try:
                    os.remove(temp_csv)
                except:
                    pass
            
            return file_metrics
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
            return {
                'filename': os.path.basename(file_path),
                'error': str(e)
            }
    
    def create_consolidated_report(self, directory_path, output_path=None):
        """
        Create a consolidated report for all transcription files in a directory
        """
        # Find all Excel files
        excel_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(('.xlsx', '.xls')) and 'merged' in file:
                    excel_files.append(os.path.join(root, file))
        
        if not excel_files:
            print(f"❌ No Excel files found in {directory_path}")
            return None
        
        print(f"📊 Found {len(excel_files)} Excel files for analysis")
        
        # Process each file
        results = []
        for i, file_path in enumerate(excel_files, 1):
            print(f"[{i}/{len(excel_files)}] Processing {os.path.basename(file_path)}...")
            file_results = self.process_excel_file(file_path)
            if file_results:
                results.append(file_results)
        
        # Generate consolidated PDF report
        if not output_path:
            output_path = os.path.join(directory_path, "11labs_transcription_analysis_report.pdf")
        
        success = self.generate_consolidated_pdf(results, output_path)
        
        if success:
            print(f"✅ Consolidated report generated: {output_path}")
            return output_path
        else:
            print("❌ Failed to generate consolidated report")
            return None
    
    def generate_consolidated_pdf(self, results, output_path):
        """
        Generate a consolidated PDF report
        """
        if not REPORTLAB_AVAILABLE:
            print("❌ ReportLab not available. Cannot generate PDF report.")
            return False
        
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Title and introduction
            story.append(Paragraph("11Labs Transcription Analysis Report", self.title_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", self.normal_style))
            story.append(Spacer(1, 12))
            
            # Overview
            story.append(Paragraph("Executive Summary", self.heading_style))
            story.append(Paragraph(f"This report analyzes {len(results)} transcription files processed by ElevenLabs API. It examines bias indicators, transcription quality, and speaker distribution patterns.", self.normal_style))
            story.append(Spacer(1, 12))
            
            # Summary statistics
            total_segments = sum(r.get('total_segments', 0) for r in results if 'error' not in r)
            total_words = sum(r.get('word_count', 0) for r in results if 'error' not in r)
            files_with_speaker_info = sum(1 for r in results if r.get('has_speaker_info', False))
            
            summary_data = [
                ["Total Files Analyzed:", str(len(results))],
                ["Total Segments:", str(total_segments)],
                ["Total Words:", str(total_words)],
                ["Files with Speaker Info:", f"{files_with_speaker_info} ({files_with_speaker_info/len(results)*100:.1f}%)"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Add overall bias assessment if available
            bias_scores = []
            for result in results:
                if 'bias_analysis' in result and 'overall_bias_score' in result['bias_analysis']:
                    bias_scores.append({
                        'filename': result['filename'],
                        'bias_score': result['bias_analysis']['overall_bias_score']['bias_score'],
                        'bias_level': result['bias_analysis']['overall_bias_score']['bias_level']
                    })
            
            if bias_scores:
                story.append(Paragraph("Overall Bias Assessment", self.heading_style))
                story.append(Paragraph(f"Bias was analyzed in {len(bias_scores)} of {len(results)} files.", self.normal_style))
                story.append(Spacer(1, 12))
                
                # Create bias table
                bias_table_data = [['Filename', 'Bias Score', 'Bias Level']]
                for score in bias_scores:
                    bias_table_data.append([
                        score['filename'],
                        str(score['bias_score']),
                        score['bias_level']
                    ])
                
                bias_table = Table(bias_table_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
                bias_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(bias_table)
                story.append(Spacer(1, 20))
            
            # Individual file analysis
            story.append(PageBreak())
            story.append(Paragraph("Detailed File Analysis", self.heading_style))
            
            for i, result in enumerate(results):
                # Add page break between files (except for the first one)
                if i > 0:
                    story.append(PageBreak())
                
                # File header
                story.append(Paragraph(f"File {i+1}: {result['filename']}", self.subheading_style))
                
                # Basic file info
                file_info = [
                    ["Segments:", str(result.get('total_segments', 'N/A'))],
                    ["Word Count:", str(result.get('word_count', 'N/A'))],
                    ["Speaker Info:", "Available" if result.get('has_speaker_info', False) else "Not Available"]
                ]
                
                file_table = Table(file_info, colWidths=[2*inch, 4*inch])
                file_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(file_table)
                story.append(Spacer(1, 12))
                
                # Bias analysis if available
                if 'bias_analysis' in result and 'error' not in result['bias_analysis']:
                    bias_analysis = result['bias_analysis']
                    
                    # Overall bias score
                    bias_score = bias_analysis.get('overall_bias_score', {})
                    if bias_score:
                        story.append(Paragraph("Bias Assessment", self.subheading_style))
                        
                        bias_data = [
                            ["Bias Score:", f"{bias_score.get('bias_score', 0)}/{bias_score.get('max_score', 100)}"],
                            ["Bias Level:", bias_score.get('bias_level', 'N/A')],
                            ["Interpretation:", bias_score.get('interpretation', 'N/A')]
                        ]
                        
                        bias_table = Table(bias_data, colWidths=[2*inch, 4*inch])
                        bias_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                            ('FONTSIZE', (0, 0), (-1, -1), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(bias_table)
                        story.append(Spacer(1, 12))
                    
                    # Speaker distribution
                    speaker_analysis = bias_analysis.get('speaker_bias_analysis', {})
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
                    
                    # Recommendations
                    recommendations = speaker_analysis.get('recommendations', [])
                    if recommendations:
                        story.append(Paragraph("Recommendations:", self.normal_style))
                        for j, rec in enumerate(recommendations, 1):
                            story.append(Paragraph(f"• {rec}", self.normal_style))
                        story.append(Spacer(1, 12))
                
                else:
                    # If there was an error or no bias analysis was performed
                    if 'bias_analysis' in result and 'error' in result['bias_analysis']:
                        story.append(Paragraph(f"Error during bias analysis: {result['bias_analysis']['error']}", self.normal_style))
                    elif 'error' in result:
                        story.append(Paragraph(f"Error processing file: {result['error']}", self.normal_style))
                    else:
                        story.append(Paragraph("No bias analysis was performed for this file.", self.normal_style))
                
                story.append(Spacer(1, 12))
            
            # Final summary and conclusions
            story.append(PageBreak())
            story.append(Paragraph("Conclusions and Recommendations", self.heading_style))
            
            # Calculate average bias score if available
            if bias_scores:
                avg_bias = sum(score['bias_score'] for score in bias_scores) / len(bias_scores)
                bias_levels = [score['bias_level'] for score in bias_scores]
                most_common_level = max(set(bias_levels), key=bias_levels.count)
                
                story.append(Paragraph("Bias Assessment Summary", self.subheading_style))
                story.append(Paragraph(f"Average Bias Score: {avg_bias:.2f}/100", self.normal_style))
                story.append(Paragraph(f"Most Common Bias Level: {most_common_level}", self.normal_style))
                story.append(Spacer(1, 12))
                
                # General recommendations based on bias analysis
                story.append(Paragraph("General Recommendations", self.subheading_style))
                
                if avg_bias < 20:
                    story.append(Paragraph("• The transcription quality is generally good with minimal bias indicators.", self.normal_style))
                    story.append(Paragraph("• Continue monitoring for consistency across different speakers and languages.", self.normal_style))
                elif avg_bias < 50:
                    story.append(Paragraph("• Moderate bias indicators were detected. Consider fine-tuning the transcription model for your specific use cases.", self.normal_style))
                    story.append(Paragraph("• Review speaker identification algorithms to ensure balanced treatment.", self.normal_style))
                else:
                    story.append(Paragraph("• Significant bias indicators were detected. Immediate attention is recommended.", self.normal_style))
                    story.append(Paragraph("• Consider alternative transcription services or major adjustments to your current setup.", self.normal_style))
                
                story.append(Spacer(1, 12))
            
            # Footer with timestamp
            story.append(Spacer(1, 30))
            footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • AI Verify Integration"
            story.append(Paragraph(footer_text, 
                                 ParagraphStyle('Footer', parent=self.styles['Normal'], 
                                              fontSize=8, alignment=TA_CENTER, 
                                              textColor=colors.grey)))
            
            # Build the PDF
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"❌ Error generating consolidated PDF report: {str(e)}")
            return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python consolidated_report.py <data_directory> [output_pdf_path]")
        return
    
    data_directory = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(data_directory) or not os.path.isdir(data_directory):
        print(f"❌ Directory not found: {data_directory}")
        return
    
    generator = ConsolidatedReportGenerator()
    output_file = generator.create_consolidated_report(data_directory, output_path)
    
    if output_file:
        print(f"\n📄 Consolidated report generated: {output_file}")
    else:
        print("\n❌ Failed to generate consolidated report")


if __name__ == "__main__":
    main()
