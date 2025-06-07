#!/usr/bin/env python3
"""
Test script to demonstrate the frontend-friendly JSON output format
"""

import json
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent))

from modules.document_analyzer import DocumentAnalyzer
from modules.response_models import transform_analysis_result

def test_frontend_format():
    """Test the frontend format transformation"""
    
    # Sample raw analysis result (simplified)
    raw_result = {
        'status': 'success',
        'pdf_info': {
            'title': 'Sample Research Paper',
            'total_pages': 10
        },
        'total_pages': 10,
        'pages': [
            {
                'page_index': 0,
                'status': 'success',
                'chapter_info': {
                    'chapter': 1,
                    'section': None,
                    'title': 'Introduction'
                },
                'layouts': [
                    {'label': 'title', 'bbox': [100, 100, 500, 150]},
                    {'label': 'paragraph', 'bbox': [100, 200, 500, 400]}
                ],
                'figure_layouts': [
                    {
                        'id': 'fig_0_1',
                        'bbox': [100, 500, 400, 700],
                        'figure_id': 'ch1_fig1',
                        'figure_number': 1,
                        'caption': 'Figure 1. System Architecture',
                        'chapter': 1
                    }
                ],
                'recognized_texts': [
                    {'bbox': [100, 100, 500, 150], 'text': 'Chapter 1: Introduction', 'confidence': 0.98},
                    {'bbox': [100, 200, 500, 250], 'text': 'This paper presents a novel approach...', 'confidence': 0.95},
                    {'bbox': [100, 250, 500, 300], 'text': 'As shown in Figure 1, the system consists of...', 'confidence': 0.96}
                ],
                'figure_references': [
                    {
                        'text': 'Figure 1',
                        'bbox': [200, 250, 250, 270],
                        'figure_number': 1,
                        'mapped_figure_id': 'ch1_fig1',
                        'mapping_confidence': 0.9
                    }
                ]
            },
            {
                'page_index': 1,
                'status': 'success',
                'chapter_info': {
                    'chapter': 1,
                    'section': '1.1',
                    'title': 'Background'
                },
                'layouts': [
                    {'label': 'paragraph', 'bbox': [100, 100, 500, 300]}
                ],
                'figure_layouts': [],
                'recognized_texts': [
                    {'bbox': [100, 100, 500, 150], 'text': '1.1 Background', 'confidence': 0.97},
                    {'bbox': [100, 150, 500, 200], 'text': 'Previous work in this area includes...', 'confidence': 0.94},
                    {'bbox': [100, 200, 500, 250], 'text': 'See Figure 2 for comparison results.', 'confidence': 0.95}
                ],
                'figure_references': [
                    {
                        'text': 'Figure 2',
                        'bbox': [120, 200, 170, 220],
                        'figure_number': 2,
                        'mapped_figure_id': None,
                        'mapping_confidence': 0.0
                    }
                ]
            }
        ],
        'document_structure': {
            'chapters': [
                {
                    'chapter': 1,
                    'title': 'Introduction',
                    'start_page': 0,
                    'end_page': 3,
                    'sections': ['1.1', '1.2'],
                    'figure_count': 2
                }
            ],
            'figure_registry': {
                'ch1_fig1': {
                    'figure_id': 'ch1_fig1',
                    'figure_number': 1,
                    'chapter': 1,
                    'page_index': 0,
                    'caption': 'Figure 1. System Architecture'
                }
            }
        },
        'summary': {
            'total_layouts': 3,
            'total_figures': 1,
            'total_texts': 6,
            'total_figure_references': 2,
            'error_pages': 0,
            'success_rate': 100.0
        },
        'processing_time': 45.2
    }
    
    # Transform to frontend format
    frontend_result = transform_analysis_result(raw_result)
    
    # Print the result
    print("Frontend-Friendly JSON Format:")
    print("=" * 50)
    print(json.dumps(frontend_result, indent=2, ensure_ascii=False))
    
    # Demonstrate the structure
    print("\n\nStructure Overview:")
    print("=" * 50)
    print(f"Title: {frontend_result['title']}")
    print(f"Total Pages: {frontend_result['pages'][0]['page_index'] + 1} of {len(frontend_result['pages'])}")
    print(f"Chapters: {len(frontend_result['chapters'])}")
    
    print("\nFirst Page Layout Elements:")
    for layout in frontend_result['pages'][0]['layouts']:
        print(f"  - Type: {layout['type']}")
        if layout['type'] == 'text':
            print(f"    Text: {layout['text'][:50]}...")
        elif layout['type'] == 'figure':
            print(f"    Figure ID: {layout['figure_id']}")
            print(f"    Caption: {layout['figure_caption']}")
        elif layout['type'] == 'figure_reference':
            print(f"    Reference: {layout['reference_text']}")
            print(f"    Referenced Figure: {layout['referenced_figure_id']}")


if __name__ == '__main__':
    test_frontend_format()