from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    @classmethod
    def from_list(cls, bbox: List) -> 'BoundingBox':
        """Create BoundingBox from list format"""
        if len(bbox) == 4:
            return cls(x_min=bbox[0], y_min=bbox[1], x_max=bbox[2], y_max=bbox[3])
        elif len(bbox) >= 4 and isinstance(bbox[0], (list, tuple)):
            # Handle [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            return cls(
                x_min=min(x_coords), 
                y_min=min(y_coords), 
                x_max=max(x_coords), 
                y_max=max(y_coords)
            )
        else:
            return cls(x_min=0, y_min=0, x_max=0, y_max=0)
    
    def to_list(self) -> List[float]:
        """Convert to list format"""
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclass
class TextElement:
    """Text element in layout"""
    bounding_box: List[float]
    text: str
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "text",
            "bounding_box": self.bounding_box,
            "text": self.text,
            "confidence": self.confidence
        }


@dataclass
class FigureElement:
    """Figure element in layout"""
    figure_id: str
    bounding_box: List[float]
    figure_caption: Optional[str] = None
    related_chapter: Optional[int] = None
    related_section: Optional[str] = None
    figure_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "figure",
            "figure_id": self.figure_id,
            "bounding_box": self.bounding_box,
            "figure_caption": self.figure_caption,
            "related_chapter": self.related_chapter,
            "related_section": self.related_section,
            "figure_number": self.figure_number
        }


@dataclass
class FigureReference:
    """Figure reference in text"""
    referenced_figure_id: Optional[str]
    bounding_box: List[float]
    reference_text: str
    figure_number: Optional[int] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "figure_reference",
            "referenced_figure_id": self.referenced_figure_id,
            "bounding_box": self.bounding_box,
            "reference_text": self.reference_text,
            "figure_number": self.figure_number,
            "confidence": self.confidence
        }


@dataclass
class PageInfo:
    """Page information with layouts"""
    page_index: int
    chapter: Optional[str] = None
    section: Optional[str] = None
    layouts: List[Dict[str, Any]] = field(default_factory=list)
    full_text: str = ""
    
    def add_text_element(self, bbox: List[float], text: str, confidence: Optional[float] = None):
        """Add text element to page"""
        element = TextElement(bbox, text, confidence)
        self.layouts.append(element.to_dict())
        self.full_text += text + " "
    
    def add_figure_element(self, figure_id: str, bbox: List[float], 
                          caption: Optional[str] = None, 
                          chapter: Optional[int] = None,
                          section: Optional[str] = None,
                          figure_number: Optional[int] = None):
        """Add figure element to page"""
        element = FigureElement(
            figure_id=figure_id,
            bounding_box=bbox,
            figure_caption=caption,
            related_chapter=chapter if chapter else self.chapter,
            related_section=section if section else self.section,
            figure_number=figure_number
        )
        self.layouts.append(element.to_dict())
    
    def add_figure_reference(self, ref_id: Optional[str], bbox: List[float], 
                           ref_text: str, figure_num: Optional[int] = None,
                           confidence: Optional[float] = None):
        """Add figure reference to page"""
        element = FigureReference(
            referenced_figure_id=ref_id,
            bounding_box=bbox,
            reference_text=ref_text,
            figure_number=figure_num,
            confidence=confidence
        )
        self.layouts.append(element.to_dict())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "page_index": self.page_index,
            "chapter": self.chapter,
            "section": self.section,
            "layouts": self.layouts,
            "full_text": self.full_text.strip()
        }


@dataclass
class ChapterInfo:
    """Chapter information"""
    title: str
    chapter_number: Optional[int] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    sections: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "chapter_number": self.chapter_number,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "sections": self.sections
        }


@dataclass
class DocumentInfo:
    """PDF document information"""
    title: Optional[str] = None
    total_pages: int = 0
    chapters: List[Dict[str, Any]] = field(default_factory=list)
    pages: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: Optional[float] = None
    
    def add_chapter(self, chapter_info: ChapterInfo):
        """Add chapter information"""
        self.chapters.append(chapter_info.to_dict())
    
    def add_page(self, page_info: PageInfo):
        """Add page information"""
        self.pages.append(page_info.to_dict())
    
    def add_figure(self, figure_info: Dict[str, Any]):
        """Add figure information"""
        self.figures.append(figure_info)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to frontend-friendly JSON structure"""
        return {
            "title": self.title,
            "chapters": self.chapters,
            "pages": self.pages,
            "figures": self.figures,
            "metadata": {
                "total_pages": self.total_pages,
                "processing_time": self.processing_time,
                "total_figures": len(self.figures)
            }
        }


def transform_analysis_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Transform raw analysis result to frontend-friendly format"""
    
    # Create document info
    doc_info = DocumentInfo(
        title=raw_result.get('pdf_info', {}).get('title'),
        total_pages=raw_result.get('total_pages', 0),
        processing_time=raw_result.get('processing_time')
    )
    
    # Process chapters from document structure
    chapters = raw_result.get('document_structure', {}).get('chapters', [])
    for chapter in chapters:
        chapter_info = ChapterInfo(
            title=chapter.get('title', f"Chapter {chapter.get('chapter', '')}"),
            chapter_number=chapter.get('chapter'),
            start_page=chapter.get('start_page'),
            end_page=chapter.get('end_page'),
            sections=chapter.get('sections', [])
        )
        doc_info.add_chapter(chapter_info)
    
    # Collect all figures first for the figures array
    all_figures = {}
    reference_counts = {}
    
    # Process pages to collect figures and count references
    pages = raw_result.get('pages', [])
    for page_data in pages:
        if page_data.get('status') == 'error':
            continue
            
        page_idx = page_data.get('page_index', 0)
        chapter_info = page_data.get('chapter_info', {})
        
        # Collect figures from this page
        figure_layouts = page_data.get('figure_layouts', [])
        for figure in figure_layouts:
            figure_id = figure.get('figure_id', f"fig_{page_idx}_{figure.get('id', '')}")
            bbox = figure.get('bbox', [])
            
            if bbox:
                # Calculate dimensions
                if len(bbox) >= 4:
                    if isinstance(bbox[0], (list, tuple)):
                        x_coords = [point[0] for point in bbox[:4]]
                        y_coords = [point[1] for point in bbox[:4]]
                        normalized_bbox = [
                            min(x_coords), min(y_coords),
                            max(x_coords), max(y_coords)
                        ]
                    else:
                        normalized_bbox = bbox[:4]
                    
                    width = normalized_bbox[2] - normalized_bbox[0]
                    height = normalized_bbox[3] - normalized_bbox[1]
                    area = width * height
                else:
                    normalized_bbox = bbox
                    width = height = area = None
                
                all_figures[figure_id] = {
                    "figure_id": figure_id,
                    "figure_number": figure.get('figure_number'),
                    "bounding_box": normalized_bbox,
                    "page_index": page_idx,
                    "chapter": figure.get('chapter'),
                    "section": figure.get('section'),
                    "caption": figure.get('caption'),
                    "caption_bbox": figure.get('caption_bbox'),
                    "dimensions": {
                        "width": width,
                        "height": height,
                        "area": area
                    } if width and height else None,
                    "confidence_score": figure.get('confidence'),
                    "figure_type": figure.get('figure_type'),  # Could be enhanced to detect type
                }
        
        # Count figure references
        figure_references = page_data.get('figure_references', [])
        for ref in figure_references:
            ref_figure_id = ref.get('mapped_figure_id')
            if ref_figure_id:
                reference_counts[ref_figure_id] = reference_counts.get(ref_figure_id, 0) + 1
    
    # Update figures with reference information
    for figure_id, figure_info in all_figures.items():
        ref_count = reference_counts.get(figure_id, 0)
        figure_info["reference_count"] = ref_count
        figure_info["is_referenced"] = ref_count > 0
        doc_info.add_figure(figure_info)
    
    # Now process pages for layout information
    for page_data in pages:
        if page_data.get('status') == 'error':
            continue
            
        page_idx = page_data.get('page_index', 0)
        chapter_info = page_data.get('chapter_info', {})
        
        # Create page info
        page_info = PageInfo(
            page_index=page_idx,
            chapter=str(chapter_info.get('chapter')) if chapter_info.get('chapter') else None,
            section=chapter_info.get('section')
        )
        
        # Add text elements
        recognized_texts = page_data.get('recognized_texts', [])
        for text_data in recognized_texts:
            bbox = text_data.get('bbox', [])
            text = text_data.get('text', '')
            confidence = text_data.get('confidence')
            
            if bbox and text:
                # Normalize bbox to [x_min, y_min, x_max, y_max] format
                if len(bbox) >= 4:
                    if isinstance(bbox[0], (list, tuple)):
                        # Convert from [[x1,y1], [x2,y2], ...] format
                        x_coords = [point[0] for point in bbox[:4]]
                        y_coords = [point[1] for point in bbox[:4]]
                        normalized_bbox = [
                            min(x_coords), min(y_coords),
                            max(x_coords), max(y_coords)
                        ]
                    else:
                        normalized_bbox = bbox[:4]
                    
                    page_info.add_text_element(normalized_bbox, text, confidence)
        
        # Add figure elements
        figure_layouts = page_data.get('figure_layouts', [])
        for figure in figure_layouts:
            figure_id = figure.get('figure_id', f"fig_{page_idx}_{figure.get('id', '')}")
            bbox = figure.get('bbox', [])
            
            if bbox:
                page_info.add_figure_element(
                    figure_id=figure_id,
                    bbox=bbox[:4] if len(bbox) >= 4 else bbox,
                    caption=figure.get('caption'),
                    chapter=figure.get('chapter'),
                    section=figure.get('section'),
                    figure_number=figure.get('figure_number')
                )
        
        # Add figure references
        figure_references = page_data.get('figure_references', [])
        for ref in figure_references:
            bbox = ref.get('bbox', [])
            ref_text = ref.get('text', '')
            
            if bbox and ref_text:
                # Normalize bbox
                if len(bbox) >= 4:
                    if isinstance(bbox[0], (list, tuple)):
                        x_coords = [point[0] for point in bbox[:4]]
                        y_coords = [point[1] for point in bbox[:4]]
                        normalized_bbox = [
                            min(x_coords), min(y_coords),
                            max(x_coords), max(y_coords)
                        ]
                    else:
                        normalized_bbox = bbox[:4]
                    
                    page_info.add_figure_reference(
                        ref_id=ref.get('mapped_figure_id'),
                        bbox=normalized_bbox,
                        ref_text=ref_text,
                        figure_num=ref.get('figure_number'),
                        confidence=ref.get('mapping_confidence')
                    )
        
        doc_info.add_page(page_info)
    
    return doc_info.to_dict()