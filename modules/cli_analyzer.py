import click
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.syntax import Syntax
from rich import box

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.document_analyzer import DocumentAnalyzer
from modules.pdf_processor import PDFProcessor

console = Console()


class CLIAnalyzer:
    def __init__(self):
        self.document_analyzer = DocumentAnalyzer()
        self.pdf_processor = PDFProcessor()

    def analyze_pdf_with_progress(self, pdf_path: str, output_dir: str = None, frontend_format: bool = False) -> Dict[str, Any]:
        """진행 상황을 표시하면서 PDF 분석"""
        # PDF 정보 먼저 확인
        pdf_info = self.pdf_processor.get_pdf_info(pdf_path)
        total_pages = pdf_info['total_pages']

        # 레이아웃 설정
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=10),
            Layout(name="stats", size=15),
            Layout(name="logs", size=10)
        )

        # 헤더
        header_text = Text()
        header_text.append("📄 PDF Document Analyzer\n", style="bold blue")
        header_text.append(f"File: {Path(pdf_path).name}", style="cyan")
        layout["header"].update(Panel(header_text, box=box.ROUNDED))

        # 통계 테이블 초기화
        stats_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan", width=25)
        stats_table.add_column("Value", style="green", width=15)

        # 로그 메시지
        log_messages = []

        # 진행 상황 추적 변수
        progress_data = {
            'current_page': 0,
            'total_pages': total_pages,
            'figures_found': 0,
            'references_found': 0,
            'texts_recognized': 0,
            'start_time': datetime.now()
        }

        with Live(layout, refresh_per_second=4) as live:
            # Progress bar 설정
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                expand=True
            )

            task = progress.add_task("[cyan]Analyzing PDF...", total=total_pages)
            layout["progress"].update(Panel(progress, title="Progress", box=box.ROUNDED))

            # 분석 시작
            def update_progress(current, total):
                progress.update(task, completed=current)
                progress_data['current_page'] = current

                # 통계 업데이트
                elapsed = (datetime.now() - progress_data['start_time']).total_seconds()
                pages_per_sec = current / elapsed if elapsed > 0 else 0

                stats_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
                stats_table.add_column("Metric", style="cyan", width=25)
                stats_table.add_column("Value", style="green", width=15)

                stats_table.add_row("Total Pages", str(total_pages))
                stats_table.add_row("Current Page", f"{current}/{total}")
                stats_table.add_row("Progress", f"{(current / total * 100):.1f}%")
                stats_table.add_row("Processing Speed", f"{pages_per_sec:.2f} pages/sec")
                stats_table.add_row("Elapsed Time", f"{elapsed:.1f}s")
                stats_table.add_row("Figures Detected", str(progress_data['figures_found']))
                stats_table.add_row("References Found", str(progress_data['references_found']))
                stats_table.add_row("Texts Recognized", str(progress_data['texts_recognized']))

                layout["stats"].update(Panel(stats_table, title="Statistics", box=box.ROUNDED))

                # 로그 업데이트
                if len(log_messages) > 8:
                    log_messages.pop(0)
                log_panel = Panel(
                    "\n".join(log_messages[-8:]),
                    title="Recent Activity",
                    box=box.ROUNDED
                )
                layout["logs"].update(log_panel)

            # 페이지별 콜백
            def page_callback(page_result):
                if 'error' not in page_result:
                    progress_data['figures_found'] += len(page_result.get('figure_layouts', []))
                    progress_data['references_found'] += len(page_result.get('figure_references', []))
                    progress_data['texts_recognized'] += len(page_result.get('recognized_texts', []))

                    page_idx = page_result['page_index']
                    log_messages.append(
                        f"[green]✓[/green] Page {page_idx + 1}: "
                        f"{len(page_result.get('figure_layouts', []))} figures, "
                        f"{len(page_result.get('figure_references', []))} references"
                    )

            # 실제 분석 수행
            results = self.document_analyzer.analyze_pdf_with_callbacks(
                pdf_path,
                progress_callback=update_progress,
                page_callback=page_callback,
                frontend_format=frontend_format
            )

            # 완료
            progress.update(task, completed=total_pages)

        # 결과 저장
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"analysis_{Path(pdf_path).stem}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            console.print(f"\n[green]✓[/green] Results saved to: {output_path}")

        return results

    def display_results_summary(self, results: Dict[str, Any]):
        """결과 요약 표시"""
        console.print("\n[bold cyan]Analysis Summary[/bold cyan]")
        console.print("=" * 50)

        # Frontend format 체크
        if 'title' in results and 'chapters' in results and 'pages' in results:
            # Frontend format
            metadata = results.get('metadata', {})
            
            # 요약 테이블
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan", width=30)
            table.add_column("Value", style="green", width=20)

            table.add_row("Document Title", results.get('title', 'N/A'))
            table.add_row("Total Pages", str(metadata.get('total_pages', len(results.get('pages', [])))))
            table.add_row("Total Chapters", str(len(results.get('chapters', []))))
            
            # 레이아웃 통계 계산
            total_figures = 0
            total_texts = 0
            total_references = 0
            
            for page in results.get('pages', []):
                for layout in page.get('layouts', []):
                    if layout.get('type') == 'figure':
                        total_figures += 1
                    elif layout.get('type') == 'text':
                        total_texts += 1
                    elif layout.get('type') == 'figure_reference':
                        total_references += 1
            
            table.add_row("Total Figures", str(total_figures))
            table.add_row("Total Text Elements", str(total_texts))
            table.add_row("Total Figure References", str(total_references))
            table.add_row("Processing Time", f"{metadata.get('processing_time', 0):.2f}s")
        
        else:
            # Original format
            summary = results.get('summary', {})

            # 요약 테이블
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan", width=30)
            table.add_column("Value", style="green", width=20)

            table.add_row("Total Pages", str(results.get('total_pages', 0)))
            table.add_row("Total Layouts Detected", str(summary.get('total_layouts', 0)))
            table.add_row("Total Figures", str(summary.get('total_figures', 0)))
            table.add_row("Total Texts Recognized", str(summary.get('total_texts', 0)))
            table.add_row("Total Figure References", str(summary.get('total_figure_references', 0)))
            table.add_row("Error Pages", str(summary.get('error_pages', 0)))
            table.add_row("Success Rate", f"{summary.get('success_rate', 0):.1f}%")
            table.add_row("Processing Time", f"{results.get('processing_time', 0):.2f}s")

        console.print(table)

        # Figure 맵핑 결과
        if results.get('pages'):
            mapped_refs = 0
            unmapped_refs = 0

            if 'title' in results and 'chapters' in results:
                # Frontend format
                for page in results['pages']:
                    for layout in page.get('layouts', []):
                        if layout.get('type') == 'figure_reference':
                            if layout.get('referenced_figure_id'):
                                mapped_refs += 1
                            else:
                                unmapped_refs += 1
            else:
                # Original format
                for page in results['pages']:
                    if 'figure_references' in page:
                        for ref in page['figure_references']:
                            if ref.get('mapped_figure_id'):
                                mapped_refs += 1
                            else:
                                unmapped_refs += 1

            if mapped_refs > 0 or unmapped_refs > 0:
                console.print(f"\n[bold]Figure Mapping Results:[/bold]")
                console.print(f"  [green]✓[/green] Successfully mapped: {mapped_refs}")
                console.print(f"  [red]✗[/red] Failed to map: {unmapped_refs}")
                console.print(
                    f"  [blue]→[/blue] Mapping rate: {(mapped_refs / (mapped_refs + unmapped_refs) * 100):.1f}%")


@click.group()
def cli():
    """PDF Figure Reference Analyzer CLI"""
    pass


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory for results')
@click.option('--pages', '-p', type=str, help='Page range (e.g., "1-10" or "1,3,5")')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--frontend-format', '-f', is_flag=True, help='Output in frontend-friendly format')
def analyze(pdf_path: str, output: Optional[str], pages: Optional[str], verbose: bool, frontend_format: bool):
    """Analyze a PDF document for figure references"""

    # 파일 확인
    if not pdf_path.lower().endswith('.pdf'):
        console.print("[red]Error:[/red] Only PDF files are supported")
        sys.exit(1)

    # 파일 정보 표시
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
    console.print(f"\n[bold]File Information:[/bold]")
    console.print(f"  Path: {pdf_path}")
    console.print(f"  Size: {file_size:.2f} MB")

    # PDF 정보 확인
    analyzer = CLIAnalyzer()
    pdf_info = analyzer.pdf_processor.get_pdf_info(pdf_path)
    console.print(f"  Pages: {pdf_info['total_pages']}")

    if pdf_info.get('encrypted'):
        console.print("[red]Error:[/red] PDF is encrypted")
        sys.exit(1)

    # 페이지 범위 파싱
    if pages:
        # TODO: 페이지 범위 구현
        console.print(f"[yellow]Note:[/yellow] Page range not implemented yet")

    # 분석 시작
    console.print("\n[bold green]Starting analysis...[/bold green]\n")

    try:
        # 분석 수행
        results = analyzer.analyze_pdf_with_progress(pdf_path, output, frontend_format)

        # 결과 요약 표시
        analyzer.display_results_summary(results)

        # 상세 결과 표시 (verbose 모드)
        if verbose and results.get('pages'):
            console.print("\n[bold]Detailed Results:[/bold]")
            for page in results['pages'][:5]:  # 처음 5페이지만
                if page.get('figure_references'):
                    console.print(f"\nPage {page['page_index'] + 1}:")
                    for ref in page['figure_references']:
                        status = "[green]✓[/green]" if ref.get('mapped_figure_id') else "[red]✗[/red]"
                        console.print(f"  {status} {ref['text']} → {ref.get('mapped_figure_id', 'Not mapped')}")

        console.print("\n[bold green]Analysis completed successfully![/bold green]")

    except Exception as e:
        console.print(f"\n[red]Error during analysis:[/red] {str(e)}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument('result_file', type=click.Path(exists=True))
@click.option('--page', '-p', type=int, help='Show specific page results')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'summary']), default='summary')
def view(result_file: str, page: Optional[int], format: str):
    """View analysis results from a JSON file"""

    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        if format == 'summary':
            analyzer = CLIAnalyzer()
            analyzer.display_results_summary(results)

        elif format == 'table':
            # 특정 페이지 또는 전체 테이블 표시
            if page is not None:
                # 특정 페이지 결과
                page_data = None
                for p in results.get('pages', []):
                    if p.get('page_index') == page - 1:
                        page_data = p
                        break

                if page_data:
                    console.print(f"\n[bold]Page {page} Results:[/bold]")

                    # Figure References 테이블
                    if page_data.get('figure_references'):
                        ref_table = Table(title="Figure References", show_header=True)
                        ref_table.add_column("Text", style="cyan")
                        ref_table.add_column("Figure Number", style="yellow")
                        ref_table.add_column("Mapped To", style="green")
                        ref_table.add_column("Status", style="magenta")

                        for ref in page_data['figure_references']:
                            status = "✓ Mapped" if ref.get('mapped_figure_id') else "✗ Not Mapped"
                            ref_table.add_row(
                                ref['text'],
                                str(ref.get('figure_number', 'N/A')),
                                ref.get('mapped_figure_id', '-'),
                                status
                            )

                        console.print(ref_table)
                else:
                    console.print(f"[red]Page {page} not found[/red]")

        elif format == 'json':
            # JSON 형식으로 표시
            syntax = Syntax(
                json.dumps(results if not page else results.get('pages', [])[page - 1],
                           indent=2, ensure_ascii=False),
                "json",
                theme="monokai",
                line_numbers=True
            )
            console.print(syntax)

    except Exception as e:
        console.print(f"[red]Error reading results:[/red] {str(e)}")
        sys.exit(1)


@cli.command()
def info():
    """Show system information and configuration"""

    import torch
    import paddle

    console.print("[bold]System Information:[/bold]")

    # 시스템 정보 테이블
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=25)
    table.add_column("Status/Version", style="green", width=40)

    # PaddlePaddle
    table.add_row("PaddlePaddle", paddle.__version__)
    table.add_row("PaddlePaddle GPU", "✓ Available" if paddle.is_compiled_with_cuda() else "✗ Not Available")

    # PyTorch
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", "✓ Yes" if torch.cuda.is_available() else "✗ No")

    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda)
        table.add_row("GPU Device", torch.cuda.get_device_name(0))
        table.add_row("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # 시스템 정보
    import psutil
    table.add_row("CPU Cores", str(psutil.cpu_count()))
    table.add_row("RAM", f"{psutil.virtual_memory().total / 1024 ** 3:.1f} GB")
    table.add_row("Available RAM", f"{psutil.virtual_memory().available / 1024 ** 3:.1f} GB")

    console.print(table)

    # 모델 정보
    console.print("\n[bold]Configured Models:[/bold]")
    from config import config

    model_table = Table(show_header=True, header_style="bold magenta")
    model_table.add_column("Model Type", style="cyan", width=20)
    model_table.add_column("Model Name", style="green", width=30)

    model_table.add_row("Layout Detection", config.LAYOUT_MODEL)
    model_table.add_row("Text Detection", config.DET_MODEL)
    model_table.add_row("Text Recognition", config.REC_MODEL)
    model_table.add_row("BERT Model", config.BERT_MODEL)

    console.print(model_table)


if __name__ == '__main__':
    cli()