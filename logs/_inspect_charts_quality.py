import os
import sys
from pathlib import Path

sys.path.insert(0, r'c:\python\rag_textbook\src')
from settings import Settings
from pdf_parser import PdfParser

s = Settings()
os.environ['MINERU_MODEL_SOURCE'] = s.mineru_model_source
os.environ['MINERU_TOOLS_CONFIG_JSON'] = str(Path.home() / s.mineru_tools_config_json)
pdf = Path(s.pdf_dir) / 'KET-RAG.pdf'
parser = PdfParser(output_dir=s.markdown_dir, post_release_wait_seconds=0, post_release_poll_seconds=1, release_check_enabled=False)
blocks = parser.parse_doc([pdf])
charts = [b for b in blocks if str(b.get('type',''))=='chart']
with_caption = sum(1 for b in charts if (b.get('chart_caption') or b.get('chart_footnote')))
with_text = sum(1 for b in charts if str(b.get('text','')).strip())
print('charts_total', len(charts))
print('charts_with_caption_or_footnote', with_caption)
print('charts_with_text', with_text)
