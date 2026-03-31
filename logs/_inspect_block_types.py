import os
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, r'c:\python\rag_textbook\src')

from settings import Settings
from pdf_parser import PdfParser

s = Settings()
os.environ['MINERU_MODEL_SOURCE'] = s.mineru_model_source
os.environ['MINERU_TOOLS_CONFIG_JSON'] = str(Path.home() / s.mineru_tools_config_json)

pdf = Path(s.pdf_dir) / 'KET-RAG.pdf'
parser = PdfParser(
    output_dir=s.markdown_dir,
    post_release_wait_seconds=0,
    post_release_poll_seconds=1,
    release_check_enabled=False,
)
blocks = parser.parse_doc([pdf])
ctr = Counter(str(b.get('type','')) for b in blocks)
print('TOTAL_BLOCKS', len(blocks))
print('TOP_TYPES')
for k,v in ctr.most_common(25):
    print(f'{k}\t{v}')

strict = {'image','table','equation'}
extended = {'image','image_body','table','table_body','equation','interline_equation'}
print('STRICT_SPECIAL_COUNT', sum(1 for b in blocks if str(b.get('type','')) in strict))
print('EXTENDED_SPECIAL_COUNT', sum(1 for b in blocks if str(b.get('type','')) in extended))
print('STRICT_TYPES_PRESENT', sorted({str(b.get('type','')) for b in blocks if str(b.get('type','')) in strict}))
print('EXTENDED_TYPES_PRESENT', sorted({str(b.get('type','')) for b in blocks if str(b.get('type','')) in extended}))
