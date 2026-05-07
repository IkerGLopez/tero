from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError as e:
    print('pypdf missing', e)
    raise SystemExit(1)

root = Path('c:/dev/tero')
pdfs = list(root.glob('src/backend/tests/assets/*.pdf'))
print('pdfs', [p.name for p in pdfs])
for p in pdfs:
    print('---', p.name)
    try:
        r = PdfReader(str(p))
        print('pages', len(r.pages))
        for i, page in enumerate(r.pages[:2]):
            text = page.extract_text() or ''
            print('PAGE', i, repr(text[:400]))
    except Exception as e:
        print('error', e)
