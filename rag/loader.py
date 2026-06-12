from pypdf import PdfReader

def load_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)

    texts = [
        page.extract_text().strip()
        for page in reader.pages
    ]

    return [text for text in texts if text]

    '''
    result = []

    for text in texts:

        if text:
            result.append(text)

    return result
    '''