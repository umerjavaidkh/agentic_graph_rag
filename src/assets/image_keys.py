def _safe_document_id(document_id: str) -> str:
    return document_id.replace("/", "_")


def images_dir(document_id: str) -> str:
    return f"{_safe_document_id(document_id)}/images"


def page_full_image_key(document_id: str, pdf_page: int) -> str:
    return f"{images_dir(document_id)}/page_{pdf_page:04d}_full.jpg"


def region_image_key(document_id: str, pdf_page: int, kind: str, index: int) -> str:
    return f"{images_dir(document_id)}/page_{pdf_page:04d}_{kind}_{index:02d}.jpg"


def page_image_key(document_id: str, pdf_page: int) -> str:
    return page_full_image_key(document_id, pdf_page)


# Deprecated aliases (book_*)
_safe_book_id = _safe_document_id
