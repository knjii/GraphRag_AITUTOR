import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from langchain_core.documents import Document

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json


DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))


def _join_text(parts: Iterable[str]) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return "\n".join(cleaned)


def _normalize_image_path(img_path: str, base_dir: str) -> Optional[Dict[str, str]]:
    if not img_path:
        return None
    if os.path.isabs(img_path):
        abs_path = os.path.normpath(img_path)
        rel_path = os.path.relpath(abs_path, base_dir) if base_dir else abs_path
    else:
        rel_path = img_path
        abs_path = os.path.normpath(os.path.join(base_dir, img_path))
    return {"path": abs_path, "path_rel": rel_path}


def _content_item_to_text(item: Dict[str, Any]) -> str:
    content_type = item.get("type")
    if content_type == "text":
        text = item.get("text", "")
        level = item.get("text_level")
        if level:
            text = f"{'#' * int(level)} {text}"
        return text.strip()
    if content_type == "equation":
        return (item.get("text") or "").strip()
    if content_type == "image":
        return _join_text((item.get("image_caption") or []) + (item.get("image_footnote") or []))
    if content_type == "table":
        parts: List[str] = []
        parts.extend(item.get("table_caption") or [])
        table_body = item.get("table_body") or ""
        if table_body:
            parts.append(table_body)
        parts.extend(item.get("table_footnote") or [])
        return _join_text(parts)
    return (item.get("text") or "").strip()


def _extract_images(item: Dict[str, Any], base_dir: str) -> List[Dict[str, Any]]:
    img_path = item.get("img_path")
    normalized = _normalize_image_path(img_path, base_dir) if img_path else None
    if not normalized:
        return []
    image_entry: Dict[str, Any] = {
        "path": normalized["path"],
        "path_rel": normalized["path_rel"],
        "type": item.get("type"),
    }
    if item.get("page_idx") is not None:
        image_entry["page_idx"] = item["page_idx"]
    if item.get("bbox"):
        image_entry["bbox"] = item["bbox"]

    if item.get("type") == "image":
        caption = _join_text(item.get("image_caption") or [])
        footnote = _join_text(item.get("image_footnote") or [])
        if caption:
            image_entry["caption"] = caption
        if footnote:
            image_entry["footnote"] = footnote
    elif item.get("type") == "table":
        caption = _join_text(item.get("table_caption") or [])
        footnote = _join_text(item.get("table_footnote") or [])
        if caption:
            image_entry["caption"] = caption
        if footnote:
            image_entry["footnote"] = footnote
    elif item.get("type") == "equation":
        latex = item.get("text")
        if latex:
            image_entry["latex"] = latex

    return [image_entry]


def _build_segments(content_list: List[Dict[str, Any]], base_dir: str) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for item in content_list:
        if item.get("type") == "discarded":
            continue
        text = _content_item_to_text(item)
        images = _extract_images(item, base_dir)
        if not text and not images:
            continue
        segments.append(
            {
                "text": text,
                "page_idx": item.get("page_idx"),
                "bbox": item.get("bbox"),
                "images": images,
                "content_type": item.get("type"),
            }
        )
    return segments


def _split_text(text: str, max_len: int) -> List[str]:
    if max_len <= 0 or len(text) <= max_len:
        return [text]
    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_len, text_len)
        if end < text_len:
            split_at = text.rfind(" ", start, end)
            if split_at <= start:
                split_at = end
        else:
            split_at = end
        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)
        start = split_at
    return chunks


def _expand_segments_for_chunking(segments: List[Dict[str, Any]], chunk_size: int) -> List[Dict[str, Any]]:
    if chunk_size <= 0:
        return segments
    expanded: List[Dict[str, Any]] = []
    for segment in segments:
        text = segment.get("text", "").strip()
        if text and len(text) > chunk_size:
            for part in _split_text(text, chunk_size):
                new_segment = segment.copy()
                new_segment["text"] = part
                expanded.append(new_segment)
        else:
            expanded.append(segment)
    return expanded


def _iter_chunk_segments(
    segments: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int
) -> Iterable[List[Dict[str, Any]]]:
    if chunk_size <= 0:
        if segments:
            yield segments
        return

    current: List[Dict[str, Any]] = []
    current_len = 0
    has_text = False

    for segment in segments:
        seg_text = segment.get("text", "").strip()
        add_len = 0
        if seg_text:
            add_len = len(seg_text) + (2 if has_text else 0)

        if current and add_len and current_len + add_len > chunk_size:
            yield current

            if chunk_overlap > 0:
                overlap_segments: List[Dict[str, Any]] = []
                overlap_len = 0
                overlap_has_text = False
                for prev in reversed(current):
                    prev_text = prev.get("text", "").strip()
                    if not prev_text:
                        continue
                    prev_len = len(prev_text) + (2 if overlap_has_text else 0)
                    if overlap_len + prev_len > chunk_overlap:
                        break
                    overlap_segments.append(prev)
                    overlap_len += prev_len
                    overlap_has_text = True
                current = list(reversed(overlap_segments))
                current_len = overlap_len
                has_text = overlap_has_text
            else:
                current = []
                current_len = 0
                has_text = False

        current.append(segment)
        if seg_text:
            current_len += add_len
            has_text = True

    if current:
        yield current


def _merge_bbox(existing: List[int], new_bbox: List[int]) -> List[int]:
    return [
        min(existing[0], new_bbox[0]),
        min(existing[1], new_bbox[1]),
        max(existing[2], new_bbox[2]),
        max(existing[3], new_bbox[3]),
    ]


def _segments_to_document(
    segments: List[Dict[str, Any]],
    source_path: str,
    md_path: str,
    image_dir: str,
    chunk_index: int,
) -> Optional[Document]:
    text_parts = [seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip()]
    page_content = "\n\n".join(text_parts).strip()

    page_idxs = sorted({seg.get("page_idx") for seg in segments if seg.get("page_idx") is not None})
    page_spans: Dict[int, List[int]] = {}
    images: List[Dict[str, Any]] = []
    for seg in segments:
        page_idx = seg.get("page_idx")
        bbox = seg.get("bbox")
        if page_idx is not None and bbox:
            if page_idx not in page_spans:
                page_spans[page_idx] = list(bbox)
            else:
                page_spans[page_idx] = _merge_bbox(page_spans[page_idx], list(bbox))
        images.extend(seg.get("images", []))

    if not page_content:
        if images and page_idxs:
            page_content = f"Image from {Path(source_path).name} page {page_idxs[0]}"
        elif images:
            page_content = f"Image from {Path(source_path).name}"
        else:
            return None

    metadata: Dict[str, Any] = {
        "source": str(source_path),
        "source_file": Path(source_path).name,
        "source_md": str(md_path),
        "source_images_dir": str(image_dir),
        "chunk_index": chunk_index,
        "page_idxs": page_idxs,
    }
    if page_idxs:
        metadata["page_start"] = page_idxs[0]
        metadata["page_end"] = page_idxs[-1]
    if page_spans:
        metadata["page_spans"] = [
            {"page_idx": page_idx, "bbox": bbox} for page_idx, bbox in sorted(page_spans.items())
        ]
    if images:
        metadata["images"] = images
        metadata["image_paths"] = [img["path"] for img in images if img.get("path")]
        metadata["image_paths_rel"] = [img["path_rel"] for img in images if img.get("path_rel")]

    return Document(page_content=page_content, metadata=metadata)


def build_documents_from_content_list(
    content_list: List[Dict[str, Any]],
    source_path: str,
    md_path: str,
    image_dir: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    segments = _build_segments(content_list, os.path.dirname(md_path))
    if not segments:
        return []
    segments = _expand_segments_for_chunking(segments, chunk_size)

    documents: List[Document] = []
    for chunk_index, chunk_segments in enumerate(_iter_chunk_segments(segments, chunk_size, chunk_overlap)):
        doc = _segments_to_document(chunk_segments, source_path, md_path, image_dir, chunk_index)
        if doc is not None:
            documents.append(doc)
    return documents


def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    pdf_paths: Optional[List[Path]] = None,  # Optional list of original PDF paths
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    backend="pipeline",  # Only 'pipeline' is supported in this file
    formula_enable=True,  # Enable formula parsing
    table_enable=True,  # Enable table parsing
    server_url=None,  # Unused in pipeline mode
    f_draw_layout_bbox=False,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=False,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=False,  # Whether to dump middle JSON files
    f_dump_model_output=False,  # Whether to dump model output files
    f_dump_orig_pdf=False,  # Whether to dump original PDF files
    f_dump_content_list=False,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
):
    """
    Uses 'pipeline' as backend by default, because OS windows doesn't support vllm
    """
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = DEFAULT_CHUNK_OVERLAP
    documents: List[Document] = []

    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        pdf_bytes_list[idx] = new_pdf_bytes

    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list, p_lang_list, parse_method=parse_method, formula_enable=formula_enable,table_enable=table_enable)

    for idx, model_list in enumerate(infer_results):
        model_json = copy.deepcopy(model_list)
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        images_list = all_image_lists[idx]
        pdf_doc = all_pdf_docs[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]
        middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, formula_enable)

        pdf_info = middle_json["pdf_info"]

        pdf_bytes = pdf_bytes_list[idx]
        content_list = _process_output(
            pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
            f_make_md_mode, middle_json, model_json, return_content_list=True
        )
        if content_list:
            source_path = (
                str(pdf_paths[idx]) if pdf_paths and idx < len(pdf_paths) else pdf_file_name
            )
            md_path = os.path.join(local_md_dir, f"{pdf_file_name}.md")
            documents.extend(
                build_documents_from_content_list(
                    content_list,
                    source_path=source_path,
                    md_path=md_path,
                    image_dir=local_image_dir,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )

    return documents


def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
        return_content_list=False
):
    """output processing"""

    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.pdf",
            pdf_bytes,
        )

    image_dir = str(os.path.basename(local_image_dir))
    make_func = pipeline_union_make

    if f_dump_md:
        md_content_str = make_func(pdf_info, f_make_md_mode, image_dir)
        if not isinstance(md_content_str, str):
            md_content_str = json.dumps(md_content_str, ensure_ascii=False, indent=2)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

    content_list = None
    if f_dump_content_list or return_content_list:
        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir)
    if f_dump_content_list and content_list is not None:
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        md_writer.write_string(
            f"{pdf_file_name}_model.json",
            json.dumps(model_output, ensure_ascii=False, indent=4),
        )

    logger.info(f"local output dir is {local_md_dir}")
    return content_list


def parse_doc(
        path_list: list[Path],
        output_dir,
        lang="ch",
        method="auto",
        start_page_id=0,
        end_page_id=None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to "pipeline".
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
        chunk_size: Target chunk size for Document splitting (defaults to env CHUNK_SIZE or 200)
        chunk_overlap: Overlap between chunks (defaults to env CHUNK_OVERLAP or 40)
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        return do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            parse_method=method,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            pdf_paths=[Path(path) for path in path_list],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as e:
        logger.exception(e)
        return []
