# Copyright (c) Opendatalab. All rights reserved.

import copy
import json
import os
from pathlib import Path
from typing import List, Optional

from loguru import logger

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json


class PdfParser:
    def __init__(
        self,
        output_dir: str | Path,
        lang: str = "ru",
        method: str = "auto",
        start_page_id: int = 0,
        end_page_id: Optional[int] = None,
        formula_enable: bool = True,
        table_enable: bool = True,
        draw_layout_bbox: bool = False,
        draw_span_bbox: bool = False,
        dump_md: bool = True,
        dump_middle_json: bool = False,
        dump_model_output: bool = False,
        dump_orig_pdf: bool = False,
        dump_content_list: bool = False,
        make_md_mode=MakeMode.CONTENT_LIST,
    ) -> None:
        self.output_dir = output_dir
        self.lang = lang
        self.method = method
        self.start_page_id = start_page_id
        self.end_page_id = end_page_id
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.draw_layout_bbox = draw_layout_bbox
        self.draw_span_bbox = draw_span_bbox
        self.dump_md = dump_md
        self.dump_middle_json = dump_middle_json
        self.dump_model_output = dump_model_output
        self.dump_orig_pdf = dump_orig_pdf
        self.dump_content_list = dump_content_list
        self.make_md_mode = make_md_mode

    def parse_doc(self, path_list: List[Path]) -> List[dict]:
        """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm: supported in demo, but removed from this file
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to "pipeline".
        server_url: When the backend is `http-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
         """
        
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(self.lang)

        return self._do_parse(
            output_dir=self.output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            parse_method=self.method,
            start_page_id=self.start_page_id,
            end_page_id=self.end_page_id,
        )

    def _do_parse(
        self,
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        parse_method="auto",
        start_page_id=0,
        end_page_id=None,
    ) -> List[dict]:
        """Uses 'pipeline' as backend by default, because OS windows doesn't support vllm"""
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            pdf_bytes_list[idx] = new_pdf_bytes

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
            pdf_bytes_list,
            p_lang_list,
            parse_method=parse_method,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
        )

        content_list_out: List[dict] = []

        for idx, model_list in enumerate(infer_results):
            model_json = copy.deepcopy(model_list)
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, self.formula_enable
            )

            pdf_info = middle_json["pdf_info"]
            pdf_bytes = pdf_bytes_list[idx]

            content_list = self._process_output(
                pdf_info,
                pdf_bytes,
                pdf_file_name,
                local_md_dir,
                local_image_dir,
                md_writer,
                self.draw_layout_bbox,
                self.draw_span_bbox,
                self.dump_orig_pdf,
                self.dump_md,
                self.dump_content_list,
                self.dump_middle_json,
                self.dump_model_output,
                self.make_md_mode,
                middle_json,
                model_json,
            )
            content_list_out = content_list

        return content_list_out

    def _process_output(
        self,
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
    ) -> List[dict]:
        
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

        content_list = []
        if f_dump_md:
            make_func = pipeline_union_make
            content_list = make_func(pdf_info, f_make_md_mode, image_dir)
            md_writer.write_string(
                f"{pdf_file_name}.md",
                content_list,
            )

        if f_dump_content_list:
            make_func = pipeline_union_make
            content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir)
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
