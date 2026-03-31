# Copyright (c) Opendatalab. All rights reserved.

import json
import os
import time
from pathlib import Path
from typing import List, Optional

from loguru import logger

from mineru.cli.common import convert_pdf_bytes_to_bytes, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming as pipeline_doc_analyze_streaming
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make


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
        release_models_after_parse: bool = True,
        post_release_wait_seconds: int = 120,
        post_release_poll_seconds: int = 5,
        release_check_enabled: bool = True,
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
        self.release_models_after_parse = release_models_after_parse
        self.post_release_wait_seconds = max(0, int(post_release_wait_seconds))
        self.post_release_poll_seconds = max(1, int(post_release_poll_seconds))
        self.release_check_enabled = bool(release_check_enabled)

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

        try:
            return self._do_parse(
                output_dir=self.output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                parse_method=self.method,
                start_page_id=self.start_page_id,
                end_page_id=self.end_page_id,
            )
        finally:
            if self.release_models_after_parse:
                self._release_pipeline_models()
                self._wait_for_release_barrier()

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
        """Uses MinerU pipeline backend with streaming callback API."""
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes(pdf_bytes, start_page_id, end_page_id)
            pdf_bytes_list[idx] = new_pdf_bytes
        image_writer_list = []
        md_writer_list = []
        local_output_info = []
        for idx, _ in enumerate(pdf_bytes_list):
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer = FileBasedDataWriter(local_image_dir)
            md_writer = FileBasedDataWriter(local_md_dir)
            image_writer_list.append(image_writer)
            md_writer_list.append(md_writer)
            local_output_info.append((pdf_file_name, local_image_dir, local_md_dir))

        per_doc_content: list = [None] * len(pdf_bytes_list)

        def on_doc_ready(doc_index, model_list, middle_json, _ocr_enable):
            pdf_file_name, local_image_dir, local_md_dir = local_output_info[doc_index]
            md_writer = md_writer_list[doc_index]
            pdf_bytes = pdf_bytes_list[doc_index]
            content_list = self._process_output(
                middle_json["pdf_info"],
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
                model_list,
            )
            per_doc_content[doc_index] = content_list

        pipeline_doc_analyze_streaming(
            pdf_bytes_list,
            image_writer_list,
            p_lang_list,
            on_doc_ready,
            parse_method=parse_method,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
        )

        merged: List[dict] = []
        for item in per_doc_content:
            if isinstance(item, list):
                merged.extend(item)
        return merged

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

    @staticmethod
    def _release_pipeline_models() -> None:
        """
        Best-effort cleanup for MinerU singleton caches to free GPU memory
        before the embedding phase starts.
        """
        released = False
        try:
            from mineru.backend.pipeline.pipeline_analyze import ModelSingleton as PipelineModelSingleton

            PipelineModelSingleton._models.clear()
            released = True
        except Exception:
            pass

        try:
            from mineru.backend.pipeline.model_init import AtomModelSingleton

            AtomModelSingleton._models.clear()
            released = True
        except Exception:
            pass

        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

        if released:
            logger.info("Released MinerU pipeline model cache after parse.")

    @staticmethod
    def _mineru_loaded_model_count() -> int:
        """
        Returns number of models currently visible in MinerU singleton caches.
        """
        total = 0
        try:
            from mineru.backend.pipeline.pipeline_analyze import ModelSingleton as PipelineModelSingleton

            total += len(getattr(PipelineModelSingleton, "_models", {}) or {})
        except Exception:
            pass
        try:
            from mineru.backend.pipeline.model_init import AtomModelSingleton

            total += len(getattr(AtomModelSingleton, "_models", {}) or {})
        except Exception:
            pass
        return int(total)

    @staticmethod
    def _cuda_mem_info_mb() -> tuple[int, int] | None:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return int(free_bytes // (1024 * 1024)), int(total_bytes // (1024 * 1024))
        except Exception:
            return None

    def _wait_for_release_barrier(self) -> None:
        """
        Barrier between MinerU parse and next stage:
        wait and verify MinerU singleton caches are empty.
        """
        if not self.release_check_enabled:
            return

        max_wait = int(self.post_release_wait_seconds)
        poll = int(self.post_release_poll_seconds)
        logger.info(
            f"MinerU release barrier: waiting up to {max_wait}s (poll={poll}s) before next stage."
        )

        waited = 0
        while True:
            loaded = self._mineru_loaded_model_count()
            vram_info = self._cuda_mem_info_mb()
            if vram_info is None:
                logger.info(
                    f"MinerU unload check: loaded_models={loaded}, CUDA unavailable."
                )
            else:
                free_mb, total_mb = vram_info
                logger.info(
                    f"MinerU unload check: loaded_models={loaded}, free_vram={free_mb}/{total_mb} MB."
                )

            if loaded == 0 and waited >= max_wait:
                logger.info(
                    f"MinerU release barrier passed after {waited}s (models unloaded)."
                )
                return
            if waited >= max_wait:
                if loaded == 0:
                    logger.info(
                        f"MinerU release barrier passed after {waited}s (models unloaded)."
                    )
                else:
                    logger.warning(
                        f"MinerU release barrier timeout ({max_wait}s): loaded_models={loaded}. Continuing to next stage."
                    )
                return

            sleep_s = min(poll, max_wait - waited)
            time.sleep(sleep_s)
            waited += sleep_s
