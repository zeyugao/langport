import argparse
import asyncio
import dataclasses
import logging
import json
import os
import time
from typing import Iterable, List, Optional, Union
import threading
import uuid
import traceback

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests
from tenacity import retry, stop_after_attempt
from langport.core.cluster_worker import ClusterWorker
from langport.model.executor.base import BaseModelExecutor
from langport.model.executor.generation import GenerationExecutor

from langport.protocol.worker_protocol import (
    BaseWorkerResult,
    GenerationTask,
    GenerationWorkerResult,
    UsageInfo,
)

import torch

from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
    TopKLogitsWarper,
)
from langport.constants import (
    GENERATION_INFERENCE_INTERVAL,
    WORKER_API_TIMEOUT,
    WORKER_HEART_BEAT_INTERVAL,
    ErrorCode,
)
from langport.utils import server_error_msg, pretty_print_semaphore


class GenerationModelWorker(ClusterWorker):
    def __init__(
        self,
        node_addr: str,
        node_id: str,
        init_neighborhoods_addr: List[str],
        executor: GenerationExecutor,
        limit_model_concurrency: int,
        max_batch: int,
        stream_interval: int,
        logger
    ):
        super(GenerationModelWorker, self).__init__(
            node_addr=node_addr,
            node_id=node_id,
            init_neighborhoods_addr=init_neighborhoods_addr,
            limit_model_concurrency=limit_model_concurrency,
            max_batch=max_batch,
            stream_interval=stream_interval,
            logger=logger,
        )
        self.executor = executor
        workers = max(1, 2 * self.limit_model_concurrency // self.max_batch)
        self.add_timer(
            "generation_inference",
            GENERATION_INFERENCE_INTERVAL,
            self.executor.inference,
            args=[self,],
            kwargs=None,
            workers=workers,
        )
    
        self.on_start("set_features", self.set_features)
        self.on_start("set_model_name", self.set_model_name)

    async def set_features(self):
        await self.set_local_state("features", ["generation"], ttl=360)
    
    async def set_model_name(self):
        await self.set_local_state("model_name", self.executor.model_name, ttl=360)


    async def generation_stream(self, task: GenerationTask):
        await self.add_task(task)
        async for chunk in self.fetch_task_result(task.task_id):
            yield chunk

    async def generation_bytes_stream(self, task: GenerationTask):
        async for chunk in self.generation_stream(task):
            yield json.dumps(chunk.dict()).encode() + b"\0"
