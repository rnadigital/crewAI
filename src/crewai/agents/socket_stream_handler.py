import uuid
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Callable
from uuid import UUID

from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult


class SocketStreamHandler(BaseCallbackHandler):
    def __init__(self, socket_write_fn: Callable, agent_name: str, task_name: str, tools_names: str):
        self.send_to_socket = socket_write_fn
        self.agent_name = agent_name
        self.chunkId = str(uuid.uuid4())
        self.task_chunkId = str(uuid.uuid4())
        self.first = True
        self.send_to_socket(
            text=f"""**Running task**: {task_name} **Available tools**: {tools_names}""",
            event="message",
            first=self.first,
            chunk_id=self.task_chunkId,
            timestamp=datetime.now().timestamp() * 1000,
            display_type="inline"
        )

    def on_chain_end(
            self,
            outputs: Dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:
        self.chunkId = str(uuid.uuid4())
        self.first = True

    def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:
        self.send_to_socket(
            text="",
            event="terminate"
        )

    def on_llm_new_token(
            self,
            token: str,
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:
        if token:
            self.send_to_socket(
                text=token,
                event="message",
                first=True,
                chunk_id=self.chunkId,
                timestamp=datetime.now().timestamp() * 1000,
                display_type="bubble",
                author_name=self.agent_name
            )
