from datetime import datetime
from uuid import uuid4

from socketio import SimpleClient
from typing import Any, Optional
from pydantic import BaseModel, constr, field_validator

from crewai.agentcloud.types.socket import SocketMessage, SocketEvents, Message
from .utils import check_instance_of_class
from ..utilities import Logger


class AgentCloudSocketIO:
    """
    Wrapper containing methods for sending and receiving messages over sockets to AgentCloud
    """
    logger = Logger(verbose_level=2)

    def __init__(self, socket: SimpleClient, session_id: str) -> None:
        self.socket = socket
        self.session_id = session_id

    @classmethod
    def send(cls, client: Optional[SimpleClient], event: Optional[SocketEvents],
             message: Optional[SocketMessage], socket_logging: str = "socket"):
        # Check inputs
        class Params(BaseModel):
            client: Any
            event: Any
            message: Any
            socket_logging: constr(min_length=1)

            @field_validator("socket_logging")
            def check_socket_or_logging(cls, v):
                socket_log = v.lower()
                assert socket_logging in [
                    "socket",
                    "logging",
                    "both",
                ], f"Invalid socket_logging value: {v}"
                return socket_log

        params = Params(
            client=client, event=event, message=message, socket_logging=socket_logging
        )

        # Advanced checks
        if params.socket_logging in ["socket", "both"]:
            assert params.client is not None, "client cannot be None"
            assert params.event is not None, "event cannot be None"
            assert params.message is not None, "message cannot be None"
            # Assert types
            params.client = check_instance_of_class(params.client, SimpleClient)
            params.event = check_instance_of_class(params.event, SocketEvents)
            params.message = check_instance_of_class(params.message, SocketMessage)

        # If logging, print message
        if params.socket_logging in ["logging", "both"]:
            cls.logger.log(level="debug", message=f"Sending message to socket: {params.message.model_dump()}")

        # If socket or both, send message to socket
        if params.socket_logging in ["socket", "both"]:
            # If client is not connected, raise an error
            assert params.client.connected, "Socket client is not connected"
            client.emit(event=params.event.value, data=params.message.model_dump())

    def send_to_socket(self, text='', event=SocketEvents.MESSAGE, first=True, chunk_id=None,
                       timestamp=None, display_type='bubble', author_name='System', overwrite=False):

        if type(text) is not str:
            text = "NON STRING MESSAGE"

        # Set default timestamp if not provided
        if timestamp is None:
            timestamp = int(datetime.now().timestamp() * 1000)

        # send the message
        self.send(
            self.socket,
            SocketEvents(event),
            SocketMessage(
                room=self.session_id,
                authorName=author_name,
                message=Message(
                    chunkId=chunk_id,
                    text=text,
                    first=first,
                    tokens=1,
                    timestamp=timestamp,
                    displayType=display_type,
                    overwrite=overwrite,
                )
            ),
            "socket"
        )

    def get_human_input(self, input_prompt: str, author_name='System'):
        try:
            self.send(
                self.socket,
                SocketEvents.MESSAGE,
                SocketMessage(
                    room=self.session_id,
                    authorName=author_name,
                    message=Message(
                        chunkId=str(uuid4()),
                        text="",
                        first=True,
                        tokens=1,  # Assumes 1 token is a constant value for message segmentation.
                        timestamp=datetime.now().timestamp() * 1000,
                        single=True,
                    ),
                    isFeedback=True,
                ),
                "socket"
            )
            feedback = self.socket.receive()
            return feedback[1]
        except TimeoutError:
            self.socket.emit(
                "message",
                {
                    "room": self.session_id,
                    "type": "error",
                    "message": "TimeOutError"
                },
            )
            return "exit"
