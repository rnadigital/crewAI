import os
import uuid
import json
import asyncio
import threading
from typing import Any, Dict, List, Optional, Tuple, Type
from datetime import datetime

from langchain.agents.agent import RunnableAgent
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.tools import tool as LangChainTool
from langchain.tools.render import render_text_description
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from queue import Queue
from crewai.agents import CacheHandler, CrewAgentExecutor, CrewAgentParser, ToolsHandler
from crewai.agents.custom_parsers import GeminiAgentParser
from crewai.memory.contextual.contextual_memory import ContextualMemory
from crewai.utilities import I18N, Logger, Prompts, RPMController
from crewai.utilities.token_counter_callback import TokenCalcHandler, TokenProcess

import logging
logging.basicConfig(level=(os.getenv("LOGGING_LEVEL", "debug").lower() or logging.DEBUG))

class Agent(BaseModel):
    """Represents an agent in a system.

    Each agent has a role, a goal, a backstory, and an optional language model (llm).
    The agent can also have memory, can operate in verbose mode, and can delegate tasks to other agents.

    Attributes:
            agent_executor: An instance of the CrewAgentExecutor class.
            role: The role of the agent.
            goal: The objective of the agent.
            backstory: The backstory of the agent.
            config: Dict representation of agent configuration.
            llm: The language model that will run the agent.
            function_calling_llm: The language model that will the tool calling for this agent, it overrides the crew function_calling_llm.
            max_iter: Maximum number of iterations for an agent to execute a task.
            memory: Whether the agent should have memory or not.
            max_rpm: Maximum number of requests per minute for the agent execution to be respected.
            verbose: Whether the agent execution should be in verbose mode.
            allow_delegation: Whether the agent is allowed to delegate tasks to other agents.
            tools: Tools at agents disposal
            step_callback: Callback to be executed after each step of the agent execution.
            stop_generating_check: Callback to be executed every nth chunk to check if generation should be stopped
            callbacks: A list of callback functions from the langchain library that are triggered during the agent's execution process
    """

    __hash__ = object.__hash__  # type: ignore
    _logger: Logger = PrivateAttr()
    _rpm_controller: RPMController = PrivateAttr(default=None)
    _request_within_rpm_limit: Any = PrivateAttr(default=None)
    _token_process: TokenProcess = TokenProcess()

    formatting_errors: int = 0
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )
    name: str = Field(description="Name of the agent")
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Objective of the agent")
    backstory: str = Field(description="Backstory of the agent")
    cache: bool = Field(
        default=True,
        description="Whether the agent should use a cache for tool usage.",
    )
    config: Optional[Dict[str, Any]] = Field(
        description="Configuration for the agent",
        default=None,
    )
    max_rpm: Optional[int] = Field(
        default=None,
        description="Maximum number of requests per minute for the agent execution to be respected.",
    )
    verbose: bool = Field(
        default=False, description="Verbose mode for the Agent Execution"
    )
    allow_delegation: bool = Field(
        default=True, description="Allow delegation of tasks to agents"
    )
    tools: Optional[List[Any]] = Field(
        default_factory=list, description="Tools at agents disposal"
    )
    max_iter: Optional[int] = Field(
        default=25, description="Maximum iterations for an agent to execute a task"
    )
    max_execution_time: Optional[int] = Field(
        default=None,
        description="Maximum execution time for an agent to execute a task",
    )
    agent_executor: InstanceOf[CrewAgentExecutor] = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    crew: Any = Field(default=None, description="Crew to which the agent belongs.")
    tools_handler: InstanceOf[ToolsHandler] = Field(
        default=None, description="An instance of the ToolsHandler class."
    )
    cache_handler: InstanceOf[CacheHandler] = Field(
        default=None, description="An instance of the CacheHandler class."
    )
    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each step of the agent execution.",
    )
    stop_generating_check: Optional[Any] = Field(
        default=None,
        description="Callback to be executed every nth chunk to check if generation should be stopped",
    )
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")
    llm: Any = Field(
        default_factory=lambda: ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4")
        ),
        description="Language model that will run the agent.",
    )
    function_calling_llm: Optional[Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    callbacks: Optional[List[InstanceOf[BaseCallbackHandler]]] = Field(
        default=None, description="Callback to be executed"
    )
    system_template: Optional[str] = Field(
        default=None, description="System format for the agent."
    )
    prompt_template: Optional[str] = Field(
        default=None, description="Prompt format for the agent."
    )
    response_template: Optional[str] = Field(
        default=None, description="Response format for the agent."
    )

    _original_role: str | None = None
    _original_goal: str | None = None
    _original_backstory: str | None = None

    def __init__(__pydantic_self__, **data):
        config = data.pop("config", {})
        super().__init__(**config, **data)

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def set_attributes_based_on_config(self) -> "Agent":
        """Set attributes based on the agent configuration."""
        if self.config:
            for key, value in self.config.items():
                setattr(self, key, value)
        return self

    @model_validator(mode="after")
    def set_private_attrs(self):
        """Set private attributes."""
        self._logger = Logger(self.verbose)
        if self.max_rpm and not self._rpm_controller:
            self._rpm_controller = RPMController(
                max_rpm=self.max_rpm, logger=self._logger
            )
        return self

    @model_validator(mode="after")
    def set_agent_executor(self) -> "Agent":
        """set agent executor is set."""
        if hasattr(self.llm, "model_name"):
            token_handler = TokenCalcHandler(self.llm.model_name, self._token_process)

            # Ensure self.llm.callbacks is a list
            if not isinstance(self.llm.callbacks, list):
                self.llm.callbacks = []

            # Check if an instance of TokenCalcHandler already exists in the list
            if not any(
                isinstance(handler, TokenCalcHandler) for handler in self.llm.callbacks
            ):
                self.llm.callbacks.append(token_handler)

        if not self.agent_executor:
            if not self.cache_handler:
                self.cache_handler = CacheHandler()
            self.set_cache_handler(self.cache_handler)
        return self

    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        """Execute a task with the agent.

        Args:
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.

        Returns:
            Output of the agent
        """
        if self.tools_handler:
            self.tools_handler.last_used_tool = {}

        task_prompt = task.prompt()

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        if self.crew and self.crew.memory:
            contextual_memory = ContextualMemory(
                self.crew._short_term_memory,
                self.crew._long_term_memory,
                self.crew._entity_memory,
            )
            memory = contextual_memory.build_context_for_task(task, context)
            if memory.strip() != "":
                task_prompt += self.i18n.slice("memory").format(memory=memory)

        tools = tools or self.tools
        parsed_tools = self._parse_tools(tools)

        self.create_agent_executor(tools=tools)
        self.agent_executor.tools = parsed_tools
        self.agent_executor.task = task

        self.agent_executor.tools_description = render_text_description(parsed_tools)
        self.agent_executor.tools_names = self.__tools_names(parsed_tools)

        if self.step_callback:
            result_queue = Queue()
            _thread = threading.Thread(target=self.wrap_async_func, args=(task_prompt, task.name, result_queue))
            _thread.start()
            _thread.join()
            result = result_queue.get()
        else:
            result = self.agent_executor.invoke(
                {
                    "input": task_prompt,
                    "tool_names": self.agent_executor.tools_names,
                    "tools": self.agent_executor.tools_description,
                }
            )["output"]

        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

        return result

    def wrap_async_func(self, task_prompt, task_name, queue):
        asyncio.run(self.stream_execute(task_prompt, task_name, queue))

    async def stream_execute(self, task_prompt, task_name, result_queue):
        result = ""
        acc = ""
        chunkId = str(uuid.uuid4())
        task_chunkId = str(uuid.uuid4())
        tool_chunkId = str(uuid.uuid4())
        first = True
        agent_name = ""
        step = 1
        try:
            self.step_callback(f"""**Running task**: {task_name} **Available tools**: {self.agent_executor.tools_names}""", "message", True, task_chunkId, datetime.now().timestamp() * 1000, "inline")
            async for event in self.agent_executor.astream_events(
                {
                    "input": task_prompt,
                    "tool_names": self.agent_executor.tools_names,
                    "tools": self.agent_executor.tools_description,
                },
                version="v1",
            ):
                    if self.stop_generating_check(step):
                        self.step_callback(f"🛑 Stopped generating.", "message", True, str(uuid.uuid4()), datetime.now().timestamp() * 1000, "inline")
                        return
                    step = step + 1

                    kind = event["event"]
                    logging.debug(f"{kind}:\n{event}", flush=True)
                    match kind:

                        # message chunk
                        case "on_chat_model_stream":
                            content = event['data']['chunk'].content
                            chunk = repr(content)
                            self.step_callback(content, "message", first, chunkId, datetime.now().timestamp() * 1000, "bubble", agent_name)
                            first = False
                            logging.debug(f"Text chunkId ({chunkId}): {chunk}", flush=True)
                            acc += content
                            result += chunk

                        # praser chunk
                        case "on_parser_stream":
                            logging.debug(f"Parser chunk ({kind}): {event['data']['chunk']}", flush=True)

                        # all done
                        case "on_llm_end":
                            logging.debug(f"{kind}:\n{event}", flush=True)
                            self.step_callback("", "terminate")

                        # agent started, get their name
                        case "on_chain_start":
                            if not agent_name or len(agent_name) == 0:
                                agent_name = self.name
                                # agent_name = event["name"]

                        # tool chat message finished
                        case "on_chain_end":
                            # self.step_callback(acc, "message", True, chunkId, datetime.now().timestamp() * 1000, "bubble", agent_name)
                            chunkId = str(uuid.uuid4())
                            first = True

                        # tool started being used
                        case "on_tool_start":
                            logging.info(f"{kind}:\n{event}", flush=True)
                            tool_chunkId = str(uuid.uuid4()) #TODO:
                            tool_name = event.get('name').replace('_', ' ').capitalize()
                            self.step_callback(f"Using tool: {tool_name}", "message", True, tool_chunkId, datetime.now().timestamp() * 1000, "inline")

                        # tool finished being used
                        case "on_tool_end":
                            logging.debug(f"{kind}:\n{event}", flush=True)
                            tool_name = event.get('name').replace('_', ' ').capitalize()
                            self.step_callback(f"Finished using tool: {tool_name}", "message", True, tool_chunkId, datetime.now().timestamp() * 1000, "inline", None, True)
                            if tool_name == '_Exception' or tool_name == 'exception' or tool_name == 'invalid_tool':
                                self.step_callback(f"""Tool usage failed:
```
{json.dumps(event.get('data'), indent=4)}
```
""", "message", True, str(uuid.uuid4()), datetime.now().timestamp() * 1000, "bubble")
                        # see https://python.langchain.com/docs/expression_language/streaming#event-reference
                        case _:
                            logging.debug(f"unhandled {kind} event", flush=True)
            self.step_callback(f"**Completed task**", "message", True, task_chunkId, datetime.now().timestamp() * 1000, "inline", None, True)
        except Exception as chunk_error:
            import sys, traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            logging.error(err_lines)
            tool_chunkId = str(uuid.uuid4())
            # chunkId = str(uuid.uuid4())
            self.step_callback(f"⛔ An unexpected error occurred", "message", True, str(uuid.uuid4()), datetime.now().timestamp() * 1000, "inline")
            #TODO: if debug:
            self.step_callback(f"""Stack trace:
```
{chunk_error}
```
""", "message", True, str(uuid.uuid4()), datetime.now().timestamp() * 1000, "bubble")
            pass
        result_queue.put(acc)

    def set_cache_handler(self, cache_handler: CacheHandler) -> None:
        """Set the cache handler for the agent.

        Args:
            cache_handler: An instance of the CacheHandler class.
        """
        self.tools_handler = ToolsHandler()
        if self.cache:
            self.cache_handler = cache_handler
            self.tools_handler.cache = cache_handler
        self.create_agent_executor()

    def set_rpm_controller(self, rpm_controller: RPMController) -> None:
        """Set the rpm controller for the agent.

        Args:
            rpm_controller: An instance of the RPMController class.
        """
        if not self._rpm_controller:
            self._rpm_controller = rpm_controller
            self.create_agent_executor()

    def create_agent_executor(self, tools=None) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the CrewAgentExecutor class.
        """
        tools = tools or self.tools

        agent_args = {
            "input": lambda x: x["input"],
            "tools": lambda x: x["tools"],
            "tool_names": lambda x: x["tool_names"],
            "agent_scratchpad": lambda x: self.format_log_to_str(
                x["intermediate_steps"]
            ),
        }

        executor_args = {
            "llm": self.llm,
            "i18n": self.i18n,
            "crew": self.crew,
            "crew_agent": self,
            "tools": self._parse_tools(tools),
            "verbose": self.verbose,
            "original_tools": tools,
            "handle_parsing_errors": True,
            "max_iterations": self.max_iter,
            "max_execution_time": self.max_execution_time,
            "step_callback": self.step_callback,
            "tools_handler": self.tools_handler,
            "function_calling_llm": self.function_calling_llm or self.llm,
            "callbacks": self.callbacks,
        }

        if self._rpm_controller:
            executor_args["request_within_rpm_limit"] = (
                self._rpm_controller.check_or_wait
            )

        prompt = Prompts(
            i18n=self.i18n,
            tools=tools,
            system_template=self.system_template,
            prompt_template=self.prompt_template,
            response_template=self.response_template,
        ).task_execution()

        execution_prompt = prompt.partial(
            goal=self.goal,
            role=self.role,
            backstory=self.backstory,
        )

        stop_words = [self.i18n.slice("observation")]
        if self.response_template:
            stop_words.append(
                self.response_template.split("{{ .Response }}")[1].strip()
            )

        bind = self.llm.bind(stop=stop_words)

        parser_class = self.get_parser_class_for_llm()
        inner_agent = agent_args | execution_prompt | bind | parser_class(agent=self)

        self.agent_executor = CrewAgentExecutor(
            agent=RunnableAgent(runnable=inner_agent), **executor_args
        )

    def get_parser_class_for_llm(self) -> Type[ReActSingleInputOutputParser]:
        return GeminiAgentParser if self._llm_is_gemini() else CrewAgentParser

    def _llm_is_gemini(self) -> bool:
        # not using isinstance() to avoid import and dependency on langchain-google-vertexai
        return "model_name='gemini" in str(self.llm)

    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Interpolate inputs into the agent description and backstory."""
        if self._original_role is None:
            self._original_role = self.role
        if self._original_goal is None:
            self._original_goal = self.goal
        if self._original_backstory is None:
            self._original_backstory = self.backstory

        if inputs:
            self.role = self._original_role.format(**inputs)
            self.goal = self._original_goal.format(**inputs)
            self.backstory = self._original_backstory.format(**inputs)

    def increment_formatting_errors(self) -> None:
        """Count the formatting errors of the agent."""
        self.formatting_errors += 1

    def format_log_to_str(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        observation_prefix: str = "Observation: ",
        llm_prefix: str = "",
    ) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
        return thoughts

    def _parse_tools(self, tools: List[Any]) -> List[LangChainTool]:
        """Parse tools to be used for the task."""
        # tentatively try to import from crewai_tools import BaseTool as CrewAITool
        tools_list = []
        try:
            from crewai_tools import BaseTool as CrewAITool

            for tool in tools:
                if isinstance(tool, CrewAITool):
                    tools_list.append(tool.to_langchain())
                else:
                    tools_list.append(tool)
        except ModuleNotFoundError:
            for tool in tools:
                tools_list.append(tool)
        return tools_list

    @staticmethod
    def __tools_names(tools) -> str:
        return ", ".join([t.name for t in tools])

    def __repr__(self):
        return f"Agent(role={self.role}, goal={self.goal}, backstory={self.backstory})"
