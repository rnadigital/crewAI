import re

from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from ..parser import CrewAgentParser


FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = "I did it wrong. Invalid Format: I missed the 'Action:' after 'Thought:'. I will do right next, and don't use a tool I have already used.\n"
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = "I did it wrong. Invalid Format: I missed the 'Action Input:' after 'Action:'. I will do right next, and don't use a tool I have already used.\n"
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = "I did it wrong. Tried to both perform Action and give a Final Answer at the same time, I must do one or the other"

GEMINI_FALSE_POSITIVE_FINAL_ANSWER_PATTERN = re.compile('Final Answer[\s:\\n*]*\(.*\)\*?\*?')


class GeminiAgentParser(CrewAgentParser):
    """
    Handle Gemini output differently.
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        Mostly the same as the parent's `parse` method, the only difference being that it doesn't consider
        messages containing a placeholder 'Final Answer' to be the actual 'Final Answer'.
        ```
        Example:
            ### Final Answer: \n\n**(This is where the final answer will be displayed once the user's response is processed.)**
        ```
        """
        includes_answer = FINAL_ANSWER_ACTION in text and not re.search(GEMINI_FALSE_POSITIVE_FINAL_ANSWER_PATTERN, text)
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            tool_input = tool_input.strip('"')
            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            self.agent.increment_formatting_errors()
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=f"{MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE}\n{self._i18n.slice('final_answer_format')}",
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
                r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            self.agent.increment_formatting_errors()
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            format = self._i18n.slice("format_without_tools")
            error = f"{format}"
            self.agent.increment_formatting_errors()
            raise OutputParserException(
                error,
                observation=error,
                llm_output=text,
                send_to_llm=True,
            )

