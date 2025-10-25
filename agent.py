# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Literal, Optional, Union, Any, Type
from google import genai
from google.genai import types
import termcolor
from google.genai.types import (
    Part,
    GenerateContentConfig,
    Content,
    Candidate,
    FunctionResponse,
    FinishReason,
)
import json
from pydantic import BaseModel
import time
from rich.console import Console
from rich.table import Table

from client import EnvState,PlaywrightComputer

MAX_RECENT_TURN_WITH_SCREENSHOTS = 3
PREDEFINED_COMPUTER_USE_FUNCTIONS = [
    "open_web_browser",
    "click_at",
    "hover_at",
    "type_text_at",
    "scroll_document",
    "scroll_at",
    "wait_5_seconds",
    "go_back",
    "go_forward",
    "search",
    "navigate",
    "key_combination",
    "drag_and_drop",
]


console = Console()
FunctionResponseT = Union[EnvState, dict]


def multiply_numbers(x: float, y: float) -> dict:
    """Multiplies two numbers."""
    return {"result": x * y}


class BrowserAgent:
    def __init__(
        self,
        browser_computer: PlaywrightComputer,
        api_key: str,
        query: str,
        model_name: str,
        verbose: bool = True,
        debug: bool = False,
        max_total_input_tokens: Optional[int] = None,
        max_total_output_tokens: Optional[int] = None,
        response_schema: Optional[Type[BaseModel]] = None,
    ):
        self._browser_computer = browser_computer
        self._query = query
        self._model_name = model_name
        self._verbose = verbose
        self._response_schema = response_schema
        self.final_reasoning = None
        self._client = genai.Client(
            api_key=api_key
        )

        # 初期プロンプトを設定
        initial_prompt = self._query
        if self._response_schema:
            # response_schemaがある場合は構造化出力の指示を追加
            schema_fields = self._response_schema.model_fields
            field_descriptions = []
            for field_name, field_info in schema_fields.items():
                field_type = str(field_info.annotation)
                field_descriptions.append(f"- {field_name}: {field_type}")

            initial_prompt = f"""{self._query}

タスクが完了したら、必ずfinalize_task関数を呼び出して結果をまとめてください。
関数のパラメータは以下の形式に従ってください:
{chr(10).join(field_descriptions)}"""
        else:
            # response_schemaがない場合もfinalize_taskを呼ぶように指示
            initial_prompt = f"""{self._query}

タスクが完了したら、必ずfinalize_task関数を呼び出してタスクの完了を報告してください。
関数のパラメータ:
- summary: タスクの実行結果の要約（文字列）"""

        self._contents: list[Content] = [
            Content(
                role="user",
                parts=[
                    Part(text=initial_prompt),
                ],
            )
        ]
        self._debug = debug
        self._max_total_input_tokens = max_total_input_tokens
        self._max_total_output_tokens = max_total_output_tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._structured_result = None  # 構造化出力の結果を保存

        # Exclude any predefined functions here.
        excluded_predefined_functions = []

        # Add your own custom functions here.
        custom_functions = [
            # For example:
            types.FunctionDeclaration.from_callable(
                client=self._client, callable=multiply_numbers
            )
        ]

        # finalize_task関数を追加（response_schemaの有無に関わらず）
        finalize_function = self._create_finalize_function()
        custom_functions.append(
            types.FunctionDeclaration.from_callable(
                client=self._client, callable=finalize_function
            )
        )

        self._generate_content_config = GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER,
                        excluded_predefined_functions=excluded_predefined_functions,
                    ),
                ),
                types.Tool(function_declarations=custom_functions),
            ],
        )

    def _create_finalize_function(self):
        """タスク完了関数を動的に生成"""
        if self._response_schema:
            # response_schemaがある場合は構造化出力用の関数
            def finalize_task(**kwargs) -> dict:
                """タスクを完了し、結果を構造化された形式で返す"""
                # Pydanticモデルに変換して保存
                try:
                    self._structured_result = self._response_schema(**kwargs)
                except Exception as e:
                    # バリデーションエラーの場合は生データを保存
                    self._structured_result = kwargs
                    if self._verbose:
                        termcolor.cprint(f"スキーマ検証エラー: {e}", color="yellow")

                return {"status": "TASK_COMPLETED", "message": "タスクを完了しました"}

            # 関数のドキュメントをスキーマに基づいて設定
            schema_fields = self._response_schema.model_fields
            field_descriptions = []
            for field_name, field_info in schema_fields.items():
                field_desc = f"- {field_name}: {field_info.annotation}"
                if field_info.description:
                    field_desc += f" - {field_info.description}"
                field_descriptions.append(field_desc)

            finalize_task.__doc__ = f"""タスクを完了し、結果を以下の形式で返す:
{chr(10).join(field_descriptions)}"""
        else:
            # response_schemaがない場合はテキストサマリーを受け取る関数
            def finalize_task(summary: str) -> dict:
                """タスクを完了し、結果のサマリーを返す"""
                # サマリーを保存
                self._structured_result = summary
                return {"status": "TASK_COMPLETED", "message": "タスクを完了しました"}

            finalize_task.__doc__ = """タスクを完了し、結果のサマリーを返す:
- summary: タスクの実行結果の要約（文字列）"""

        return finalize_task

    def handle_action(self, action: types.FunctionCall) -> FunctionResponseT:
        """Handles the action and returns the environment state."""
        if action.name == "open_web_browser":
            return self._browser_computer.open_web_browser()
        elif action.name == "click_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            return self._browser_computer.click_at(
                x=x,
                y=y,
            )
        elif action.name == "hover_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            return self._browser_computer.hover_at(
                x=x,
                y=y,
            )
        elif action.name == "type_text_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            press_enter = action.args.get("press_enter", False)
            clear_before_typing = action.args.get("clear_before_typing", True)
            return self._browser_computer.type_text_at(
                x=x,
                y=y,
                text=action.args["text"],
                press_enter=press_enter,
                clear_before_typing=clear_before_typing,
            )
        elif action.name == "scroll_document":
            return self._browser_computer.scroll_document(action.args["direction"])
        elif action.name == "scroll_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            magnitude = action.args.get("magnitude", 800)
            direction = action.args["direction"]

            if direction in ("up", "down"):
                magnitude = self.denormalize_y(magnitude)
            elif direction in ("left", "right"):
                magnitude = self.denormalize_x(magnitude)
            else:
                raise ValueError("Unknown direction: ", direction)
            return self._browser_computer.scroll_at(
                x=x, y=y, direction=direction, magnitude=magnitude
            )
        elif action.name == "wait_5_seconds":
            return self._browser_computer.wait_5_seconds()
        elif action.name == "go_back":
            return self._browser_computer.go_back()
        elif action.name == "go_forward":
            return self._browser_computer.go_forward()
        elif action.name == "search":
            return self._browser_computer.search()
        elif action.name == "navigate":
            return self._browser_computer.navigate(action.args["url"])
        elif action.name == "key_combination":
            return self._browser_computer.key_combination(
                action.args["keys"].split("+")
            )
        elif action.name == "drag_and_drop":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            destination_x = self.denormalize_x(action.args["destination_x"])
            destination_y = self.denormalize_y(action.args["destination_y"])
            return self._browser_computer.drag_and_drop(
                x=x,
                y=y,
                destination_x=destination_x,
                destination_y=destination_y,
            )
        # Handle the custom function declarations here.
        elif action.name == multiply_numbers.__name__:
            return multiply_numbers(x=action.args["x"], y=action.args["y"])
        elif action.name == "finalize_task":
            # finalize_task関数の処理（_create_finalize_functionで動的に作成された関数）
            return self._create_finalize_function()(**action.args)
        else:
            raise ValueError(f"Unsupported function: {action}")

    def get_model_response(
        self, max_retries=5, base_delay_s=1
    ) -> types.GenerateContentResponse:
        for attempt in range(max_retries):
            try:
                if self._debug:
                    termcolor.cprint(
                            "DEBUG: Current contents sent to model:",
                            color="white",
                    )
                    termcolor.cprint(
                            self._contents,
                            color="white",
                    )

                    termcolor.cprint(
                        f"DEBUG: Generating content with model :",
                        color="yellow",
                    )
                    termcolor.cprint(
                        response.model_dump_json(indent=2),
                        color="yellow",
                    )
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=self._contents,
                    config=self._generate_content_config,
                )

                return response  # Return response on success
            except Exception as e:
                print(e)
                if attempt < max_retries - 1:
                    delay = base_delay_s * (2**attempt)
                    message = (
                        f"Generating content failed on attempt {attempt + 1}. "
                        f"Retrying in {delay} seconds...\n"
                    )
                    termcolor.cprint(
                        message,
                        color="yellow",
                    )
                    time.sleep(delay)
                else:
                    termcolor.cprint(
                        f"Generating content failed after {max_retries} attempts.\n",
                        color="red",
                    )
                    raise

    def get_text(self, candidate: Candidate) -> Optional[str]:
        """Extracts the text from the candidate."""
        if not candidate.content or not candidate.content.parts:
            return None
        text = []
        for part in candidate.content.parts:
            if part.text:
                text.append(part.text)
        return " ".join(text) or None

    def extract_function_calls(self, candidate: Candidate) -> list[types.FunctionCall]:
        """Extracts the function call from the candidate."""
        if not candidate.content or not candidate.content.parts:
            return []
        ret = []
        for part in candidate.content.parts:
            if part.function_call:
                ret.append(part.function_call)
        return ret

    def run_one_iteration(self) -> Literal["COMPLETE", "CONTINUE"]:
        # Generate a response from the model.
        if self._verbose:
            with console.status(
                "Generating response from Gemini Computer Use...", spinner_style=None
            ):
                try:
                    response = self.get_model_response()
                except Exception as e:
                    return "COMPLETE"
        else:
            try:
                response = self.get_model_response()
            except Exception as e:
                return "COMPLETE"

        # トークン使用量を更新
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            if hasattr(response.usage_metadata, 'prompt_token_count'):
                self.total_input_tokens += (response.usage_metadata.prompt_token_count or 0)
            if hasattr(response.usage_metadata, 'candidates_token_count'):
                self.total_output_tokens += (response.usage_metadata.candidates_token_count or 0)

            # トークン制限チェック
            if self._max_total_input_tokens and self.total_input_tokens > self._max_total_input_tokens:
                termcolor.cprint(
                    f"入力トークン制限を超過しました: {self.total_input_tokens} > {self._max_total_input_tokens}",
                    color="red",
                )
                return "COMPLETE"

            if self._max_total_output_tokens and self.total_output_tokens > self._max_total_output_tokens:
                termcolor.cprint(
                    f"出力トークン制限を超過しました: {self.total_output_tokens} > {self._max_total_output_tokens}",
                    color="red",
                )
                return "COMPLETE"

        if not response.candidates:
            print("Response has no candidates!")
            print(response)
            raise ValueError("Empty response")

        # Extract the text and function call from the response.
        candidate = response.candidates[0]
        # Append the model turn to conversation history.
        if candidate.content:
            self._contents.append(candidate.content)
        elif candidate.finish_reason == FinishReason.PROHIBITED_CONTENT:
            self._contents.append(
                Content(
                    role="user",
                    parts=[
                        Part(
                            text="The model's response was blocked due to prohibited content."
                        )
                    ],
                )
            )
            return "CONTINUE"



        reasoning = self.get_text(candidate)
        function_calls = self.extract_function_calls(candidate)

        # Retry the request in case of malformed FCs.
        if (
            not function_calls
            and not reasoning
            and candidate.finish_reason == FinishReason.MALFORMED_FUNCTION_CALL
        ):
            return "CONTINUE"



        function_call_strs = []
        for function_call in function_calls:
            # Print the function call and any reasoning.
            function_call_str = f"Name: {function_call.name}"
            if function_call.args:
                function_call_str += f"\nArgs:"
                for key, value in function_call.args.items():
                    function_call_str += f"\n  {key}: {value}"
            function_call_strs.append(function_call_str)

        table = Table(expand=True)
        table.add_column(
            "Gemini Computer Use Reasoning", header_style="magenta", ratio=1
        )
        table.add_column("Function Call(s)", header_style="cyan", ratio=1)
        table.add_row(reasoning, "\n".join(function_call_strs))
        if self._verbose:
            console.print(table)

            # トークン使用量を表示
            token_info = f"トークン使用量: 入力: {self.total_input_tokens:,}"
            if self._max_total_input_tokens:
                token_info += f" / {self._max_total_input_tokens:,}"
            token_info += f" | 出力: {self.total_output_tokens:,}"
            if self._max_total_output_tokens:
                token_info += f" / {self._max_total_output_tokens:,}"

            # 制限に近づいた場合は色を変える
            color = "green"
            if self._max_total_input_tokens and self.total_input_tokens > self._max_total_input_tokens * 0.8:
                color = "yellow"
            if self._max_total_output_tokens and self.total_output_tokens > self._max_total_output_tokens * 0.8:
                color = "yellow"

            termcolor.cprint(token_info, color=color)
            print()

        if not function_calls:
            print(f"ループ終了。理由: {candidate.finish_reason}")
            self.final_reasoning = reasoning
            return "COMPLETE"

        function_responses = []
        for function_call in function_calls:
            # finalize_task関数が呼ばれた場合は特別な処理
            if function_call.name == "finalize_task":
                fc_result = self.handle_action(function_call)
                if fc_result.get("status") == "TASK_COMPLETED":
                    if self._verbose:
                        termcolor.cprint(
                            f"タスク完了: {fc_result.get('message')}",
                            color="green",
                        )
                    return "COMPLETE"

            extra_fr_fields = {}
            if function_call.args and (
                safety := function_call.args.get("safety_decision")
            ):
                decision = self._get_safety_confirmation(safety)
                if decision == "TERMINATE":
                    print("Terminating agent loop")
                    return "COMPLETE"
                # Explicitly mark the safety check as acknowledged.
                extra_fr_fields["safety_acknowledgement"] = "true"
            if self._verbose:
                with console.status(
                    "Sending command to Computer...", spinner_style=None
                ):
                    fc_result = self.handle_action(function_call)
            else:
                fc_result = self.handle_action(function_call)
            if isinstance(fc_result, EnvState):
                function_responses.append(
                    FunctionResponse(
                        name=function_call.name,
                        response={
                            "url": fc_result.url,
                            **extra_fr_fields,
                        },
                        parts=[
                            types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    mime_type="image/png", data=fc_result.screenshot
                                )
                            )
                        ],
                    )
                )
            elif isinstance(fc_result, dict):
                function_responses.append(
                    FunctionResponse(name=function_call.name, response=fc_result)
                )

        self._contents.append(
            Content(
                role="user",
                parts=[Part(function_response=fr) for fr in function_responses],
            )
        )

        # only keep screenshots in the few most recent turns, remove the screenshot images from the old turns.
        turn_with_screenshots_found = 0
        for content in reversed(self._contents):
            if content.role == "user" and content.parts:
                # check if content has screenshot of the predefined computer use functions.
                has_screenshot = False
                for part in content.parts:
                    if (
                        part.function_response
                        and part.function_response.parts
                        and part.function_response.name
                        in PREDEFINED_COMPUTER_USE_FUNCTIONS
                    ):
                        has_screenshot = True
                        break

                if has_screenshot:
                    turn_with_screenshots_found += 1
                    # remove the screenshot image if the number of screenshots exceed the limit.
                    if turn_with_screenshots_found > MAX_RECENT_TURN_WITH_SCREENSHOTS:
                        for part in content.parts:
                            if (
                                part.function_response
                                and part.function_response.parts
                                and part.function_response.name
                                in PREDEFINED_COMPUTER_USE_FUNCTIONS
                            ):
                                part.function_response.parts = None


        return "CONTINUE"

    def _get_safety_confirmation(
        self, safety: dict[str, Any]
    ) -> Literal["CONTINUE", "TERMINATE"]:
        if safety["decision"] != "require_confirmation":
            raise ValueError(f"Unknown safety decision: safety['decision']")
        termcolor.cprint(
            "Safety service requires explicit confirmation!",
            color="yellow",
            attrs=["bold"],
        )
        print(safety["explanation"])
        decision = ""
        while decision.lower() not in ("y", "n", "ye", "yes", "no"):
            decision = input("Do you wish to proceed? [Yes]/[No]\n")
        if decision.lower() in ("n", "no"):
            return "TERMINATE"
        return "CONTINUE"

    def agent_loop(self):
        status = "CONTINUE"
        while status == "CONTINUE":
            status = self.run_one_iteration()

        # response_schemaがある場合は構造化結果を返す
        return self._structured_result

    def denormalize_x(self, x: int) -> int:
        return int(x / 1000 * self._browser_computer.screen_size()[0])

    def denormalize_y(self, y: int) -> int:
        return int(y / 1000 * self._browser_computer.screen_size()[1])