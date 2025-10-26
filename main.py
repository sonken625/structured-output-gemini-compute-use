from client import PlaywrightComputer
from agent import BrowserAgent
from pydantic import BaseModel
from typing import Optional
import os

PLAYWRIGHT_SCREEN_SIZE = (1440, 900)


def run_agent(
    query: str,
    api_key: str,
    max_input_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    remote_debugging_port: Optional[int] = None,
    response_schema=None,
) -> int:
    env = PlaywrightComputer(
        screen_size=PLAYWRIGHT_SCREEN_SIZE,
        initial_url="https://www.google.com",
        highlight_mouse=True,
        remote_debugging_port=remote_debugging_port,
        close_on_exit=False,
    )
    with env as browser_computer:
        agent = BrowserAgent(
            browser_computer=browser_computer,
            api_key=api_key,
            query=query,
            model_name="gemini-2.5-computer-use-preview-10-2025",
            max_total_input_tokens=max_input_tokens,
            max_total_output_tokens=max_output_tokens,
            response_schema=response_schema,  # response_schemaをBrowserAgentに渡す
        )
        result = agent.agent_loop()  # agent_loopに引数は不要

    return result


if __name__ == "__main__":
    query = "googleで東京の天気を検索して何度か教えて"
    api_key = os.getenv("GEMINI_API_KEY")
    # 例1: トークン制限付きで実行
    # main(query=query, api_key=api_key,
    #      max_input_tokens=10000,
    #      max_output_tokens=5000)

    # 例2: 構造化出力を使用
    # main(query=query, api_key=api_key, response_schema=WeatherInfo)

    # 構造化出力用のスキーマ例
    class TaskResult(BaseModel):
        """タスク実行結果のスキーマ"""

        success: bool
        summary: str
        details: Optional[str] = None

    class WeatherInfo(BaseModel):
        """天気情報のスキーマ"""

        location: str
        temperature: Optional[str] = None
        weather_condition: Optional[str] = None
        additional_info: Optional[str] = None

    # 例3: トークン制限と構造化出力を両方使用
    result = run_agent(
        query=query,
        api_key=api_key,
        max_input_tokens=15000,
        max_output_tokens=10000,
        # remote_debugging_port=9222,
        response_schema=WeatherInfo,
    )

    print("Final Result:", result)
