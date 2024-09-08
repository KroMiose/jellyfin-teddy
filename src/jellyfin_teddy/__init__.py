import json
import sys
import asyncio

from pathlib import Path
from appdirs import user_config_dir
from typing import Dict, List
from miose_toolkit_llm.clients.chat_openai import gen_openai_chat_response, set_openai_base_url, set_openai_proxy

APP_NAME = "JellyfinTeddy"
APP_AUTHOR = "KroMiose"
CONFIG_DIR = Path(user_config_dir(APP_NAME, APP_AUTHOR))
CONFIG_FILE = CONFIG_DIR / "config.json"
OPENAI_BASE_URL = "https://api.openai.com/v1"


def load_config() -> dict:
    try:
        with CONFIG_FILE.open("r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}
    return config


def save_config(config: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w") as f:
        json.dump(config, f)


def analyze_files(files: List[Path]) -> str:
    file_list_str = "\n".join(str(f) for f in files)
    prompt = (
        "你是一位具备丰富经验的媒体文件管理助手，我需要你帮助分析以下电视剧/动漫剧集文件，并按照 Jellyfin 媒体库要求的格式生成重命名计划。\n\n"
        "文件列表：\n"
        f"{file_list_str}\n\n"
        "Jellyfin 文件命名规则：\n"
        "1. 每一季的剧集必须放在对应的 `Season XX` 文件夹中，`XX` 为两位数的季号。\n"
        "2. 剧集文件命名格式应为：`<剧集名称> S<季号>E<集号>[其他描述].<扩展名>`。例如：`Series Name A S01E01-E02.mkv` 或 `Series Name A S02E03 Part 1.mkv`。\n"
        "3. 剧集的季文件夹名称不要包含剧集名称。文件夹名必须为：`Season 01`, `Season 02`，等。\n"
        "4. 特殊剧集应放在 `Season 00` 文件夹，文件名应包含对内容的简要描述。\n\n"
        "任务：分析文件并返回一个 JSON 格式的数据结构，每个文件应包含以下字段：\n"
        "- 原文件名\n"
        "- 剧集名称\n"
        "- 季号 (整数)\n"
        "- 集号 (列表，支持跨集，例如 [1, 2])\n"
        "- 其他描述 (如有)\n\n"
        "返回格式示例：\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "原文件名": "Series Name A S01E01-E02.mkv",\n'
        '    "剧集名称": "Series Name A",\n'
        '    "季号": 1,\n'
        '    "集号": [1, 2],\n'
        '    "其他描述": ""\n'
        "  },\n"
        "  {\n"
        '    "原文件名": "Series Name A S02E03 Part 1.mkv",\n'
        '    "剧集名称": "Series Name A",\n'
        '    "季号": 2,\n'
        '    "集号": [3],\n'
        '    "其他描述": "Part 1"\n'
        "  }\n"
        "]\n"
        "```"
    )
    return prompt


def generate_rename_plan(analyzed_data: List[Dict]) -> List[Dict]:
    """生成 Jellyfin 媒体库重命名计划。"""
    rename_plan = []
    for item in analyzed_data:
        season = str(item["季号"]).zfill(2)
        if len(item["集号"]) > 1:
            episode_range = f"{str(item['集号'][0]).zfill(2)}-{str(item['集号'][-1]).zfill(2)}"
        else:
            episode_range = str(item["集号"][0]).zfill(2)

        original_file = Path(item["原文件名"])
        new_filename = f"{item['剧集名称']} S{season}E{episode_range}{original_file.suffix}"
        new_filepath = Path(f"Season {season}") / new_filename

        rename_plan.append({"原路径": original_file, "新路径": new_filepath})
    return rename_plan


def execute_rename(rename_plan: List[Dict]):
    """执行 Jellyfin 媒体库文件重命名操作。"""
    for item in rename_plan:
        original_path = item["原路径"]
        new_path = item["新路径"]

        try:
            # 创建目标路径中的目录
            new_path.parent.mkdir(parents=True, exist_ok=True)

            # 重命名文件
            original_path.rename(new_path)
            print(f"已重命名: {original_path} -> {new_path}")
        except Exception as e:
            print(f"重命名失败: {original_path} -> {new_path}, 错误: {e}")


# CLI 入口
async def cli():
    config = load_config() if "--config" not in sys.argv else {}

    # 检查并获取 API Key
    if "api_key" not in config:
        config["api_key"] = input("请输入您的 OpenAI API Key: ")
        save_config(config)

    # 检查并获取 API Base URL (可选)
    if "base_url" not in config:
        config["base_url"] = input("请输入您的 OpenAI API Base URL (可选，留空为官方 API): ") or "https://api.openai.com/v1"
        save_config(config)

    if "proxy" not in config:
        config["proxy"] = input("请输入代理地址 (可选，留空不使用代理): ")
        save_config(config)

    if "model" not in config:
        config["model"] = input("请输入您要使用的模型名称 (默认: gpt-3.5-turbo): ") or "gpt-3.5-turbo"
        save_config(config)

    # 初始化 OpenAI 客户端
    set_openai_proxy(config.get("proxy", None))
    set_openai_base_url(config["base_url"])

    # 列出当前目录下的文件
    current_directory = Path(".")
    files = [f for f in current_directory.iterdir() if f.is_file() and not f.name.startswith(".")]

    # 按照文件名排序
    files.sort(key=lambda f: f.name)

    if not files:
        print("当前目录没有可用的文件。")
        return

    # 列出需要整理的文件
    print(f"当前目录下共有 {len(files)} 个文件：")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")

    # 分析文件并生成 LLM 提示
    prompt = analyze_files(files)

    # 调用 LLM API
    response_text, _ = await gen_openai_chat_response(
        model=config["model"],
        messages=[{"role": "user", "content": prompt}],
        api_key=config["api_key"],
    )

    if response_text.lower().startswith("```json"):
        response_text = response_text[7:-3].strip()

    try:
        analyzed_data = json.loads(response_text)
    except json.JSONDecodeError:
        print("原始结果:", response_text)
        print("LLM 返回结果格式错误，请检查提示词是否合理。")
        return

    # 展示分析结果
    print("Analyzed Data:", analyzed_data)

    # 生成并展示重命名计划
    rename_plan = generate_rename_plan(analyzed_data)

    print("\n重命名计划：")
    for item in rename_plan:
        print(f"{item['原路径']}  ->  {item['新路径']}")

    # 检查是否存在冲突文件
    conflict_files = [item["新路径"] for item in rename_plan if item["新路径"].exists()]
    if conflict_files:
        print("\n以下文件存在冲突，请尝试重新生成命名：")

        for f in conflict_files:
            print(f)
        return

    # 检查命名结果是否存在重复文件
    new_filenames = [item["新路径"].name for item in rename_plan]
    if len(new_filenames) != len(set(new_filenames)):
        print("\n重命名计划存在重复文件，请尝试重新生成命名。")
        return

    # 用户确认是否执行
    confirm = input("\n确认执行重命名计划? (y/n): ").lower()
    if confirm == "y":
        execute_rename(rename_plan)
        print("剧集整理完成！")
    else:
        print("已取消操作。")


def main():
    asyncio.run(cli())
