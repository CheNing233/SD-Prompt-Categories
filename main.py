import gradio as gr
import os
import re
import json
import sys
import openai

CONFIG_FILE = "config.json"


# 重启应用函数
def restart_app():
    """重启Gradio应用"""
    python = sys.executable
    os.execl(python, python, *sys.argv)


# --- 配置管理 ---
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # 创建默认配置
        default_config = {
            "categories": [
                {"name": "Poses", "path": "Poses"},
                {"name": "Clothes", "path": "Clothes"},
                {"name": "Others", "path": "Others"},
            ],
            "api_key": "",
            "base_url": "",
            "system_prompt": "你是AI分类助手",
            "model": "deepseek-chat"
        }
        save_config(default_config)
        return default_config


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


# --- 核心逻辑 ---
def load_category_words(folder_path):
    words = set()
    if not os.path.isdir(folder_path):
        return words
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                words.update(line.strip() for line in f if line.strip())
    return words


def get_all_words(config, replace_underscore=False):
    all_words = {}
    for category in config["categories"]:
        name = category["name"]
        path = category["path"]
        words = load_category_words(path)
        if replace_underscore:
            all_words[name] = {word.replace("_", " ") for word in words}
        else:
            all_words[name] = words
    return all_words


def extract_core_word(part):
    part = part.strip()
    while len(part) > 1 and (
        (part.startswith("(") and part.endswith(")"))
        or (part.startswith("[") and part.endswith("]"))
        or (part.startswith("{") and part.endswith("}"))
    ):
        part = part[1:-1].strip()
    part = re.sub(r":\s*[-+]?\d*\.?\d+\s*$", "", part).strip()
    return part


def classify_prompt(text, use_fuzzy, replace_underscore, config):
    parts = re.split(",", text)
    category_words = get_all_words(config, replace_underscore)

    # 添加"未分类"类别
    results = {category["name"]: [] for category in config["categories"]}
    results["未分类"] = []  # 新增未分类列表

    for part in parts:
        part = part.strip()
        raw_part = extract_core_word(part).strip()
        if not raw_part:
            continue

        matched = False
        for category_name, words in category_words.items():
            if use_fuzzy:
                for word in words:
                    if word in raw_part or raw_part in word:
                        results[category_name].append(part)
                        matched = True
                        break
            else:
                if raw_part in words:
                    results[category_name].append(part)
                    matched = True
            if matched:
                break

        # 如果没有匹配任何类别，放入未分类
        if not matched:
            results["未分类"].append(part)

    # 返回所有类别结果和未分类结果
    output = [", ".join(results[cat["name"]]) for cat in config["categories"]]
    output.append(", ".join(results["未分类"]))  # 添加未分类结果
    return output


def save_results(*args):
    config = load_config()
    fast_save = args[0]
    output_boxes = args[1:-1]  # 排除最后一个未分类框

    def save_unique(category_name, text_to_save):
        # 判断文件是否存在
        is_file_exists = os.path.exists(f"extract_{category_name}.txt")

        if fast_save:
            with open(f"extract_{category_name}.txt", "a", encoding="utf-8") as f:
                f.write("" if not is_file_exists else "\n" + text_to_save)
        else:
            # 判断文件是否存在
            if is_file_exists:
                with open(f"extract_{category_name}.txt", "r", encoding="utf-8") as f:
                    existing = [line.strip() for line in f if line.strip()]
            else:
                existing = []

            to_save = set(existing + [text_to_save])

            with open(f"extract_{category_name}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(to_save))
        return "保存成功！"

    for i, category in enumerate(config["categories"]):
        save_unique(category["name"], output_boxes[i])

    gr.Info("保存成功！")

    return "保存成功！"


# --- UI界面 ---
def create_ui(config):
    with gr.Blocks() as demo:
        config_state = gr.State(config)

        with gr.Row():
            input_text = gr.Textbox(
                label="输入提示词", placeholder="请输入逗号分隔的提示词...", scale=4
            )

        output_boxes = []
        # 每行最多显示3个分类
        for i in range(0, len(config["categories"]), 3):
            with gr.Row():
                for category in config["categories"][i : i + 3]:
                    output_boxes.append(gr.Textbox(label=category["name"]))

        # 新增未分类文本框（总是放在最后一行）
        with gr.Row():
            unclassified_box = gr.Textbox(label="未分类")
            output_boxes.append(unclassified_box)

        with gr.Row():
            fuzzy_checkbox = gr.Checkbox(
                label="双向模糊匹配", info="启用时：关键词互为子串即匹配（有BUG，不建议启用）"
            )
            replace_underscore_checkbox = gr.Checkbox(
                value=True, label="识别空格类提示词", info="启用时：将按照空格类提示词进行字典匹配，关闭时：将按照下划线类提示词进行字典匹配"
            )
            fast_save = gr.Checkbox(
                value=False, label="快速保存", info="启用时：保存时不查重"
            )

        with gr.Row():
            classify_btn = gr.Button("分类")
            ai_classify_btn = gr.Button("未分类部分进行AI分类")
            save_btn = gr.Button("保存结果")

        ai_result_box = gr.Textbox(label="AI分类结果", interactive=True)
        result_msg = gr.Textbox(label="操作结果", interactive=False)

        with gr.Accordion("AI分类配置", open=False):
            with gr.Row():
                api_key_box = gr.Textbox(label="API Key", value=config.get("api_key", ""), type="password")
                base_url_box = gr.Textbox(label="Base URL", value=config.get("base_url", ""))
            system_prompt_box = gr.Textbox(label="System Prompt", value=config.get("system_prompt", "你是AI分类助手"))
            model_box = gr.Textbox(label="Model", value=config.get("model", "deepseek-chat"))
            with gr.Row():
                save_ai_config_btn = gr.Button("保存AI配置")

        with gr.Accordion("分类配置管理", open=False):
            with gr.Row():
                new_cat_name = gr.Textbox(
                    label="新分类名称", placeholder="例如：Emotions"
                )
                new_cat_path = gr.Textbox(
                    label="新分类文件夹路径", placeholder="例如：Emotions"
                )
                add_cat_btn = gr.Button("添加分类")

            gr.Markdown("---删除分类---")
            delete_buttons = []
            for i, category in enumerate(config["categories"]):
                with gr.Row():
                    gr.Textbox(
                        value=f"{category['name']} ({category['path']})",
                        interactive=False,
                        scale=3,
                    )
                    delete_buttons.append(
                        gr.Button(f"删除 {category['name']}", scale=1)
                    )

        # --- 事件处理 ---
        classify_btn.click(
            fn=classify_prompt,
            inputs=[
                input_text,
                fuzzy_checkbox,
                replace_underscore_checkbox,
                config_state,
            ],
            outputs=output_boxes,
        )

        save_btn.click(
            fn=save_results, inputs=[fast_save, *output_boxes], outputs=result_msg
        )

        def save_ai_config(api_key, base_url, system_prompt, model, current_config):
            current_config["api_key"] = api_key
            current_config["base_url"] = base_url
            current_config["system_prompt"] = system_prompt
            current_config["model"] = model
            save_config(current_config)
            gr.Info("AI配置已保存！")
            return current_config

        def classify_with_ai(api_key, base_url, system_prompt, model, unclassified_text):
            if not api_key or not base_url:
                gr.Warning("API Key和Base URL不能为空！")
                return ""
            if not unclassified_text:
                gr.Info("没有需要分类的内容。")
                return ""

            client = openai.OpenAI(api_key=api_key, base_url=base_url)

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"请将以下内容进行分类：\n{unclassified_text}"},
                    ],
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                gr.Error(f"AI分类失败：{e}")
                return f"错误: {e}"

        save_ai_config_btn.click(
            fn=save_ai_config,
            inputs=[api_key_box, base_url_box, system_prompt_box, model_box, config_state],
            outputs=[config_state]
        )

        ai_classify_btn.click(
            fn=classify_with_ai,
            inputs=[api_key_box, base_url_box, system_prompt_box, model_box, unclassified_box],
            outputs=[ai_result_box]
        )

        def add_category_and_reload(name, path, current_config):
            if not name or not path:
                gr.Warning("分类名称和路径不能为空！")
                return current_config, gr.update(), gr.update()
            current_config["categories"].append({"name": name, "path": path})
            save_config(current_config)
            gr.Info("已成功，请重启整个应用")
            restart_app()
            return current_config, "", ""

        add_cat_btn.click(
            fn=add_category_and_reload,
            inputs=[new_cat_name, new_cat_path, config_state],
            outputs=[config_state, new_cat_name, new_cat_path],
        )

        def delete_category_and_reload(index_to_delete, current_config):
            # 使用 gr.State 的 .value 属性来获取最新值
            current_config["categories"].pop(index_to_delete)
            save_config(current_config)
            gr.Info("已成功，请重启整个应用")
            restart_app()
            return current_config

        for i, btn in enumerate(delete_buttons):
            # 使用 functools.partial 来正确捕获循环变量
            from functools import partial

            btn.click(
                fn=partial(delete_category_and_reload, i),
                inputs=[config_state],
                outputs=[config_state],
            )

    return demo


if __name__ == "__main__":
    config = load_config()
    demo = create_ui(config)
    demo.launch()
