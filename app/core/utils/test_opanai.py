import openai


def _is_translategemma_model(model: str) -> bool:
    return "translategemma" in model.lower().replace(" ", "")


def _build_translategemma_prompt():
    return (
        "<bos><start_of_turn>user\n"
        "You are a professional English (en) to Simplified Chinese (zh-Hans) translator. "
        "Your goal is to accurately convey the meaning and nuances of the original English text "
        "while adhering to Simplified Chinese grammar, vocabulary, and cultural sensitivities.\n"
        "Produce only the Simplified Chinese translation, without any additional explanations or commentary. "
        "Please translate the following English text into Simplified Chinese:\n\n\n"
        "Hello!<end_of_turn>\n<start_of_turn>model\n"
    )


def test_openai(base_url, api_key, model):
    """
    这是一个测试OpenAI API的函数。
    它使用指定的API设置与OpenAI的GPT模型进行对话。

    参数:
    user_message (str): 用户输入的消息

    返回:
    bool: 是否成功
    str: 错误信息或者AI助手的回复
    """
    try:
        client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=10)

        if _is_translategemma_model(model):
            response = client.completions.create(
                model=model,
                prompt=_build_translategemma_prompt(),
                max_tokens=100,
                timeout=10,
            )
            return True, str(response.choices[0].text)

        # 创建OpenAI客户端并发送请求到OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            max_tokens=100,
            timeout=10,
        )
        # 返回AI的回复
        return True, str(response.choices[0].message.content)
    except Exception as e:
        return False, str(e)


def get_openai_models(base_url, api_key):
    try:
        # 创建OpenAI客户端并获取模型列表
        models = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=5
        ).models.list()

        # 根据不同模型设置权重进行排序
        def get_model_weight(model_name):
            model_name = model_name.lower()
            if model_name.startswith(("gpt-4o", "claude-3-5")):
                return 10
            elif model_name.startswith("gpt-4"):
                return 5
            elif model_name.startswith("claude-3"):
                return 6
            elif model_name.startswith(("deepseek", "glm")):
                return 3
            return 0

        sorted_models = sorted(
            [model.id for model in models], key=lambda x: (-get_model_weight(x), x)
        )
        return sorted_models
    except Exception:
        return []
