from transformers import AutoTokenizer
from llamafactory.data.template import parse_template


def test_parse_qwen_template():
    tokenizer = AutoTokenizer.from_pretrained("hf_iter_168000")
    template = parse_template(tokenizer)
    assert template.format_user.slots == [
            (
            "<|im_start|><|user|>\n{{content}}<|im_end|"
            "<|im_start|><|assistant|>\n"
            )
            ]
    assert template.format_assistant.slots == ["{{content}}<|im_end|>\n"]
    assert template.format_system.slots == ["<|im_start|><|system|>\n{{content}}<|im_end|>\n"]
    assert template.default_system == "You are an AI assistant developed by Microsoft. You are helpful for user to handle daily tasks."