import re
from typing import Tuple, List, Optional


def extract_code_blocks(text: str, target_filename: Optional[str] = None) -> Tuple[List, str]:
    # 只匹配格式为 ```language:filename 的代码块
    pattern = r'```[a-zA-Z]*:([^\n\r`]+)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    result = []

    for filename, code in matches:
        filename = filename.strip()
        if target_filename is None or filename == target_filename:
            result.append({
                'filename': filename,
                'code': code.strip()
            })

    if target_filename is not None:
        remove_pattern = rf'```[a-zA-Z]*:{re.escape(target_filename)}\n.*?```'
    else:
        remove_pattern = pattern

    remaining_text = re.sub(remove_pattern, '', text, flags=re.DOTALL)
    remaining_text = re.sub(r'\n\s*\n\s*\n', '\n\n', remaining_text)
    remaining_text = remaining_text.strip()

    return result, remaining_text
