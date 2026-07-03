import torch.nn as nn
import thulac
# 自定义分词器
"""_summary_

Returns:
    _type_: _description_
"""
class InputDivision(nn.Module):
    def __init__(self):
        super().__init__()
        self.division = thulac.thulac(seg_only=True)  # 初始化分词器
        # 初始化分词器参数

    def forward(self, x):
        # 实现分词逻辑
        divided = self.division.cut(x)
        tokens = [item[0] if isinstance(item, (list, tuple)) else item for item in divided]
        return self._split_digits(tokens)

    def _split_digits(self, tokens):
        split_tokens = []
        for token in tokens:
            if token is None:
                continue
            token = str(token)
            if token == "":
                continue

            if token.isdigit():
                split_tokens.extend(list(token))
                continue

            buffer = ""
            for ch in token:
                if ch.isdigit():
                    if buffer:
                        split_tokens.append(buffer)
                        buffer = ""
                    split_tokens.append(ch)
                else:
                    buffer += ch

            if buffer:
                split_tokens.append(buffer)

        return split_tokens
    
    
if __name__ == "__main__":
    division = InputDivision()
    test_input = "你正在玩《崩坏：星穹铁道》，在货币战争里打到了财富造物主40。"
    result = division(test_input)
    print(result)  # 输出分词结果 
