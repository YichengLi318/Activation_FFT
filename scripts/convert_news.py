import json
import os
import re
from typing import List, Dict

INPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'news', 'News_Category_Dataset_v3.json')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'news', 'news_test.json')
# 已移除文本级 <pad> 填充相关参数
SAMPLE_SIZE = 100


def clean_text(s: str) -> str:
    if s is None:
        return ''
    s = s.replace('\r', ' ').replace('\n', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    return s


# 移除 pad_to_len 函数及其调用，保持原文本


def load_items(input_path: str, sample_size: int) -> List[Dict]:
    items: List[Dict] = []
    with open(os.path.abspath(input_path), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            items.append(obj)
    return items


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    raw_items = load_items(INPUT_PATH, SAMPLE_SIZE)
    if not raw_items:
        print('No news items loaded. Please check input path or encoding.')
        return

    texts: List[str] = []
    for obj in raw_items:
        headline = clean_text(obj.get('headline', ''))
        short_desc = clean_text(obj.get('short_description', ''))
        # 正文：优先使用 short_description，若存在则与 headline 拼接；否则仅 headline
        if short_desc:
            text = f"{headline}\n\n{short_desc}" if headline else short_desc
        else:
            text = headline
        texts.append(text)

    out_items: List[Dict] = []
    for i, t in enumerate(texts):
        out_items.append({'id': i, 'paragraphs': t})

    with open(os.path.abspath(OUTPUT_PATH), 'w', encoding='utf-8') as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    lengths = [len(x['paragraphs']) for x in out_items]
    print(f'Wrote {len(out_items)} items to: {OUTPUT_PATH}')
    print(f'Length min/max: {min(lengths)} / {max(lengths)}')


if __name__ == '__main__':
    main()