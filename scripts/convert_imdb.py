import csv
import json
import os
import re
from typing import List, Dict

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'imdb', 'IMDB Dataset.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'imdb', 'imdb_test.json')
# 已移除文本级 <pad> 填充相关参数
SAMPLE_SIZE = 100


def clean_text(s: str) -> str:
    if s is None:
        return ''
    # 单段连续：去换行、归一空白
    s = s.replace('\r', ' ').replace('\n', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    return s


# 移除 pad_to_len 函数及其调用，保持原文本


def read_reviews(csv_path: str, sample_size: int) -> List[str]:
    texts: List[str] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= sample_size:
                break
            review = clean_text(row.get('review'))
            # 忽略 sentiment
            texts.append(review)
    return texts


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    texts = read_reviews(os.path.abspath(CSV_PATH), SAMPLE_SIZE)
    if not texts:
        print('No reviews loaded. Please check CSV path or encoding.')
        return

    items: List[Dict] = []
    for i, t in enumerate(texts):
        items.append({
            'id': i,
            'paragraphs': t,
        })

    with open(os.path.abspath(OUTPUT_PATH), 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    # 简要校验输出长度范围
    lengths = [len(x['paragraphs']) for x in items]
    print(f'Wrote {len(items)} items to: {OUTPUT_PATH}')
    print(f'Length min/max: {min(lengths)} / {max(lengths)}')


if __name__ == '__main__':
    main()