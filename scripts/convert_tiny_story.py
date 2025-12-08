import csv
import json
import os
import re
from typing import List, Dict

TRAIN_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'tiny_story', 'train.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'tiny_story', 'tiny_story_test.json')
SAMPLE_SIZE = 100


def clean_text(s: str) -> str:
    if s is None:
        return ''
    # 单段连续：去换行、归一空白
    s = s.replace('\r', ' ').replace('\n', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def read_stories(csv_path: str, sample_size: int) -> List[str]:
    texts: List[str] = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        # 字段候选：常见 tiny story 数据为 'text'；若不存在则尝试其他名称
        candidates = ['text', 'story', 'tiny_story', 'content']
        for i, row in enumerate(reader):
            if i >= sample_size:
                break
            value = None
            for key in candidates:
                if key in row and row.get(key):
                    value = row.get(key)
                    break
            # 若没有候选字段，回退到第一列
            if value is None and isinstance(row, dict) and len(row) > 0:
                value = list(row.values())[0]
            texts.append(clean_text(value))
    return texts


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    texts = read_stories(os.path.abspath(TRAIN_CSV_PATH), SAMPLE_SIZE)
    if not texts:
        print('No stories loaded. Please check CSV path or encoding.')
        return

    items: List[Dict] = []
    for i, t in enumerate(texts):
        items.append({'id': i, 'paragraphs': t})

    with open(os.path.abspath(OUTPUT_PATH), 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    lengths = [len(x['paragraphs']) for x in items]
    print(f'Wrote {len(items)} items to: {OUTPUT_PATH}')
    print(f'Length min/max: {min(lengths)} / {max(lengths)}')


if __name__ == '__main__':
    main()