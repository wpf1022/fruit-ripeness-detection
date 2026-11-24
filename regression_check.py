"""真实照片回归检查：三张用户实际遇到误判的照片，必须给出正确种类。

用法：python regression_check.py
"""
from __future__ import annotations

import glob
import os
import sys

from predict import predict_path

BASE = os.path.dirname(os.path.abspath(__file__))
EXPECT = {
    "banana_spotted": "banana",     # 带黑斑的成熟香蕉：曾被判“非支持类别”
    "strawberry_field": "strawberry",  # 田间草莓：曾被判葡萄/非支持类别
    "apple_red": "apple",          # 红苹果：曾被判“未熟芒果”
}


def main() -> int:
    bad = 0
    for path in sorted(glob.glob(os.path.join(BASE, "regression", "*.png"))):
        key = os.path.splitext(os.path.basename(path))[0]
        want = EXPECT.get(key)
        res = predict_path(path)
        got = res.get("species")
        ok = got == want
        bad += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {key:18s} 期望 {want:11s} 实际 {str(got):11s} "
              f"{res['label_cn']} {res['confidence']*100:.1f}%")
    print(f"\n{len(EXPECT) - bad}/{len(EXPECT)} 通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
