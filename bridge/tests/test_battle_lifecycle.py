"""归一化匹配使用合成数组验证，不把游戏截图或模板入库。"""

import numpy as np

from bridge.battle_lifecycle import normalized_template_score


def test_template_matches_at_any_valid_offset():
    rng = np.random.default_rng(42)
    image = rng.uniform(0, 255, (16, 21, 3))
    assert normalized_template_score(image, image[5:11, 7:15].copy()) > 0.99999


def test_invalid_or_unrelated_template_is_not_a_match():
    rng = np.random.default_rng(7)
    image = rng.uniform(0, 255, (16, 21, 3))
    other = rng.uniform(0, 255, (8, 8, 3))
    assert normalized_template_score(image, other) < 0.9
    assert normalized_template_score(image, np.ones((30, 40, 3))) == 0
