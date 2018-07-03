from ..minimize import expected_improvement


def test_expected_improvement():
    assert expected_improvement(0, 1, 0) > expected_improvement(1, 1, 0), \
        "a lower mean should be more promising"
    assert expected_improvement(0, 2, -1) > expected_improvement(0, 1, -1), \
        "a larger std should be more promising"
    assert expected_improvement(0, 1, 0) > expected_improvement(0, 1, -1), \
        "a worse known minimum should be more promising"
