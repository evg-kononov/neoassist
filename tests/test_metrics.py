from neocore.metrics import precision, recall


def test_precision():
    fixture = {"tp": 0, "fp": 0}
    expected = 0.0
    actual = precision(**fixture)
    assert actual == expected


def test_recall():
    fixture = {"tp": 0, "fn": 0}
    expected = 0.0
    actual = recall(**fixture)
    assert actual == expected
