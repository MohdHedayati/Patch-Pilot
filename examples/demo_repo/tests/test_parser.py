from src.parser import parse_input


def test_parser():
    assert parse_input('{"a": 1}') == {"a": 1}
