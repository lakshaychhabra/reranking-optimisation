from scripts.benchmark_vllm_serving import parse_top3, percentile


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 3.8499999999999996
    assert percentile([], 0.5) is None


def test_parse_top3_checks_json_schema_uniqueness_and_ids() -> None:
    valid = {"1", "2", "3", "4"}
    text = '{"top_3":[{"rank":1,"tariff_id":"1"},{"rank":2,"tariff_id":"2"},{"rank":3,"tariff_id":"3"}]}'
    assert parse_top3(text, valid) == (True, True, ["1", "2", "3"])
    assert parse_top3('{"top_3":[]}', valid) == (True, False, [])
    assert parse_top3("not json", valid) == (False, False, [])
