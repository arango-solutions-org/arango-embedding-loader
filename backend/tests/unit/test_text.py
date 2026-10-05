from app.ingest.text import get_path, render_template, set_path, value_to_text


def test_get_and_set_path():
    doc = {"a": {"b": {"c": 1}}, "l": [{"x": 1}, {"x": 2}]}
    assert get_path(doc, "a.b.c") == 1
    assert get_path(doc, "l.1.x") == 2
    assert get_path(doc, "a.missing", "d") == "d"
    set_path(doc, "v.w", [1.0])
    assert doc["v"] == {"w": [1.0]}


def test_value_to_text():
    assert value_to_text(None) is None
    assert value_to_text("  ") is None
    assert value_to_text(["a", "", "b"]) == "a\nb"
    assert value_to_text(3) == "3"
    assert value_to_text({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'


def test_render_template():
    doc = {"title": "T", "meta": {"kw": ["x", "y"]}}
    assert render_template("{title}: {meta.kw}", doc) == "T: x\ny"
    assert render_template("{nope} {other}", doc) is None
