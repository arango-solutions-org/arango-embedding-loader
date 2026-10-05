import gzip
import io

import pytest

from app.ingest.fieldpaths import detect_field_paths
from app.ingest.parsers import CsvOptions, ParseError, detect_format, iter_documents


def docs(data: bytes, filename: str | None = None, **kw):
    return list(iter_documents(io.BytesIO(data), filename=filename, **kw))


def test_jsonl_customer_format():
    data = b'{"id":1234,"text":"title abstract keywords"}\n\n{"id":5,"text":"b"}\n'
    assert docs(data, "part.jsonl") == [
        {"id": 1234, "text": "title abstract keywords"},
        {"id": 5, "text": "b"},
    ]


def test_json_array_and_object_and_path():
    assert docs(b'[{"a":1},{"a":2.5}]') == [{"a": 1}, {"a": 2.5}]
    assert docs(b'{"a": {"b": 1}}', "one.json") == [{"a": {"b": 1}}]
    assert docs(b'{"data":[{"x":1}]}', json_path="data.item") == [{"x": 1}]


def test_autodetect():
    assert detect_format(None, b'[{"a":1}]') == "json"
    assert detect_format(None, b'{"a":1}\n{"a":2}') == "jsonl"
    assert detect_format(None, b"a,b\n1,2") == "csv"
    assert detect_format("x.ndjson.gz", b"") == "jsonl"


def test_gzip():
    data = gzip.compress(b'{"id":1,"text":"a"}\n')
    assert docs(data, "f.jsonl.gz") == [{"id": 1, "text": "a"}]
    assert docs(data) == [{"id": 1, "text": "a"}]


def test_csv_inference():
    data = b'name,n,f,b,e,z\nx,"1",2.5,true,,007\n'
    assert docs(data, "t.csv") == [
        {"name": "x", "n": 1, "f": 2.5, "b": True, "e": None, "z": "007"}
    ]
    raw = docs(data, "t.csv", csv_options=CsvOptions(infer_types=False))
    assert raw[0]["n"] == "1"


def test_invalid_jsonl():
    with pytest.raises(ParseError, match="line 2"):
        docs(b'{"a":1}\nnot json\n', "x.jsonl")


def test_field_detection():
    fields = detect_field_paths(
        [{"id": 1, "text": "a long abstract about genes and proteins", "m": {"k": "v"}}]
    )
    by_path = {f.path: f for f in fields}
    assert by_path["text"].embeddable
    assert not by_path["id"].embeddable
    assert "m.k" in by_path
