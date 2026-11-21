from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

from ingestion.wms import HIMAWARI_LAYER, build_wms_url, format_gibs_timestamp


def test_format_gibs_timestamp_floors_to_10_minutes():
    dt = datetime(2026, 11, 21, 14, 37, 45, tzinfo=timezone.utc)
    assert format_gibs_timestamp(dt) == "2026-11-21T14:30:00Z"


def test_build_wms_url_default_time():
    url = build_wms_url()
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert "gibs.earthdata.nasa.gov" in parsed.netloc
    assert qs["SERVICE"] == ["WMS"]
    assert qs["VERSION"] == ["1.1.1"]
    assert qs["REQUEST"] == ["GetMap"]
    assert qs["LAYERS"] == [HIMAWARI_LAYER]
    assert qs["SRS"] == ["EPSG:4326"]
    assert qs["BBOX"] == ["75.0,3.0,86.0,12.0"]
    assert qs["WIDTH"] == ["640"]
    assert qs["HEIGHT"] == ["640"]
    assert qs["FORMAT"] == ["image/png"]
    assert qs["TIME"] == ["DEFAULT"]


def test_build_wms_url_with_datetime_and_custom_bbox():
    dt = datetime(2026, 11, 21, 8, 24, 10, tzinfo=timezone.utc)
    url = build_wms_url(timestamp=dt, bbox=(76.0, 4.0, 85.0, 11.0), width=640, height=640)
    qs = parse_qs(urlparse(url).query)
    assert qs["TIME"] == ["2026-11-21T08:20:00Z"]
    assert qs["BBOX"] == ["76.0,4.0,85.0,11.0"]


def test_build_wms_url_with_iso_string():
    url = build_wms_url(timestamp="2026-11-21T16:30:00Z")
    qs = parse_qs(urlparse(url).query)
    assert qs["TIME"] == ["2026-11-21T16:30:00Z"]
