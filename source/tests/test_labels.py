import numpy as np
from PIL import Image

from ma_psdun.labels import audit_label, foreground_bbox, preprocess_label


def test_label_preprocess_is_square_and_audited(tmp_path):
    image = np.zeros((8, 12), dtype=np.uint8)
    image[2:6, 4:9] = 200
    path = tmp_path / "truth.png"
    Image.fromarray(image).save(path)
    output, audit = preprocess_label(path, 4, crop="center", contrast="percentile")
    assert output.shape == (4, 4)
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0
    assert audit["original_size"] == [12, 8]
    assert audit["cropped_size"] == [8, 8]
    assert audit["crop_mode"] == "center"
    assert "principal_axis_angle_deg" in audit
    assert foreground_bbox(image) == (4, 2, 9, 6)


def test_audit_reports_off_center_foreground():
    image = np.zeros((10, 10), dtype=np.float32)
    image[1:3, 7:9] = 1.0
    audit = audit_label(image, threshold=0.5)
    assert audit["touches_border"] is False
    assert audit["center_offset_normalized"][0] > 0
