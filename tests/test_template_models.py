import pytest
from pydantic import ValidationError
from template_models import (
    PlanViewportConfig,
    ProfileViewportConfig,
    DataBandItem,
    SidebarConfig,
    SidebarNotes,
    SidebarRect,
    SidebarTitleBlock,
    LSectionSheetTemplate,
)


def test_plan_viewport_validation():
    # Valid with snake_case and camelCase aliases
    vp = PlanViewportConfig(
        x=25.0,
        y=20.0,
        w=964.0,
        h=360.0,
        scale=5000,
        tickIntervalM=100,
        showNorthArrow=True,
    )
    assert vp.scale == 5000.0
    assert vp.tick_interval_m == 100.0
    assert vp.show_north_arrow is True

    # Invalid: non-positive dimensions
    with pytest.raises(ValidationError):
        PlanViewportConfig(w=-10.0)

    with pytest.raises(ValidationError):
        PlanViewportConfig(scale=0.0)


def test_profile_viewport_datum_calculation():
    pv = ProfileViewportConfig(
        x=25.0,
        y=390.0,
        w=964.0,
        h=220.0,
        hScale=5000,
        vScale=500,
        datumRoundStepM=5.0,
        datumClearanceM=20.0,
    )
    assert pv.h_scale == 5000.0
    assert pv.v_scale == 500.0
    assert pv.vertical_exaggeration == 10.0

    # Test datum elevation formula: floor((lowest_egl - 20) / 5) * 5
    # For lowest EGL = 143.5 -> 143.5 - 20 = 123.5 -> floor(123.5 / 5) * 5 = 24 * 5 = 120.0
    lowest_egl = 143.5
    step = pv.datum_round_step_m
    clearance = pv.datum_clearance_m
    import math
    datum = math.floor((lowest_egl - clearance) / step) * step
    assert datum == 120.0


def test_data_band_item():
    band = DataBandItem(
        key="proposed_rail_level",
        label="PROPOSED RAIL LEVEL (PRL)",
        heightMm=18.0,
        enabled=True,
    )
    assert band.key == "proposed_rail_level"
    assert band.height_mm == 18.0
    assert band.enabled is True

    with pytest.raises(ValidationError):
        DataBandItem(key="test", label="Test", heightMm=-5.0)


def test_lsection_sheet_template_full():
    payload = {
        "id": "rdso-a0-greenfield-standard",
        "name": "RDSO A0 Greenfield Standard",
        "corridorType": "greenfield",
        "paper": "A0",
        "sheetWidthMm": 1189.0,
        "sheetHeightMm": 841.0,
        "sidebarWidthMm": 185.0,
        "planViewport": {
            "x": 25.0,
            "y": 20.0,
            "w": 964.0,
            "h": 360.0,
            "scale": 5000,
            "tickIntervalM": 100,
            "showNorthArrow": True,
        },
        "profileViewport": {
            "x": 25.0,
            "y": 390.0,
            "w": 964.0,
            "h": 220.0,
            "hScale": 5000,
            "vScale": 500,
            "datumRoundStepM": 5.0,
            "datumClearanceM": 20.0,
        },
        "dataBands": [
            {"key": "prl", "label": "PRL", "heightMm": 18.0, "enabled": True},
            {"key": "egl", "label": "EGL", "heightMm": 18.0, "enabled": True},
            {"key": "gradient", "label": "GRADIENT", "heightMm": 24.0, "enabled": True},
            {"key": "disabled_band", "label": "EXTRA", "heightMm": 18.0, "enabled": False},
        ],
        "sidebar": {
            "notes": {"clauses": ["1. ALL DIMENSIONS IN MM.", "2. TRACK BUILDUP = 0.762m."]},
            "titleBlock": {
                "zonalRailway": "SOUTH CENTRAL RAILWAY",
                "division": "SECUNDERABAD",
            },
        },
    }

    template = LSectionSheetTemplate(**payload)
    assert template.id == "rdso-a0-greenfield-standard"
    assert template.sheet_width_mm == 1189.0
    assert template.sheet_height_mm == 841.0
    assert template.sidebar_width_mm == 185.0

    # Check bands helper
    enabled_bands = template.get_enabled_bands()
    assert len(enabled_bands) == 3
    assert template.calculate_total_bands_height_mm() == 18.0 + 18.0 + 24.0  # 60.0 mm

    # Check datum calculation
    assert template.calculate_datum_elevation(lowest_egl=143.5) == 120.0

    # Check ground length calculation: 964 mm * (5000 / 1000) = 4820 meters
    assert template.get_available_drawing_length_m() == 4820.0


def test_boundary_overflow_rejection():
    # Viewport wider than sheet
    with pytest.raises(ValidationError) as exc_info:
        LSectionSheetTemplate(
            id="invalid-template",
            name="Invalid",
            sheetWidthMm=1000.0,
            planViewport={"x": 50.0, "w": 980.0},  # 50 + 980 = 1030 > 1000
        )
    assert "exceeds sheet width" in str(exc_info.value)

