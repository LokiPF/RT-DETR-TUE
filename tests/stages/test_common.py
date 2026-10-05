def test_median_ms_warms_up_on_the_first_inputs_then_times_every_input():
    from degradation_monitor.stages.common import median_ms

    calls = []
    assert median_ms(calls.append, [1, 2, 3], 2, "cpu") >= 0.0
    assert calls == [1, 2, 1, 2, 3]
