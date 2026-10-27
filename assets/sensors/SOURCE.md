# Sensor patterns

- `mid360.npy`: the Livox MID-360 scan pattern from Livox's
  [livox_laser_simulation](https://github.com/Livox-SDK/livox_laser_simulation) `scan_mode/mid360.csv`
  (800,000 rays), converted to float32 `(azimuth, elevation)` in radians, where elevation is
  90 deg minus the CSV's zenith angle. The file is byte-identical to `scan_mode/mid360.npy` in
  [OmniPerception](https://github.com/aCodeDog/OmniPerception)'s LidarSensor package, which ships
  this conversion. Verified on 2026-10-27 against livox_laser_simulation commit cf764140fe9a.

  Used by the `rolling_livox` sensor (`ortsim/perception/livox.py`) that feeds the
  `go2_perceptive_mid360_phase5` height map.

Copyright (c) 2021 Livox, MIT License (see `LICENSE`).
