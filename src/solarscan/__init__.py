"""SolarScan — aerial PV fault detection pipeline.

Stages: detect panels -> classify faults (IEC taxonomy) -> georeference ->
estimate yield loss & severity -> generate inspection report. A parallel edge
path exports the models to TensorRT for real-time inference on NVIDIA Jetson.
"""

__version__ = "0.1.0"
