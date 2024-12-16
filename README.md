# DKA-YOLO

This repository is the implementation for DKA-YOLO object detection model


![Graphical Abstract](./assets/G_abstract.jpg)

## Overview

- **YOLOv8 Backbone**: Utilizes the advanced YOLOv8 framework for base feature extraction.

- **LDKA-Conv**: Large-size Dilation Kernels Aggregation Convolution A module to extend the receptive field and enhance detailed feature extraction.

- **MDKA-Conv**: Multi-scale Dilation Kernels Aggregation Convolution： Enhances performance and efficiency through multi-scale feature extraction.

- **MSK-Detect Head**： Multi-Scale Kernel Detect: Improves feature diversity, generalization ability, and computational efficiency.

## Citation

```bibtex
@ARTICLE{10792910,
  author={Qiu, Yicheng and Sha, Feng and Niu, Li},
  journal={IEEE Access}, 
  title={DKA-YOLO: Enhanced Small Object Detection via Dilation Kernel Aggregation Convolution Modules}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2024.3515201}}
```

## Reference
- [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) 
- [github.com/AILab-CVC/UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet) 


## License 
This project is licensed under the Apache 2.0 License, please click [LICENSE](LICENSE) for more details.

