# AdverseGen: A Practical Tool for Generating Adversarial Examples to Deep Neural Networks Using Black-box Approaches

AdverseGen is a **Python toolbox** for adversarial robustness research. It can be invoked either by the graphical user interface (GUI) or command lines to generate adversarial examples in the black-box setting.

Currently, AdverseGen supports 2 frameworks: PyTorch and TensorFlow. In the future, we will prioritize implementing attacks in PyTorch, but we also welcome contributions in all 2 frameworks. 

## Setting up AdverseGen

### Dependencies

AdverseGen uses PyTorch 1.7.1 or TensorFlow 2.3.0 & 2.5.0  to accelerate graph computations conducted by machine learning models. 

Also, the GUI of this tool is developed under PySimpleGUI 4.38.0 and pillow 7.2.0 libraries. If you want to start AdverseGen in GUI mode, run the following commands:

```shell
pip install pillow PySimpleGUI
```
## Citation

If you use this reository in your work, please use this bibtex

```
@inprocedings{zhang2021adversegen,
  title={AdverseGen: A Practical Tool for Generating Adversarial Examples to Deep Neural       Networks Using Black-box Approaches},
  author={Zhang Keyuan and Wu Kaiyue and Chen Siyu},
  booktitle={AI-2021: The Forty-first SGAI International Conference},
  pages={accepted},
  year={2021},
  organization={SGAI}
}
```

