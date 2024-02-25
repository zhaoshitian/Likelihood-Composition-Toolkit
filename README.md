# Likelihood-Composition-Toolkit
Toolkit for manipulating MLLM's likelihood on differnet generated contents.

Features:

- Support LLaVA serires model, Qwen-VL, SPHINX, Yi-VL, InternVL, Internlm-Xcomposer and CogAgent.
- Support verious composition methods.

## Getting Started

In this toolkit, you can run different models, e.g., LLaVA and InternVL. For these models, different environment is needed, and before running these models using this toolkit, you need to install the prerequisted environment. 

### Installation

|  | Github |
|-------|-------|
| LLaVA | https://github.com/haotian-liu/LLaVA |
| Yi-VL | https://github.com/01-ai/Yi |
| Qwen-VL | https://github.com/QwenLM/Qwen-VL |
| Internlm-Xcomposer | https://github.com/InternLM/InternLM-XComposer |
| InternVL | https://github.com/OpenGVLab/InternVL |
| CogAgent | https://github.com/THUDM/CogVLM |
| SPHINX | https://github.com/Alpha-VLLM/LLaMA2-Accessory |

### Get likelihood of different models

```sh
cd LiCo
bash run_inference.sh
```

> [!NOTE]
> You can also download the likelihood of different models from these links.

|  | MME | MMVP |
|-------|-------|-------|
| LLaVA |  |  |
| Yi-VL |  |  |
| Qwen-VL |  |  |
| Internlm-Xcomposer |  |  |
| InternVL |  |  |
| CogAgent |  |  |
| SPHINX |  |  |

### Applying likelihood composition

```sh
bash composition.sh
```

## Composition Methods

A quick overview of the currently supported composition methods:

| Method                                                                                       | `composition_method` value | MLLM | LLM |
| -------------------------------------------------------------------------------------------- | ---------------------------| ----- | ----- |
| Ensemble                                   | `ensemble`             | ✅          |❌          |
| Debias                                                                                        | `debias`              | ✅          |❌          |
| Contrast                                     | `contrast`    | ✅          |❌          |
| Proxy Tuning                                                    | `proxy`               | ✅          |❌          |
| Causal-CoG         | `causal_cog`          | ✅          |❌          |

### Ensemble

### Debias

### Contrast

### Proxy Tuning

### Causal-CoG

## TODO

- [ ] Support SPHINX series model.
- [ ] Add result table.
- [ ] Upload the likelihood file onto huggingface.

## Authors

- **Your Name** - *Initial work* - [Your GitHub Username](https://github.com/yourusername)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Any contributors you want to thank
- Inspiration sources
- etc.
