#!/usr/bin/env python3
"""Convert the audited Apache-2.0 Manga109-s RT-DETR ONNX model to Core ML."""

import argparse
import hashlib
import shutil
from pathlib import Path

import coremltools as ct
import onnx
import torch
from onnx2torch import convert
from onnx2torch.node_converters.matmul import OnnxMatMul
from onnx2torch.node_converters.registry import OperationDescription, add_converter
import onnx2torch.node_converters.registry as converter_registry
from onnx2torch.utils.common import (
    OnnxMapping,
    OperationConverterResult,
    get_const_value,
    onnx_mapping_from_node,
)
from torch import nn


SOURCE = "https://huggingface.co/tori29umai/rtdetrv4-x-manga109s"
SOURCE_SHA256 = "fba50583bfaaba3eed33f3eac6ca37be09b8c4882bac05da93f96697010a45b1"
INPUT_EDGE = 1280


class GridSample(nn.Module):
    def __init__(self, mode, padding_mode, align_corners):
        super().__init__()
        self.mode = mode.decode() if isinstance(mode, bytes) else mode
        self.padding_mode = padding_mode.decode() if isinstance(padding_mode, bytes) else padding_mode
        self.align_corners = bool(align_corners)

    def forward(self, value, grid):
        return torch.nn.functional.grid_sample(
            value,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


class StaticSplit(nn.Module):
    def __init__(self, sizes, axis):
        super().__init__()
        self.sizes = tuple(int(value) for value in sizes)
        self.axis = int(axis)

    def forward(self, value):
        return torch.split(value, self.sizes, dim=self.axis)


class IntegralMatMul(nn.Module):
    def forward(self, value, weights):
        return (value * weights).sum(dim=-1)


@add_converter(operation_type="GridSample", version=16)
def grid_sample_converter(node, _graph):
    return OperationConverterResult(
        GridSample(
            node.attributes.get("mode", "bilinear"),
            node.attributes.get("padding_mode", "zeros"),
            node.attributes.get("align_corners", 0),
        ),
        onnx_mapping_from_node(node),
    )


def static_split_converter(node, graph):
    sizes = get_const_value(node.input_values[1], graph).tolist()
    return OperationConverterResult(
        StaticSplit(sizes, node.attributes.get("axis", 0)),
        OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values),
    )


def matmul_converter(node, _graph):
    module = IntegralMatMul() if "/integral" in node.name else OnnxMatMul()
    return OperationConverterResult(module, onnx_mapping_from_node(node))


converter_registry._CONVERTER_REGISTRY[OperationDescription("", "Split", 13)] = static_split_converter
converter_registry._CONVERTER_REGISTRY[OperationDescription("", "MatMul", 13)] = matmul_converter


class RuntimeModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("target_size", torch.tensor([[INPUT_EDGE, INPUT_EDGE]], dtype=torch.int64))

    def forward(self, image):
        values = self.model(image, self.target_size)
        return values[0].float(), values[1].float(), values[2][0].float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if hashlib.sha256(args.onnx_model.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit("Unexpected ONNX SHA-256; refusing to convert an unaudited model")

    graph = onnx.load(args.onnx_model)
    for node in graph.graph.node:
        while node.input and node.input[-1] == "":
            node.input.pop()

    model = RuntimeModel(convert(graph).eval()).eval()
    example = torch.zeros((1, 3, INPUT_EDGE, INPUT_EDGE), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=False)

    if args.output.exists():
        shutil.rmtree(args.output)
    core_ml = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.ImageType(
            name="image",
            shape=example.shape,
            scale=1 / 255,
            color_layout=ct.colorlayout.RGB,
        )],
        outputs=[
            ct.TensorType(name="labels"),
            ct.TensorType(name="boxes"),
            ct.TensorType(name="scores"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    core_ml.author = "NicoComic"
    core_ml.short_description = "RT-DETRv4-X manga panel detector; Apache-2.0; Manga109-s."
    core_ml.version = "1.0.0"
    core_ml.user_defined_metadata["source"] = SOURCE
    core_ml.user_defined_metadata["source_sha256"] = SOURCE_SHA256
    core_ml.user_defined_metadata["license"] = "Apache-2.0"
    core_ml.save(str(args.output))


if __name__ == "__main__":
    main()
