#!/usr/bin/env python3
"""Executable protocol tests for the 910A INT8 compatibility path."""

from __future__ import annotations

import contextlib
import os
import types
import unittest
from unittest import mock

import torch

import comfy.ops
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout
from comfy_kitchen.backends.eager.quantization import (
    dequantize_int8_convrot_weight,
)


def make_weight(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    *,
    convrot: bool = False,
    group_size: int = 256,
    transposed: bool = False,
) -> QuantizedTensor:
    params = TensorWiseINT8Layout.Params(
        scale=scale,
        orig_dtype=torch.float16,
        orig_shape=tuple(qdata.shape),
        is_weight=True,
        convrot=convrot,
        convrot_groupsize=group_size,
        transposed=transposed,
    )
    return QuantizedTensor(qdata, "TensorWiseINT8Layout", params)


def dequantized(weight: QuantizedTensor) -> torch.Tensor:
    qdata = weight._qdata
    scale = weight._params.scale
    if weight._params.convrot:
        return dequantize_int8_convrot_weight(
            qdata,
            scale,
            weight._params.convrot_groupsize,
        )
    return qdata.float() * scale


class CompatHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = mock.patch.dict(
            os.environ,
            {
                "H3_910A_QUANT_COMPAT": "1",
                "H3_910A_QUANT_CHUNK": "7",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.patch = mock.patch.object(
            comfy.ops,
            "_is_npu_int8_linear",
            return_value=True,
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)
        torch.manual_seed(20261109)

    def assert_linear_close(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        input: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        convrot: bool = False,
        group_size: int = 256,
        input_act: str | None = None,
        atol: float = 2e-3,
        rtol: float = 2e-3,
    ) -> torch.Tensor:
        weight = make_weight(
            qdata,
            scale,
            convrot=convrot,
            group_size=group_size,
        )
        actual = comfy.ops._npu_int8_linear_or_dequant(
            input,
            weight,
            bias,
            input_act=input_act,
        )
        activated = input if input_act is None else comfy.ops.INPUT_ACT_EAGER[input_act](input)
        expected = torch.nn.functional.linear(
            activated,
            dequantized(weight).to(input.dtype),
            bias,
        )
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return actual


class CompatProtocolTests(CompatHarness):
    def test_plain_int8_2d_bias_nonmultiple_chunk(self) -> None:
        self.assert_linear_close(
            torch.randint(-127, 128, (25, 8), dtype=torch.int8),
            torch.rand(25, 1, dtype=torch.float32) * 0.01,
            torch.randn(3, 8, dtype=torch.float16),
            bias=torch.randn(25, dtype=torch.float16),
        )

    def test_three_dimensional_swiglu(self) -> None:
        self.assert_linear_close(
            torch.randint(-127, 128, (19, 8), dtype=torch.int8),
            torch.rand(19, 1, dtype=torch.float32) * 0.01,
            torch.randn(2, 3, 16, dtype=torch.float16),
            input_act="swiglu",
        )

    def test_scalar_scale(self) -> None:
        self.assert_linear_close(
            torch.randint(-127, 128, (13, 8), dtype=torch.int8),
            torch.tensor(0.0078125, dtype=torch.float32),
            torch.randn(4, 8, dtype=torch.float16),
        )

    def test_empty_input(self) -> None:
        weight = make_weight(
            torch.randint(-127, 128, (11, 8), dtype=torch.int8),
            torch.rand(11, 1, dtype=torch.float32) * 0.01,
        )
        output = comfy.ops._npu_int8_linear_or_dequant(
            torch.empty((0, 8), dtype=torch.float16),
            weight,
        )
        self.assertEqual(output.shape, (0, 11))
        self.assertEqual(output.numel(), 0)

    def test_single_convrot_group(self) -> None:
        with mock.patch.dict(os.environ, {"H3_910A_QUANT_CHUNK": "2"}):
            self.assert_linear_close(
                torch.randint(-127, 128, (7, 256), dtype=torch.int8),
                torch.rand(7, 1, dtype=torch.float32) * 0.01,
                torch.randn(5, 256, dtype=torch.float16),
                convrot=True,
                group_size=256,
                atol=5e-3,
                rtol=3e-3,
            )

    def test_two_convrot_groups(self) -> None:
        self.assert_linear_close(
            torch.randint(-127, 128, (9, 512), dtype=torch.int8),
            torch.rand(9, 1, dtype=torch.float32) * 0.01,
            torch.randn(4, 512, dtype=torch.float16),
            convrot=True,
            group_size=256,
            atol=5e-3,
            rtol=3e-3,
        )

    def test_native_quant_matmul_is_never_called(self) -> None:
        with mock.patch.object(
            comfy.ops,
            "_npu_int8_linear",
            side_effect=AssertionError("native NPU INT8 matmul must not run"),
        ):
            output = comfy.ops._npu_int8_linear_or_dequant(
                torch.ones((2, 8), dtype=torch.float16),
                make_weight(
                    torch.ones((7, 8), dtype=torch.int8),
                    torch.ones((7, 1), dtype=torch.float32),
                ),
            )
        self.assertEqual(output.shape, (2, 7))

    def test_transposed_uses_one_full_dequant(self) -> None:
        weight = make_weight(
            torch.ones((8, 5), dtype=torch.int8),
            torch.ones((5, 1), dtype=torch.float32),
            transposed=True,
        )
        with mock.patch.object(
            weight,
            "dequantize",
            wraps=weight.dequantize,
        ) as dequantize:
            output = comfy.ops._npu_int8_linear_or_dequant(
                torch.ones((2, 5), dtype=torch.float16),
                weight,
            )
        self.assertEqual(output.shape, (2, 8))
        dequantize.assert_called_once()

    def test_non_float16_inputs_are_rejected_with_context(self) -> None:
        weight = make_weight(
            torch.ones((7, 8), dtype=torch.int8),
            torch.ones((7, 1), dtype=torch.float32),
        )
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(
                    TypeError,
                    r"weight_shape=\(7, 8\).*scale_count=7.*input_dtype=",
                ):
                    comfy.ops._npu_int8_linear_or_dequant(
                        torch.ones((2, 8), dtype=dtype),
                        weight,
                    )
                with self.assertRaisesRegex(
                    TypeError,
                    r"weight_shape=\(7, 8\).*scale_count=7.*convrot_group_size=256",
                ):
                    comfy.ops._npu_int8_linear_compat(
                        torch.ones((2, 8), dtype=dtype),
                        weight,
                    )

    def test_scale_is_moved_once_per_linear(self) -> None:
        qdata = torch.randint(-127, 128, (19, 8), dtype=torch.int8)
        scale = torch.rand(19, 1, dtype=torch.float32)
        weight = make_weight(qdata, scale)
        real_to = torch.Tensor.to
        scale_moves = 0

        def count_scale_moves(tensor: torch.Tensor, *args, **kwargs):
            nonlocal scale_moves
            if tensor.data_ptr() == scale.data_ptr():
                scale_moves += 1
            return real_to(tensor, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "to", new=count_scale_moves):
            comfy.ops._npu_int8_linear_or_dequant(
                torch.randn(3, 8, dtype=torch.float16),
                weight,
            )
        self.assertEqual(scale_moves, 1)

    def test_three_entrypoints_select_compat(self) -> None:
        expected = torch.randn(2, 7)
        weight = make_weight(
            torch.ones((7, 8), dtype=torch.int8),
            torch.ones((7, 1), dtype=torch.float32),
        )
        x = torch.randn(2, 8, dtype=torch.float16)

        with mock.patch.object(
            comfy.ops,
            "_npu_int8_linear_or_dequant",
            return_value=expected,
        ) as dispatch:
            self.assertIs(
                comfy.ops.mixed_precision_ops().Linear._forward(
                    object(),
                    x,
                    weight,
                    None,
                ),
                expected,
            )
            dispatch.assert_called_once()

        with mock.patch.object(
            comfy.ops,
            "_npu_int8_linear_or_dequant",
            return_value=expected,
        ) as dispatch:
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        comfy.ops,
                        "cast_bias_weight",
                        return_value=(weight, None, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(comfy.ops, "uncast_bias_weight")
                )
                actual = comfy.ops.linear_input_act(
                    types.SimpleNamespace(weight=weight),
                    x,
                    "gelu_tanh",
                )
            self.assertIs(actual, expected)
            dispatch.assert_called_once()

        with mock.patch.object(
            comfy.ops,
            "_npu_int8_linear_compat",
            wraps=comfy.ops._npu_int8_linear_compat,
        ) as compat:
            comfy.ops._npu_int8_linear_or_dequant(x, weight)
            compat.assert_called_once()

    def test_tp_path_skips_scale_rendezvous(self) -> None:
        linear = mock.Mock()
        linear.return_value = torch.ones(1, 16)
        tp_context = mock.Mock()
        actual = comfy.ops.linear_tp_row(
            linear,
            torch.ones(1, 16),
            tp_context,
            0,
        )
        self.assertTrue(torch.equal(actual, torch.ones(1, 16)))
        tp_context.all_reduce_max.assert_not_called()

    def test_capability_is_disabled_in_compat_mode(self) -> None:
        self.assertFalse(
            comfy.ops._can_use_npu_int8_linear(
                torch.ones((1, 16), dtype=torch.float16),
                mock.Mock(_params=types.SimpleNamespace(transposed=False)),
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
