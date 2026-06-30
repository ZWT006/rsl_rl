# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional training add-ons for runners."""

from .amp import AMPAddon

__all__ = ["AMPAddon"]
