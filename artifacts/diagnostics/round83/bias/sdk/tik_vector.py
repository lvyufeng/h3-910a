#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
FILE:     tik_vector.py
DESC:     vector
CREATED:  2020-03-10 18:53:42
MODIFIED: 2020-12-7 19:17:00
"""
from tbe import tvm
from tbe.tvm import const
from tbe.tvm import schedule
from tbe.tvm import string_types
from tbe.tvm.tir import StringImm
from tbe.tvm.tir import Allocate
from tbe.tvm.tir.stmt import AttrStmt
from tbe.common.platform import scope_preg
from tbe.common.platform import scope_vreg
from tbe.common.platform import scope_areg
from tbe.common.platform import scope_ureg
from tbe.common.platform import scope_wreg
from tbe.tik.tik_lib.tik_util import need_check_out_of_scope
from tbe.tik.tik_lib.tik_check_util import TikCheckUtil
from tbe.tik.tik_lib.tik_expr import BasicExpr
from tbe.tik.api.tik_tensor import Tensor
from tbe.tik.common.util import DTYPE_SIZE
from tbe.tik.tik_lib.tik_source_info import source_info_decorator
from tbe.tik.tik_lib.tik_params import ONE_BYTE_BIT_LEN
from tbe.tik.tik_lib.tik_buffervar import TikBufferVar
from tbe.tik.tik_lib.tik_backend import tik_access_ptr
from tbe.tik.tik_lib.tik_basic_data import BasicData
from tbe.tik import debug

_CONST_ELEM_OFFSET_INDEX = 2
_CONST_EXTENT_INDEX = 3
_CONST_READ_WRITE_MODEL_INDEX = 4


@debug.vector_register
class Vector(BasicExpr, BasicData):
    """
    hint:vector expression
    """
    VECTOR_COUNT = 0
    PG_COUNT = 0
    ADDRESS_COUNT = 0
    ALIGNMENT_COUNT = 0
    WIDE_COUNT = 0

    def __init__(self, ir_generator, dtype="float16", name="vector_buf_", scope=scope_vreg, if_global_scope=False):
        """
        vector register initialization
        Parameters
        ----------
        ir_generator: Halide IR generator
        dtype:vector register data type
        name:vector register name
        scope: vreg, preg, or areg
        if_global_scope:global_scope

        Returns
        ----------
        None
        """
        # check dtype is str
        TikCheckUtil.check_type_match(dtype, str, "dtype should be str")
        # check name is valid
        TikCheckUtil.check_name_str_valid(name)

        self.instance_func = None
        self.ir_generator = ir_generator
        self._if_global_scope = if_global_scope
        BasicExpr.__init__(self)
        if scope == scope_vreg:
            BasicData.__init__(self, "Vector")
            Vector.VECTOR_COUNT += 1
            count = Vector.VECTOR_COUNT
            self.data_len = self.ir_generator.code_buffer_manager.get_reg_len(scope_vreg, dtype)
        elif scope == scope_preg:
            BasicData.__init__(self, "Predicate")
            Vector.PG_COUNT += 1
            count = Vector.PG_COUNT
            if name == "vector_buf_":
                name = "preg_buf_"
            # only support bool Predicate
            self.data_len = self.ir_generator.code_buffer_manager.get_reg_len(scope_preg, dtype)
        elif scope == scope_areg:
            BasicData.__init__(self, "Address")
            Vector.ADDRESS_COUNT += 1
            count = Vector.ADDRESS_COUNT
            if name == "vector_buf_":
                name = "areg_buf_"
            # ignore input dtype, only support int32 addr
            dtype = "int32"
            self.data_len = self.ir_generator.code_buffer_manager.get_reg_len(scope_areg, dtype)
        elif scope == scope_ureg:
            BasicData.__init__(self, "Alignment")
            Vector.ALIGNMENT_COUNT += 1
            count = Vector.ALIGNMENT_COUNT
            if name == "vector_buf_":
                name = "ureg_buf_"
            self.data_len = self.ir_generator.code_buffer_manager.get_reg_len(scope_ureg, dtype)
        elif scope == scope_wreg:
            BasicData.__init__(self, "Wide")
            Vector.WIDE_COUNT += 1
            count = Vector.WIDE_COUNT
            if name == "vector_buf_":
                name = "wreg_buf_"
            self.data_len = self.ir_generator.code_buffer_manager.get_reg_len(scope_wreg, dtype)
        else:
            raise RuntimeError('Unexpected scope: %s' % scope)

        # _available:The vector state variable. when vector variable
        # "_available" is True, this vector can be accessed.when vector variable
        # "_available" is False, this vector can't be accessed and assert
        # "This vector is not defined in this scope."
        self.available = True
        if if_global_scope:
            self._name = "global_" + name + str(count)
            buffer_var = tvm.var("global_" + self._name, dtype)
            self.reg_buffer = TikBufferVar(self.ir_generator, buffer_var, None, dtype)
            self.instance_func = \
                lambda x: AttrStmt(buffer_var, "storage_scope", StringImm(scope),
                                   Allocate(buffer_var, dtype, self.data_len, const(1, dtype="uint1"), x))
        else:
            self._name = name + str(count)
            self.reg_buffer = ir_generator.allocate((dtype, self.data_len, self._name), scope=scope)
            self.buffer = tvm.decl_buffer(self.data_len, dtype, name=name, scope=scope, data=self.reg_buffer)

    def __repr__(self):
        """
        repr value
        Parameters
        ----------

        Returns
        -------
        repr value
        """
        return "(%s)%s %s[%s]" % (self.scope, self.dtype, self.name, self.length())

    @property
    @source_info_decorator()
    def name(self):
        """
        get register name
        Parameters
        ----------

        Returns
        ----------
        register name
        """
        self._check_vector_scope()
        return self._name

    @property
    def scope(self):
        """
        return vector memory scope
        Parameters
        ----------

        Returns
        ----------
        return: buffer scope
        """
        reg_value = None
        if self.is_predicate():
            reg_value = scope_preg
        elif self.is_vector():
            reg_value = scope_vreg
        elif self.is_address():
            reg_value = scope_areg
        elif self.is_alignment():
            reg_value = scope_ureg
        elif self.is_wide():
            reg_value = scope_wreg
        return reg_value

    @property
    def if_global_scope(self):
        """
        judging whether is global scope
        Parameters
        ----------

        Returns
        ----------
        judging whether is global scope
        """
        self._check_vector_scope()
        return self._if_global_scope

    @staticmethod
    def eval_value():
        """
        return expression value
        Parameters
        ----------

        Returns
        ----------
        return:no return
        """

    def get(self):
        """
        get buffer
        Parameters
        ----------

        Returns
        ----------
        top register buffer
        """
        self._check_vector_scope()
        return self.reg_buffer

    def merge_vector(self, body):
        """
        merge vector
        Parameters
        ----------

        Returns
        ----------
        vector
        """
        self._check_vector_scope()
        if self.instance_func is not None:
            return self.instance_func(body)
        return body

    def access_ptr(self, access_mask, offset=0, cast_dtype="handle", extent=None):
        """
        get an access pointer to the head of buffer
        Parameters
        ----------
        access_mask: mask
        offset: buffer offset
        cast_dtype: cast data type
        extent: extent args

        Returns
        -------
        buffer
        """
        ptr_type = "handle"
        if cast_dtype == "handle":
            cast_dtype = self.dtype
        if isinstance(access_mask, string_types):
            mask = 0
            for value in access_mask:
                if value == "r":
                    mask = mask | self.buffer.READ
                elif value == "w":
                    mask = mask | self.buffer.WRITE
                else:
                    raise ValueError("Unknown access_mask %s" % access_mask)
            access_mask = mask

        offset_temp = schedule.convert(offset)
        m_args = self._get_access_ptr_args(offset_temp, access_mask, (ptr_type, cast_dtype), extent)
        return tvm.call_intrin(cast_dtype, 'tir.tvm_access_ptr', *m_args)

    def disable_vector(self):
        """
        when this vector lifecycle is in the end,
        this vector condition parameter should be changed from true to false.
        Parameters
        ----------
        No parameter

        Returns
        ----------
        None
        """
        self.available = False

    @source_info_decorator()
    def set_as(self, value):
        """
        set vector register value
        Parameters
        ----------
        value:set value

        Returns
        ----------
        None
        """
        self._check_vector_scope()
        if isinstance(value, Tensor):
            if not value.is_single_point():
                raise RuntimeError("value is not a single point")
            self.ir_generator.assign(self, value)
        else:
            if isinstance(value, BasicExpr):
                val = value.astype(self.dtype).get()
            elif isinstance(value, (float, int)):
                val = tvm.const(value, self.dtype)
            else:
                val = value
            self.reg_buffer[0] = val

    def length(self):
        """
        get current vector object length
        Parameters
        ----------

        Returns
        -------
        length
        """
        return self.data_len

    def _get_access_ptr_args(self, offset_temp, access_mask, type_list, extent):
        """
        get access ptr args
        Parameters
        ----------
        offset_temp: offset
        access_mask: mask
        type_list: ptr type and cast_dtype

        extent: extent args

        Returns
        -------
        access ptr args
        """
        ptr_type, cast_dtype = type_list
        self_dtype_bits = DTYPE_SIZE[self.dtype]*ONE_BYTE_BIT_LEN
        buffer_dtype_bits = DTYPE_SIZE[self.buffer.dtype]*ONE_BYTE_BIT_LEN
        offset_buffer_dtype = offset_temp * (self_dtype_bits // buffer_dtype_bits)
        cast_dtype_bits = DTYPE_SIZE[cast_dtype]*ONE_BYTE_BIT_LEN

        return tik_access_ptr((self.buffer, access_mask, buffer_dtype_bits, extent, cast_dtype_bits, cast_dtype),
                              ptr_type=ptr_type, offset=offset_buffer_dtype)

    def _check_vector_scope(self):
        msg = "vector %s is not defined in this scope." % self._name
        if need_check_out_of_scope(self.ir_generator):
            TikCheckUtil.check_equality(self.available, True, msg)
