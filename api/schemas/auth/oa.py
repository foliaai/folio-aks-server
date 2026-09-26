#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : oa.py
@Author  : caixiongjiang
@Date    : 2026/09/16
@Function: 
    OA 授权码登录相关请求 / 响应模型
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from typing import Optional

from pydantic import BaseModel, Field


class OALoginRequest(BaseModel):
    """OA 授权码登录请求"""

    code: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="OA 网页授权回调返回的 code（一次性）",
    )


class OAUserProfile(BaseModel):
    """OA 登录返回的用户信息（仅保留登录展示所需字段）"""

    user_id: str = Field(..., description="本系统用户标识（OA 工号 EmployeeNo）")
    employee_no: Optional[str] = Field(default=None, description="OA 工号")
    employee_id: Optional[int] = Field(default=None, description="OA 内部员工 ID")
    name: Optional[str] = Field(default=None, description="员工姓名")
    alias_name: Optional[str] = Field(default=None, description="员工别名")


class OALoginData(BaseModel):
    """OA 登录成功返回数据"""

    access_token: str = Field(..., description="本域 JWT，后续请求通过 Authorization: Bearer 携带")
    token_type: str = Field(default="bearer", description="凭证类型")
    expires_in: int = Field(..., description="JWT 有效期（秒）")
    user: OAUserProfile = Field(..., description="登录用户信息")