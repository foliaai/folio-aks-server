#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : __init__.py
@Author  : caixiongjiang
@Date    : 2026/09/16
@Function: 
    Auth API 路由模块
    包含端点：
      /api/auth/oa/login - OA 授权码登录（内部部署）
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from fastapi import APIRouter
from api.routers.auth.oa import router as oa_router

auth_router = APIRouter(prefix="/api/auth")
auth_router.include_router(oa_router)

__all__ = ["auth_router"]