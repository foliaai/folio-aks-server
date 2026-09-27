#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : __init__.py
@Author  : caixiongjiang
@Date    : 2026/01/21 10:00
@Function: 
    路由模块
    导出所有 API 路由：knowledge_router（Knowledge API）、memory_router（Memory API）
@Modify History:
    2026/02/18 - 导出 knowledge_router
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from api.routers.chat import chat_router
from api.routers.knowledge import knowledge_router
from api.routers.user import user_router

__all__ = ["knowledge_router", "chat_router", "user_router"]
