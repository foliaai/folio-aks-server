#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : team_repo.py
@Author  : caixiongjiang
@Date    : 2026/10/18
@Function:
    Team Repository（团队成员关系见 team_member_repo）
@Modify History:

@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""

from src.db.mysql.models.business.team import Team
from src.db.mysql.repositories.base_repository import BaseRepository


class TeamRepository(BaseRepository[Team]):
    """Team Repository"""

    def __init__(self) -> None:
        super().__init__(Team)


# 全局实例
team_repo = TeamRepository()
