#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""临时脚本：把 dev 库中 html 文件落库的元素信息（MySQL 元信息 + MongoDB 内容）
合并还原成 JSON，供人工检视解析质量。用完可删。"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from src.db.mysql.connection import get_mysql_manager
from src.db.mongodb.mongodb_manager import get_mongodb_manager

OUT = Path(__file__).parent.parent / "tmp_results" / "html_elements_dump.json"


async def main() -> None:
    pick = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 第几个 html 文件（0=最新）
    mongo = await get_mongodb_manager()
    db = await mongo.get_database()
    mysql = get_mysql_manager()

    with mysql.get_session() as s:
        files = s.execute(text(
            "SELECT file_id, file_name, document_id, file_sha256, create_time "
            "FROM workspace_file_system "
            "WHERE deleted = 0 AND (LOWER(file_name) LIKE '%.html' OR LOWER(file_name) LIKE '%.htm') "
            "ORDER BY create_time DESC LIMIT 5"
        )).mappings().all()

        if not files:
            print("dev 库中没有 html/htm 文件记录")
            return

        print(f"找到 {len(files)} 个 html 文件（最新在前）:")
        for i, f in enumerate(files):
            print(f"  [{i}] {f['file_name']}  doc={f['document_id']}  {f['create_time']}")
        f = files[pick]

        els = s.execute(text(
            "SELECT element_id, element_index, page_index, element_type, page_position, "
            "text_level, bucket_name, image_file_path, image_file_name, image_file_type "
            "FROM element_meta_info "
            "WHERE document_id = :d AND deleted = 0 "
            "ORDER BY page_index, element_index"
        ), {"d": f["document_id"]}).mappings().all()

    element_ids = [e["element_id"] for e in els]
    contents = {}
    if element_ids:
        async for doc in db["element_data"].find({"_id": {"$in": element_ids}}):
            contents[doc["_id"]] = doc.get("content", {})

    merged = []
    for e in els:
        item = dict(e)
        item["create_time"] = str(item.get("create_time"))
        item["content"] = contents.get(item["element_id"])
        merged.append(item)

    payload = {
        "file": {
            "file_id": f["file_id"],
            "file_name": f["file_name"],
            "document_id": f["document_id"],
            "file_sha256": f["file_sha256"],
            "create_time": str(f["create_time"]),
        },
        "total_elements": len(merged),
        "type_stats": {},
        "elements": merged,
    }
    for e in merged:
        payload["type_stats"][e["element_type"]] = payload["type_stats"].get(e["element_type"], 0) + 1

    out_path = OUT.with_name(f"html_elements_{pick}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=lambda o: o.decode("utf-8", "replace") if isinstance(o, bytes) else str(o),
        ),
        encoding="utf-8",
    )
    print(f"\n导出完成: {out_path}")
    print(f"元素总数: {len(merged)}, 类型分布: {payload['type_stats']}")


if __name__ == "__main__":
    asyncio.run(main())
