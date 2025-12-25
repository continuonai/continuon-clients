import pytest
from continuonbrain.tools.calculator import CalculatorTool

@pytest.mark.asyncio
async def test_calculator_basic():
    tool = CalculatorTool()
    res = await tool.execute(expression="2 + 2")
    assert res["result"] == 4

@pytest.mark.asyncio
async def test_calculator_math():
    tool = CalculatorTool()
    res = await tool.execute(expression="sqrt(16)")
    assert res["result"] == 4.0

@pytest.mark.asyncio
async def test_calculator_safety():
    tool = CalculatorTool()
    # Attempt unsafe operation (import)
    res = await tool.execute(expression="__import__('os').system('ls')")
    assert "error" in res

from continuonbrain.tools.filesystem import FileSystemTool
import os

@pytest.mark.asyncio
async def test_fs_list(tmp_path):
    # Setup temp dir
    (tmp_path / "test.txt").write_text("hello")
    (tmp_path / "subdir").mkdir()
    
    tool = FileSystemTool(whitelist_root=str(tmp_path))
    res = await tool.execute(action="list", path=".")
    
    names = [i["name"] for i in res["items"]]
    assert "test.txt" in names
    assert "subdir" in names

@pytest.mark.asyncio
async def test_fs_read(tmp_path):
    (tmp_path / "read_me.txt").write_text("content")
    
    tool = FileSystemTool(whitelist_root=str(tmp_path))
    res = await tool.execute(action="read", path="read_me.txt")
    assert res["content"] == "content"

@pytest.mark.asyncio
async def test_fs_safety(tmp_path):
    tool = FileSystemTool(whitelist_root=str(tmp_path))
    # Attempt traversal
    res = await tool.execute(action="list", path="../")
    assert "error" in res

from continuonbrain.tools.wikipedia import WikipediaTool

@pytest.mark.asyncio
async def test_wikipedia_search():
    tool = WikipediaTool()
    # This test requires internet. If it fails, we assume offline.
    res = await tool.execute(query="Python (programming language)")
    if "error" in res:
        assert "package not installed" in res["error"] or "Search error" in res["error"]
    else:
        assert "Python" in res["summary"]
