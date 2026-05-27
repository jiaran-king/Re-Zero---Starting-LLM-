#!/usr/bin/env python3
"""Prepare this Obsidian vault for GitHub Markdown rendering.

The script keeps the vault usable in Obsidian while replacing Obsidian-only
wiki links and embeds with standard Markdown links. It also renders Obsidian
Canvas files to simple static SVG previews for GitHub.
"""

from __future__ import annotations

import html
import json
import os
import re
import textwrap
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = ROOT / "08-图片" / "canvas-preview"
WIKILINK_RE = re.compile(r"(!?)\[\[([^\]]+)\]\]")

COLOR_MAP = {
    "1": ("#fff1f0", "#d4380d"),
    "2": ("#f6ffed", "#389e0d"),
    "3": ("#e6f4ff", "#0958d9"),
    "4": ("#f9f0ff", "#722ed1"),
    "5": ("#fff7e6", "#d46b08"),
    "6": ("#f0f5ff", "#1d39c4"),
}


def repo_files() -> list[Path]:
    return [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and ".obsidian" not in p.parts
        and p.relative_to(ROOT).as_posix() != "tools/github_render.py"
    ]


def markdown_files() -> list[Path]:
    return [p for p in repo_files() if p.suffix.lower() == ".md"]


def url_from_to(source: Path, target: Path, anchor: str = "") -> str:
    rel = os.path.relpath(target, source.parent).replace(os.sep, "/")
    url = quote(rel, safe="/")
    if anchor:
        url += "#" + quote(github_slug(anchor), safe="-_")
    return url


def display_name(target: str, alias: str | None, heading: str | None) -> str:
    if alias is not None:
        return alias.strip()
    if target:
        name = Path(target).name
        return Path(name).stem if Path(name).suffix else name
    if heading:
        return heading.strip()
    return "待填写"


def github_slug(heading: str) -> str:
    text = heading.strip().lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[`*_~\[\]()]|[：:，,。.!?？/\\|\"'“”‘’]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def split_wikilink(inner: str) -> tuple[str, str | None, str | None]:
    if "|" in inner:
        target_part, alias = inner.split("|", 1)
    else:
        target_part, alias = inner, None

    target_part = target_part.strip()
    if "#" in target_part:
        target, heading = target_part.split("#", 1)
        heading = heading.strip()
    else:
        target, heading = target_part, None

    target = target.strip()
    if "^" in target:
        target = target.split("^", 1)[0].strip()
    if heading and "^" in heading:
        heading = heading.split("^", 1)[0].strip()
    return target, heading, alias


class Resolver:
    def __init__(self) -> None:
        self.files = repo_files()
        self.by_rel = {p.relative_to(ROOT).as_posix(): p for p in self.files}
        self.by_rel_no_md = {
            rel[:-3]: p for rel, p in self.by_rel.items() if rel.endswith(".md")
        }
        self.by_name: dict[str, list[Path]] = {}
        self.by_stem: dict[str, list[Path]] = {}
        for p in self.files:
            self.by_name.setdefault(p.name, []).append(p)
            self.by_stem.setdefault(p.stem, []).append(p)

    def resolve(self, target: str, source: Path) -> Path | None:
        if not target or re.match(r"^[a-z]+://", target):
            return None

        candidates: list[Path] = []
        rel_source = (source.parent / target).resolve()
        try:
            rel_key = rel_source.relative_to(ROOT).as_posix()
            if rel_key in self.by_rel:
                candidates.append(self.by_rel[rel_key])
            if rel_key in self.by_rel_no_md:
                candidates.append(self.by_rel_no_md[rel_key])
        except ValueError:
            pass

        if target in self.by_rel:
            candidates.append(self.by_rel[target])
        if target in self.by_rel_no_md:
            candidates.append(self.by_rel_no_md[target])
        if target + ".md" in self.by_rel:
            candidates.append(self.by_rel[target + ".md"])

        name = Path(target).name
        candidates.extend(self.by_name.get(name, []))
        candidates.extend(self.by_stem.get(Path(name).stem, []))

        unique = []
        seen = set()
        for candidate in candidates:
            if candidate not in seen:
                unique.append(candidate)
                seen.add(candidate)

        if not unique:
            return None
        if len(unique) == 1:
            return unique[0]

        md_candidates = [p for p in unique if p.suffix.lower() == ".md"]
        if len(md_candidates) == 1:
            return md_candidates[0]
        return unique[0]


def node_point(node: dict, side: str | None) -> tuple[float, float]:
    x = float(node.get("x", 0))
    y = float(node.get("y", 0))
    w = float(node.get("width", 240))
    h = float(node.get("height", 140))
    if side == "right":
        return x + w, y + h / 2
    if side == "left":
        return x, y + h / 2
    if side == "top":
        return x + w / 2, y
    if side == "bottom":
        return x + w / 2, y + h
    return x + w / 2, y + h / 2


def clean_canvas_text(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = line.replace("**", "").replace("__", "")
        line = re.sub(r"^\s*[-*]\s+", "• ", line)
        if not line:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(line, width=30) or [""])
    return lines


def render_canvas(canvas_path: Path) -> Path:
    data = json.loads(canvas_path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        return canvas_path

    min_x = min(float(n.get("x", 0)) for n in nodes) - 40
    min_y = min(float(n.get("y", 0)) for n in nodes) - 40
    max_x = max(float(n.get("x", 0)) + float(n.get("width", 240)) for n in nodes) + 40
    max_y = max(float(n.get("y", 0)) + float(n.get("height", 140)) for n in nodes) + 40
    width = max_x - min_x
    height = max_y - min_y
    node_by_id = {n.get("id"): n for n in nodes}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min_x:g} {min_y:g} {width:g} {height:g}" width="1600" height="{max(360, int(1600 * height / width))}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#566070" />',
        "</marker>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#162033" flood-opacity="0.12" />',
        "</filter>",
        "</defs>",
        f'<rect x="{min_x:g}" y="{min_y:g}" width="{width:g}" height="{height:g}" fill="#fbfbf8" />',
    ]

    for edge in edges:
        source = node_by_id.get(edge.get("fromNode"))
        target = node_by_id.get(edge.get("toNode"))
        if not source or not target:
            continue
        x1, y1 = node_point(source, edge.get("fromSide"))
        x2, y2 = node_point(target, edge.get("toSide"))
        parts.append(
            f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
            'stroke="#566070" stroke-width="3" marker-end="url(#arrow)" />'
        )

    for node in nodes:
        x = float(node.get("x", 0))
        y = float(node.get("y", 0))
        w = float(node.get("width", 240))
        h = float(node.get("height", 140))
        fill, stroke = COLOR_MAP.get(str(node.get("color", "")), ("#ffffff", "#8c8c8c"))
        parts.append(
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="10" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2" filter="url(#shadow)" />'
        )
        text = node.get("text") or node.get("file") or node.get("url") or node.get("id", "")
        lines = clean_canvas_text(str(text))
        line_height = 22
        start_y = y + 30
        parts.append(
            f'<text x="{x + 18:g}" y="{start_y:g}" fill="#1f2933" '
            'font-family="Arial, sans-serif" font-size="17">'
        )
        for index, line in enumerate(lines[: max(1, int((h - 28) / line_height))]):
            dy = "0" if index == 0 else str(line_height)
            parts.append(f'<tspan x="{x + 18:g}" dy="{dy}">{html.escape(line)}</tspan>')
        parts.append("</text>")

    parts.append("</svg>\n")
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = PREVIEW_DIR / f"{canvas_path.stem}.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def render_all_canvas() -> dict[Path, Path]:
    previews: dict[Path, Path] = {}
    for canvas_path in sorted(ROOT.rglob("*.canvas")):
        if ".git" in canvas_path.parts:
            continue
        previews[canvas_path] = render_canvas(canvas_path)
    return previews


def convert_markdown(resolver: Resolver, previews: dict[Path, Path]) -> tuple[int, int]:
    converted = 0
    unresolved = 0

    for md_path in markdown_files():
        original = md_path.read_text(encoding="utf-8")

        def replace(match: re.Match[str]) -> str:
            nonlocal converted, unresolved
            is_embed = bool(match.group(1))
            target, heading, alias = split_wikilink(match.group(2))
            label = display_name(target, alias, heading)

            if target == "" and heading:
                converted += 1
                return f"[{label}](#{quote(github_slug(heading), safe='-_')})"

            resolved = resolver.resolve(target, md_path)
            if not resolved:
                unresolved += 1
                return label

            if is_embed:
                if resolved.suffix.lower() == ".canvas" and resolved in previews:
                    preview = previews[resolved]
                    converted += 1
                    preview_url = url_from_to(md_path, preview)
                    canvas_url = url_from_to(md_path, resolved)
                    return (
                        f"![{label}]({preview_url})\n\n"
                        f"[打开原始 Canvas]({canvas_url})"
                    )
                converted += 1
                return f"![{label}]({url_from_to(md_path, resolved)})"

            converted += 1
            return f"[{label}]({url_from_to(md_path, resolved, heading or '')})"

        updated = WIKILINK_RE.sub(replace, original)
        if updated != original:
            md_path.write_text(updated, encoding="utf-8")

    return converted, unresolved


def main() -> None:
    previews = render_all_canvas()
    resolver = Resolver()
    converted, unresolved = convert_markdown(resolver, previews)
    print(f"canvas previews: {len(previews)}")
    print(f"wikilinks converted: {converted}")
    print(f"wikilinks downgraded to text: {unresolved}")


if __name__ == "__main__":
    main()
