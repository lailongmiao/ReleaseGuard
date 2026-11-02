"""Read-only Markdown paths, anchors, fences, tables and whitespace gate."""

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

from pydantic import BaseModel
from rich.console import Console


class DocumentCheck(BaseModel):
    """Bounded diagnostics contain paths/reasons, never document contents."""

    files: int
    local_links: int
    tables: int
    errors: list[tuple[str, int, str]]


def check_documents(root: Path, paths: list[Path]) -> DocumentCheck:
    """Validate explicit files without scanning private/ignored local files."""
    clean: dict[Path, list[tuple[int, str]]] = {}
    ids: dict[Path, set[str]] = {}
    errors: list[tuple[str, int, str]] = []
    tables, links = 0, 0
    for path in paths:
        raw = path.read_bytes()
        name = path.relative_to(root).as_posix()
        if not raw.endswith(b"\n"):
            errors.append((name, 0, "final_newline"))
        lines: list[tuple[int, str]] = []
        fence: str | None = None
        anchors: set[str] = set()
        counts: dict[str, int] = {}
        for n, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            if line.rstrip() != line:
                errors.append((name, n, "trailing_whitespace"))
            match = re.match(r"^\s*(`{3,}|~{3,})", line)
            if match:
                token = match.group(1)
                if fence is None:
                    fence = token
                elif token[0] == fence[0] and len(token) >= len(fence):
                    fence = None
                continue
            if fence:
                continue
            lines.append((n, line))
            anchors.update(re.findall(r"<a\s+id=[\"\x27]([^\"\x27]+)", line))
            heading = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
            if heading:
                slug = re.sub(r"<[^>]+>", "", heading.group(1)).strip().lower()
                slug = re.sub(r"[^\w\-\s]", "", slug).replace(" ", "-")
                count = counts.get(slug, 0)
                counts[slug] = count + 1
                anchors.add(slug if count == 0 else f"{slug}-{count}")
        if fence:
            errors.append((name, 0, "unclosed_fence"))
        clean[path.resolve()], ids[path.resolve()] = lines, anchors
        group: list[tuple[int, str]] = []
        for n, line in lines + [(0, "")]:
            if line.strip().startswith("|"):
                group.append((n, line))
                continue
            if len(group) > 1 and re.match(r"^\s*\|\s*:?-{3,}", group[1][1]):
                tables += 1
                sizes = [
                    (number, len(re.split(r"(?<!\\)\|", re.sub(r"`[^`]*`", "CODE", row).strip().strip("|"))))
                    for number, row in group
                ]
                errors.extend((name, number, "table_columns") for number, size in sizes if size != sizes[0][1])
            group = []
    for path, lines in clean.items():
        for n, line in lines:
            for target in re.findall(r"!?\[[^\]\n]+\]\(([^)\n]+)\)", line):
                target = target.strip()
                if target.startswith("<") and ">" in target:
                    target = target[1 : target.index(">")]
                target = re.split(r"\s+[\"\x27]", target, maxsplit=1)[0]
                if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("//"):
                    continue
                name, _, fragment = unquote(target).partition("#")
                name = name.split("?", 1)[0]
                links += 1
                destination = (path.parent / name).resolve() if name else path
                if not destination.exists():
                    errors.append((path.relative_to(root).as_posix(), n, "missing_target"))
                elif fragment and destination in ids and fragment not in ids[destination]:
                    errors.append((path.relative_to(root).as_posix(), n, "missing_anchor"))
    return DocumentCheck(files=len(paths), local_links=links, tables=tables, errors=errors)


def main() -> int:
    """Run the repository gate; never repair or reformat documents."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = (
        subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    paths = sorted({root / name for name in names if name.endswith(".md") and (root / name).is_file()})
    result = check_documents(root, paths)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    Console().print(result.model_dump())
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
