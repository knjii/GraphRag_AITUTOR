"""Статическая проверка команд дня аренды; исходники никогда не исполняются.

Поддерживается намеренно ограниченное подмножество Bash. «Неизвестно» —
не ошибка и не доказательство корректности: такие места требуют чтения.
"""

from __future__ import annotations

import argparse
import ast
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ENV = {"PATH", "HOME", "HF_HOME", "UV_DEFAULT_INDEX", "CUDA_VISIBLE_DEVICES",
              "PYTHONPATH", "VIRTUAL_ENV", "TOKENIZERS_PARALLELISM"}
OUTPUTS = {"--out", "--output", "--write", "--sheet", "--json"}
INPUTS = {"--dataset", "--check", "--from-trace", "--trace", "--goldset", "--chunks",
          "--path", "--prompt", "--parsed", "--key", "--grades", "--source"}


@dataclass
class Finding:
    step: str
    command: str
    message: str
    source: Path
    line: int
    unknown: bool = False

    def __str__(self) -> str:
        status = "неизвестно: " if self.unknown else ""
        return (f"{self.step} — {self.command} — {status}{self.message} — "
                f"{self.source}:{self.line}")


@dataclass
class Argument:
    names: list[str]
    line: int
    required: bool = False
    nargs: int | str = 1
    path: bool = False


@dataclass
class Schema:
    source: Path
    arguments: list[Argument] = field(default_factory=list)
    required_groups: dict[str, list[str]] = field(default_factory=dict)


def tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def literal(node: ast.AST | None, default: object = None) -> object:
    try:
        return ast.literal_eval(node) if node is not None else default
    except (ValueError, TypeError):
        return default


def script_schema(path: Path) -> Schema:
    result = Schema(path)
    module = tree(path)
    for node in ast.walk(module):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "add_mutually_exclusive_group"
                and any(k.arg == "required" and literal(k.value) is True
                        for k in node.value.keywords)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result.required_groups[target.id] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        names = [literal(arg) for arg in node.args]
        if not names or not all(isinstance(name, str) for name in names):
            continue
        group = ast.unparse(node.func.value)
        if group in result.required_groups:
            result.required_groups[group].extend(names)
        kw = {item.arg: item.value for item in node.keywords}
        nargs = literal(kw.get("nargs"), 1)
        if literal(kw.get("action")) in {"store_true", "store_false", "count", "help"}:
            nargs = 0
        positional = not names[0].startswith("-")
        result.arguments.append(Argument(
            names, node.lineno, bool(literal(kw.get("required"), False)) or
            (positional and nargs not in {"?", "*"}), nargs,
            ast.unparse(kw.get("type", ast.Constant(None))) == "Path"))
    return result


def cli_schemas(path: Path) -> dict[tuple[str, ...], Schema]:
    module = tree(path)
    groups = {"app": ()}
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_typer" and node.args:
                name = next((literal(k.value) for k in node.keywords if k.arg == "name"), None)
                if isinstance(node.args[0], ast.Name) and isinstance(name, str):
                    groups[node.args[0].id] = (*groups.get(ast.unparse(node.func.value), ()), name)
    result = {}
    for node in module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            if not (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)
                    and deco.func.attr == "command"):
                continue
            group = groups.get(ast.unparse(deco.func.value))
            if group is None:
                continue
            name = literal(deco.args[0]) if deco.args else node.name.replace("_", "-")
            schema = Schema(path)
            defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults
            for arg, default in zip(node.args.args, defaults, strict=True):
                annotation = ast.unparse(arg.annotation) if arg.annotation else ""
                declarations = [n for n in ast.walk(arg.annotation) if isinstance(n, ast.Call)
                                and isinstance(n.func, ast.Attribute)
                                and n.func.attr in {"Option", "Argument"}] if arg.annotation else []
                if not declarations:
                    continue
                decl = declarations[0]
                positional = decl.func.attr == "Argument"
                names = [literal(a) for a in decl.args if isinstance(literal(a), str)]
                if positional:
                    names = [arg.arg]
                elif not names:
                    names = ["--" + arg.arg.replace("_", "-")]
                    if "bool" in annotation:
                        names.append("--no-" + arg.arg.replace("_", "-"))
                names = [part for n in names for part in n.split("/")]
                schema.arguments.append(Argument(names, arg.lineno,
                    default is None or literal(default) is Ellipsis,
                    0 if "bool" in annotation and not positional else
                    ("+" if positional and "list[" in annotation else 1), "Path" in annotation))
            result[(*group, str(name))] = schema
    return result


def shell_lines(text: str) -> list[tuple[int, str]]:
    """Сохраняем исходные номера строк, исключая тела heredoc."""
    result = []
    pending = ""
    start = 1
    delimiter = None
    for number, line in enumerate(text.splitlines(), 1):
        if delimiter:
            if line.strip() == delimiter:
                delimiter = None
            continue
        if not pending:
            start = number
        pending += line.rstrip("\\").strip() + " "
        if line.endswith("\\") or (re.search(r"\w+=\(", pending) and not pending.rstrip().endswith(")")):
            continue
        match = re.search(r"<<-?\s*['\"]?(\w+)", pending)
        if match:
            delimiter = match[1]
        result.append((start, pending.strip()))
        pending = ""
    return result


def hide_substitutions(text: str) -> str:
    # Подстановку команды нельзя вычислять на ноутбуке.
    while "$(" in text:
        start = text.index("$(")
        depth, end = 1, start + 2
        while end < len(text) and depth:
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        text = text[:start] + "${DYNAMIC}" + text[end:]
    return text


def check(scenario: Path, root: Path = ROOT) -> list[Finding]:
    cli = cli_schemas(root / "rag_textbook/cli/main.py")
    config = root / "rag_textbook/config.py"
    aliases = {literal(k.value) for n in ast.walk(tree(config)) if isinstance(n, ast.Call)
               for k in n.keywords if k.arg in {"alias", "validation_alias"}}
    findings: list[Finding] = []
    variables = {"REPO_DIR": root.as_posix()}
    arrays: dict[str, list[str]] = {}
    produced: set[str] = set()
    external: set[str] = set()
    opaque: set[str] = set()
    prefixes: set[str] = set()
    step = "общий"
    cache: dict[str, Schema] = {}

    def expand(value: str) -> str:
        return re.sub(r"\$\{(\w+)\}|\$(\w+)",
                      lambda m: variables.get(m[1] or m[2], m[0]), value)

    def path_key(value: str) -> str:
        return (root / expand(value)).as_posix()

    for line, raw in shell_lines(scenario.read_text(encoding="utf-8-sig")):
        if not raw or raw.startswith("#"):
            continue
        function = re.match(r"(?:function\s+)?(\w+)\s*\(\)\s*\{", raw)
        if function:
            step = function[1].removeprefix("step_")
        safe = hide_substitutions(raw)
        try:
            lexer = shlex.shlex(safe, posix=True, punctuation_chars=";&|<>")
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            findings.append(Finding(step, raw, "синтаксис Bash вне поддерживаемого подмножества",
                                    scenario, line, True))
            continue
        # Массивы опций и env раскрываем лишь при буквальном объявлении.
        array = re.search(r"(?:^|\s)(\w+)=\((.*)\)\s*$", safe)
        if array:
            arrays[array[1]] = shlex.split(array[2], comments=True)
        expanded = []
        for token in tokens:
            ref = re.fullmatch(r"\$\{(\w+)\[@\]\}", token)
            expanded.extend(arrays.get(ref[1], [token]) if ref else [token])
        tokens = expanded
        # Это заявленные входы предварительной проверки, а не существующие файлы.
        if len(tokens) > 3 and tokens[:3] == ["for", "required", "in"]:
            external.update(path_key(t) for t in tokens[3:] if t not in {";", "do"})
        if tokens and tokens[0] in {"local", "export"}:
            assignments = tokens[1:]
        else:
            assignments = tokens[:1]
        for token in assignments:
            match = re.fullmatch(r"(\w+)=(.*)", token)
            if match and match[1] != "REPO_DIR":
                variables[match[1]] = expand(match[2])

        command_index = next((i for i, t in enumerate(tokens)
                              if t == "rag-textbook" or re.fullmatch(r"scripts/[\w-]+\.py", t)), None)
        # Строки echo/warn содержат примеры, а не команды.
        if tokens and tokens[0] in {"echo", "warn", "say", "ok", "printf"}:
            command_index = None
        env_tokens = tokens[:command_index] if command_index is not None else (
            tokens[1:] if tokens and tokens[0] == "export" else [])
        for token in env_tokens:
            name = token.split("=", 1)[0]
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", name) and name not in aliases | SYSTEM_ENV:
                findings.append(Finding(step, raw, f"переменная {name} не объявлена в {config} (alias)",
                                        scenario, line))

        def read_file(value: str, step: str = step, raw: str = raw, line: int = line) -> None:
            value = expand(value)
            if path_key(value) in produced or ("$" not in value and (root / value).exists()):
                return
            dynamic = any(c in value for c in "$*?[")
            # Побочные файлы --out и вывод heredoc нельзя вывести из одной опции.
            candidate = path_key(value)
            derived = any(candidate.startswith(p + "-") or candidate.startswith(p + ".")
                          for p in prefixes)
            unknown = dynamic or derived or candidate in external | opaque
            findings.append(Finding(step, raw, f"вход {value}: " +
                                    ("источник не установлен" if unknown else "нет файла и более раннего производителя"),
                                    scenario, line, unknown))

        new_outputs = []
        if command_index is not None:
            command = tokens[command_index]
            args = tokens[command_index + 1:]
            end = next((i for i, t in enumerate(args) if t in {"|", "||", "&&", ";", ">", ">>", "<", "<<", "&"}), len(args))
            # Дескриптор 2 перед перенаправлением не является позиционным аргументом.
            args = args[:end]
            if args and args[-1] == "2" and end < len(tokens) and "2>" in raw:
                args = args[:-1]
            if command == "rag-textbook":
                key = next((key for key in sorted(cli, key=len, reverse=True)
                            if tuple(args[:len(key)]) == key), None)
                if key is None:
                    findings.append(Finding(step, raw, "неизвестная подкоманда rag-textbook; "
                                            f"объявления: {root / 'rag_textbook/cli/main.py'}:1", scenario, line))
                    continue
                schema = cli[key]
                args = args[len(key):]
            else:
                if not (root / command).is_file():
                    findings.append(Finding(step, raw, f"нет скрипта {command}", scenario, line))
                    continue
                if command not in cache:
                    cache[command] = script_schema(root / command)
                schema = cache[command]
                key = ()
            options = {name: arg for arg in schema.arguments for name in arg.names if name.startswith("-")}
            options["--help"] = Argument(["--help"], 1, nargs=0)
            seen = set()
            positionals = []
            uncertain = any("[@]}" in arg for arg in args)
            i = 0
            while i < len(args):
                token = args[i]
                name, sep, inline = token.partition("=")
                if name == "--":
                    positionals.extend(args[i + 1:])
                    break
                if name.startswith("-"):
                    spec = options.get(name)
                    if spec is None:
                        findings.append(Finding(step, raw, f"неизвестная опция {name}; объявления: {schema.source}:1", scenario, line))
                        i += 1
                        continue
                    seen.update(spec.names)
                    values = [inline] if sep else []
                    count = spec.nargs
                    while not sep and count != 0 and i + 1 < len(args) and not args[i + 1].startswith("--"):
                        i += 1
                        values.append(args[i])
                        if isinstance(count, int) and len(values) >= count or count == "?":
                            break
                    if not values and count not in {0, "?", "*"}:
                        findings.append(Finding(step, raw, f"нет значения {name}; {schema.source}:{spec.line}", scenario, line))
                    output = name in OUTPUTS or (key == ("eval", "run") and name == "--trace")
                    for value in values:
                        if output:
                            new_outputs.append(path_key(value))
                            if name == "--out":
                                prefixes.add(path_key(value))
                        elif spec.path or name in INPUTS:
                            read_file(value)
                elif "[@]}" not in token:
                    positionals.append(token)
                i += 1
            for spec in schema.arguments:
                if spec.names[0].startswith("-"):
                    missing = spec.required and not seen.intersection(spec.names)
                else:
                    missing = spec.required and not positionals
                    take = len(positionals) if spec.nargs in {"+", "*"} else 1
                    values, positionals = positionals[:take], positionals[take:]
                    if spec.path:
                        for value in values:
                            read_file(value)
                if missing and "--help" not in seen:
                    findings.append(Finding(step, raw, f"обязательный аргумент {spec.names[0]} отсутствует; "
                                            f"{schema.source}:{spec.line}", scenario, line, uncertain))
            for names in schema.required_groups.values():
                if names and not seen.intersection([*names, "--help"]):
                    findings.append(Finding(step, raw,
                        f"обязателен один из аргументов {', '.join(names)}; {schema.source}:1",
                        scenario, line, uncertain))
            if uncertain:
                findings.append(Finding(step, raw, "динамический массив аргументов", scenario, line, True))
        # Вход читается до регистрации выходов той же команды.
        for index, token in enumerate(tokens[:-1]):
            if token == "<":
                read_file(tokens[index + 1])
            if token in {">", ">>"}:
                new_outputs.append(path_key(tokens[index + 1]))
        produced.update(new_outputs)
        if "<<" in raw:
            opaque.update(path_key(t) for t in tokens if "/" in t or t.endswith((".json", ".jsonl")))
            findings.append(Finding(step, raw, "потоки файлов внутри heredoc не анализируются", scenario, line, True))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, nargs="?", default=ROOT / "deploy/day1.sh")
    args = parser.parse_args()
    findings = check(args.scenario)
    for finding in findings:
        print(finding)
    errors = sum(not finding.unknown for finding in findings)
    print(f"Расхождений: {errors}; неизвестно: {len(findings) - errors}")
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
