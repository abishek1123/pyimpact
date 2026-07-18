"""Function-level call graph construction.

Where the import graph works at file granularity, this works at *function*
granularity: nodes are individual functions/methods and an edge ``A -> B`` means
"function A calls function B". Answering "what breaks if I change function X"
needs this finer resolution — importing a file doesn't mean every function in it
is affected.

Node id scheme
--------------
``"<module>:<qualname>"`` where qualname is the function name for a top-level
function (``app.auth:login``) or ``Class.method`` for a method
(``app.models:User.check_password``). The ``:`` separates the module (dotted)
from the in-module qualified name so the two never collide.

What it resolves
----------------
- ``foo()``            -> a top-level function in the same module
- ``bar()``            -> a name brought in via ``from mod import bar``
- ``self.method()``    -> a method on the enclosing class
- ``mod.func()`` / ``pkg.mod.func()`` -> a function in an imported repo module

Known limitations (documented on purpose — see README)
------------------------------------------------------
- ``obj.method()`` where ``obj`` is an instance variable/parameter is NOT
  resolved: that needs type inference, which static analysis of dynamically
  typed Python can't do reliably.
- Class instantiation (``User()``) does not create an edge to ``__init__``.
- ``getattr(obj, name)()`` and other dynamic dispatch is not resolved.
- ``from mod import *`` is not tracked.
- Calls inside a nested ``def`` are attributed to the enclosing top-level
  function (an over-approximation), and nested functions are not call targets.
"""

from __future__ import annotations

import ast
from typing import Optional

import networkx as nx

from .import_graph import find_python_files, module_name_for


def node_id(module: str, qualname: str) -> str:
    """Compose the call-graph node id for a function/method."""
    return f"{module}:{qualname}"


def _package_of(module: str) -> str:
    return module.rpartition(".")[0]


def _resolve_from_base(module: Optional[str], level: int,
                       importer_module: str, is_package: bool) -> str:
    """Resolve the base module of a ``from ... import`` to an absolute dotted path.

    ``level`` is the number of leading dots (0 = absolute). Mirrors the relative
    resolution in the import graph but works on the structured ``ImportFrom``
    fields instead of a flattened string.
    """
    if level == 0:
        return module or ""
    anchor = importer_module if is_package else _package_of(importer_module)
    for _ in range(level - 1):
        anchor = _package_of(anchor)
    if anchor and module:
        return f"{anchor}.{module}"
    return anchor or module or ""


def _flatten_attr_chain(func: ast.AST):
    """Flatten ``a.b.c`` into ``("a", ["b", "c"])``; return None if not a pure
    Name.attr…attr chain (e.g. the base is a call or subscript)."""
    attrs: list[str] = []
    cur = func
    while isinstance(cur, ast.Attribute):
        attrs.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        attrs.reverse()
        return cur.id, attrs
    return None


class _ModuleInfo:
    """Per-module resolution tables collected in the first pass."""

    __slots__ = (
        "module", "is_package", "functions", "classes",
        "module_aliases", "symbol_aliases",
    )

    def __init__(self, module: str, is_package: bool):
        self.module = module
        self.is_package = is_package
        self.functions: set[str] = set()          # top-level function names
        self.classes: dict[str, set[str]] = {}    # class name -> method names
        # local name -> absolute repo/external module (import x, import x as y,
        # from . import submod)
        self.module_aliases: dict[str, str] = {}
        # local name -> (module, symbol) for `from module import symbol`
        self.symbol_aliases: dict[str, tuple[str, str]] = {}


class CallGraphBuilder:
    """Two-pass builder: collect definitions + imports, then resolve calls."""

    def __init__(self, repo_path: str):
        import os
        self.repo_path = os.path.abspath(repo_path)
        self.graph = nx.DiGraph()
        self.infos: dict[str, _ModuleInfo] = {}   # module -> info
        self.trees: dict[str, ast.Module] = {}    # module -> AST
        # (module, qualname) -> the FunctionDef AST, plus its class context
        self._defs: list[tuple[str, str, ast.AST, Optional[str]]] = []

    # -- pass 1 -------------------------------------------------------------

    def _collect(self):
        import os
        from .parser import parse_file

        for path in find_python_files(self.repo_path):
            module = module_name_for(path, self.repo_path)
            is_package = os.path.basename(path) == "__init__.py"
            try:
                tree = parse_file(path)
            except SyntaxError:
                # Unparseable file: still record the module, contribute no defs.
                self.infos[module] = _ModuleInfo(module, is_package)
                continue

            info = _ModuleInfo(module, is_package)
            self.infos[module] = info
            self.trees[module] = tree

            for stmt in tree.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    info.functions.add(stmt.name)
                    nid = node_id(module, stmt.name)
                    self.graph.add_node(nid, module=module, qualname=stmt.name,
                                        kind="function")
                    self._defs.append((module, stmt.name, stmt, None))
                elif isinstance(stmt, ast.ClassDef):
                    methods: set[str] = set()
                    for item in stmt.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.add(item.name)
                            qual = f"{stmt.name}.{item.name}"
                            nid = node_id(module, qual)
                            self.graph.add_node(nid, module=module, qualname=qual,
                                                kind="method")
                            self._defs.append((module, qual, item, stmt.name))
                    info.classes[stmt.name] = methods

            self._collect_imports(tree, info)

    def _collect_imports(self, tree: ast.Module, info: _ModuleInfo):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        info.module_aliases[alias.asname] = alias.name
                    else:
                        # `import a.b.c` binds `a`; chain resolution rebuilds
                        # the full path from the attribute chain at the call site.
                        top = alias.name.split(".")[0]
                        info.module_aliases[top] = top
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_from_base(node.module, node.level,
                                          info.module, info.is_package)
                for alias in node.names:
                    if alias.name == "*":
                        continue  # star imports are not tracked
                    local = alias.asname or alias.name
                    full = f"{base}.{alias.name}" if base else alias.name
                    # We can't yet know if `full` is a submodule or a symbol;
                    # decide in pass 2 against the known-module set. Store both
                    # interpretations: prefer submodule if it turns out to exist.
                    info.symbol_aliases[local] = (base, alias.name)
                    info.module_aliases.setdefault(local, full)

    # -- pass 2 -------------------------------------------------------------

    def _resolve(self):
        known_modules = set(self.infos.keys())

        for module, qualname, fdef, class_ctx in self._defs:
            info = self.infos[module]
            src = node_id(module, qualname)
            methods = info.classes.get(class_ctx, set()) if class_ctx else set()

            for call in ast.walk(fdef):
                if not isinstance(call, ast.Call):
                    continue
                target = self._resolve_call(call.func, info, known_modules,
                                            class_ctx, methods)
                if target is not None and self.graph.has_node(target) \
                        and target != src:
                    # Record every call-site line so downstream layers (e.g. the
                    # AI explainer) can ground "A calls B" in concrete locations.
                    if self.graph.has_edge(src, target):
                        self.graph.edges[src, target]["linenos"].append(call.lineno)
                    else:
                        self.graph.add_edge(src, target, linenos=[call.lineno])

    def _resolve_call(self, func: ast.AST, info: _ModuleInfo,
                      known_modules: set[str], class_ctx: Optional[str],
                      methods: set[str]) -> Optional[str]:
        # Case 1: bare name call -> foo()
        if isinstance(func, ast.Name):
            name = func.id
            if name in info.functions:
                return node_id(info.module, name)
            if name in info.symbol_aliases:
                mod, sym = info.symbol_aliases[name]
                if sym in self.infos.get(mod, _ModuleInfo("", False)).functions:
                    return node_id(mod, sym)
            return None

        # Case 2: attribute call -> a.b.func()
        flat = _flatten_attr_chain(func)
        if flat is None:
            return None
        base, attrs = flat
        func_name = attrs[-1]
        mid = attrs[:-1]

        # self.method() inside a class
        if base == "self" and class_ctx and not mid:
            if func_name in methods:
                return node_id(info.module, f"{class_ctx}.{func_name}")
            return None

        # imported-module function: mod.func() / pkg.mod.func()
        if base in info.module_aliases:
            base_mod = info.module_aliases[base]
            module_path = ".".join([base_mod, *mid]) if mid else base_mod
            if module_path in known_modules and \
                    func_name in self.infos[module_path].functions:
                return node_id(module_path, func_name)

        return None

    def build(self) -> nx.DiGraph:
        self._collect()
        self._resolve()
        return self.graph


def build_call_graph(repo_path: str) -> nx.DiGraph:
    """Build the function-level call graph for a repo. See module docstring."""
    return CallGraphBuilder(repo_path).build()
