"""Function-level call graph construction.

Nodes are functions, methods and classes; an edge ``A -> B`` means "A depends on
B" at the code level. Answering "what breaks if I change function X" needs this
granularity — importing a file doesn't mean every function in it is affected.

Node ids
--------
``"<module>:<qualname>"``: ``app.auth:login`` (function), ``app.models:User``
(class), ``app.models:User.check`` (method). The ``:`` separates the dotted
module from the in-module name so the two never collide.

Edge kinds (all point from the dependent to the dependency)
-----------------------------------------------------------
``call``         A calls B. ``strength`` says how sure we are:
                 ``direct``    a resolved call site,
                 ``reference`` B is passed around as a value (callback,
                               decorator, ``Thread(target=B)``),
                 ``virtual``   B overrides the statically resolved method in a
                               subclass, so dynamic dispatch may reach it
                               (class-hierarchy analysis).
``instantiate``  A constructs class B.
``uses``         A otherwise references class B (isinstance, annotations,
                 ``except B``, class attributes).

How names are resolved
----------------------
Python is dynamically typed, so the resolver does a light, flow-insensitive
type inference inside each scope:

- module symbols: local defs, ``import``/``from`` imports, relative imports and
  re-exports through ``__init__.py`` (``from requests import Session``);
- ``self``/``cls`` bound to the enclosing class, with inheritance (a simple MRO)
  and ``super()``;
- local variables whose every assignment agrees on a type: ``x = Foo()``,
  ``with Foo() as x``, ``x = make_foo()`` when ``make_foo`` is annotated
  ``-> Foo``, and annotated parameters ``def f(x: Foo)`` (``Optional``,
  ``X | None`` and string forward references included);
- instance attributes assigned anywhere in the class (``self.db = Database()``,
  ``self.db: Database``) so ``self.db.query()`` resolves;
- module-level instances (``app = Flask(__name__)``);
- ``getattr(obj, "literal")``.

Anything that needs real runtime information — ``getattr`` with a computed name,
monkey-patching, values flowing through containers — stays unresolved. The tool
prefers a missing edge over an invented one.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from .index import RepoIndex, ensure_index

# Higher wins when the same edge is found more than once.
STRENGTH_RANK = {"virtual": 1, "reference": 2, "direct": 3}
KIND_RANK = {"uses": 1, "instantiate": 2, "call": 3}
MAX_VIRTUAL_TARGETS = 25
_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPE_TYPES = frozenset({ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                          ast.Lambda})


def node_id(module: str, qualname: str) -> str:
    """Compose the node id for a function/method/class."""
    return f"{module}:{qualname}"


def _package_of(module: str) -> str:
    return module.rpartition(".")[0]


def _resolve_from_base(module: str | None, level: int,
                       importer_module: str, is_package: bool) -> str:
    """Absolute dotted base of a ``from ... import`` (``level`` = leading dots)."""
    if level == 0:
        return module or ""
    anchor = importer_module if is_package else _package_of(importer_module)
    for _ in range(level - 1):
        anchor = _package_of(anchor)
    if anchor and module:
        return f"{anchor}.{module}"
    return anchor or module or ""


def _toplevel(body):
    """Statements at module/class level, looking through if/try/with blocks
    (version guards, optional-import fallbacks) but not into defs."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.If, ast.With, ast.AsyncWith)):
            yield from _toplevel(stmt.body)
            yield from _toplevel(getattr(stmt, "orelse", []))
        elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
            yield from _toplevel(stmt.body)
            for handler in stmt.handlers:
                yield from _toplevel(handler.body)
            yield from _toplevel(stmt.orelse)
            yield from _toplevel(stmt.finalbody)


@dataclass
class _Scan:
    """Everything the resolver needs from one scope, gathered in one traversal.

    Walking each function body once (instead of once per question asked about
    it) is what keeps the builder linear on large trees.
    """

    assigns: list = field(default_factory=list)       # (name, value, annotation)
    attr_assigns: list = field(default_factory=list)  # (base, attr, value, annotation)
    calls: list = field(default_factory=list)         # ast.Call nodes
    refs: list = field(default_factory=list)          # maximal Name/Attribute loads


def _bind(scan: _Scan, target, value, annotation):
    if type(target) is ast.Name:
        scan.assigns.append((target.id, value, annotation))
    elif type(target) is ast.Attribute and type(target.value) is ast.Name:
        scan.attr_assigns.append((target.value.id, target.attr, value, annotation))


def _scan(roots) -> _Scan:
    """Pre-order walk of ``roots``. Calls and references are collected
    everywhere (nested functions included); assignments only in the scope
    itself, since nested scopes bind their own names."""
    scan = _Scan()
    skip: set = set()
    root_ids = {id(r) for r in roots}
    stack = [(r, False) for r in roots]
    AST = ast.AST
    while stack:
        node, nested = stack.pop()
        t = type(node)
        if t is ast.Call:
            scan.calls.append(node)
            skip.add(id(node.func))
        elif t is ast.Attribute:
            skip.add(id(node.value))          # resolve only maximal chains
            if id(node) not in skip and type(node.ctx) is ast.Load:
                scan.refs.append(node)
        elif t is ast.Name:
            if type(node.ctx) is ast.Load and id(node) not in skip:
                scan.refs.append(node)
        elif not nested:
            if t is ast.Assign:
                for target in node.targets:
                    _bind(scan, target, node.value, None)
            elif t is ast.AnnAssign:
                _bind(scan, node.target, node.value, node.annotation)
            elif t is ast.With or t is ast.AsyncWith:
                for item in node.items:
                    if type(item.optional_vars) is ast.Name:
                        scan.assigns.append((item.optional_vars.id, item.context_expr, None))
            elif t is ast.NamedExpr and type(node.target) is ast.Name:
                scan.assigns.append((node.target.id, node.value, None))
        child_nested = nested or (t in _SCOPE_TYPES and id(node) not in root_ids)
        for name in node._fields:
            value = getattr(node, name, None)
            if type(value) is list:
                for item in value:
                    if isinstance(item, AST):
                        stack.append((item, child_nested))
            elif isinstance(value, AST):
                stack.append((value, child_nested))
    return scan


def _first_line(fdef) -> int:
    lines = [fdef.lineno] + [d.lineno for d in fdef.decorator_list]
    return min(lines)


def _decorator_names(fdef) -> set[str]:
    names = set()
    for d in fdef.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _short(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _signature(fdef) -> str:
    try:
        sig = f"({ast.unparse(fdef.args)})"
    except Exception:  # pragma: no cover - defensive
        sig = "(...)"
    return _short(sig, 160)


@dataclass(eq=False)
class ClassInfo:
    module: str
    name: str
    node: ast.ClassDef
    methods: dict = field(default_factory=dict)   # name -> FunctionDef

    @property
    def id(self) -> str:
        return node_id(self.module, self.name)


@dataclass(eq=False)
class ModuleInfo:
    module: str
    is_package: bool
    tree: ast.Module
    functions: dict = field(default_factory=dict)        # name -> FunctionDef
    classes: dict = field(default_factory=dict)          # name -> ClassInfo
    module_aliases: dict = field(default_factory=dict)   # local -> dotted module
    symbol_aliases: dict = field(default_factory=dict)   # local -> (module, name)


@dataclass(eq=False)
class _Ctx:
    """Name-resolution context for one scope."""

    info: ModuleInfo
    cls: ClassInfo | None = None
    self_name: str | None = None   # first param bound to an instance
    cls_name: str | None = None    # first param bound to the class
    locals: dict = field(default_factory=dict)


# Ref shapes produced by the resolver (plain tuples keep it light):
#   ("module", dotted)               ("modprefix", dotted)  partial `import a.b`
#   ("function", module, name)       ("class", ClassInfo)
#   ("instance", ClassInfo)          ("clsvar", ClassInfo)  `cls` in a classmethod
#   ("method", definer, name, receiver_or_None)
#   ("super", ClassInfo)


class CallGraphBuilder:
    """Collect definitions, infer types, then resolve every call and reference."""

    def __init__(self, repo, exclude=()):
        self.index: RepoIndex = ensure_index(repo, exclude)
        self.graph = nx.DiGraph()
        self.module_graph = nx.DiGraph()   # module-level code -> dependency
        self.infos: dict[str, ModuleInfo] = {}
        self._bases: dict[int, list[ClassInfo]] = {}
        self._mro: dict[int, list[ClassInfo]] = {}
        self._subclasses: dict[int, list[ClassInfo]] = defaultdict(list)
        self._ctx_cache: dict[int, _Ctx] = {}
        self._globals: dict[str, dict] = {}
        self._attr_cache: dict[tuple, ClassInfo | None] = {}
        self._busy: set = set()
        self._scans: dict = {}
        self._attr_tables: dict = {}

    # -- pass 1: definitions -------------------------------------------------

    def _register(self, table: dict, key: tuple, name: str, fdef):
        """Record a definition. Like Python, the last one wins — except that
        ``@overload`` stubs never displace a real implementation. Every
        definition is kept so all of their bodies contribute edges."""
        current = table.get(name)
        if current is None or not _is_overload(fdef) or _is_overload(current):
            table[name] = fdef
        self._all_defs.setdefault(key, []).append(fdef)

    def _collect(self):
        self._all_defs: dict[tuple, list] = {}
        for sf in self.index.parsed():
            info = ModuleInfo(sf.module, sf.is_package, sf.tree)
            self.infos[sf.module] = info
            for stmt in _toplevel(sf.tree.body):
                if isinstance(stmt, _FUNC_TYPES):
                    self._register(info.functions, (sf.module, stmt.name),
                                   stmt.name, stmt)
                elif isinstance(stmt, ast.ClassDef) and stmt.name not in info.classes:
                    cls = ClassInfo(sf.module, stmt.name, stmt)
                    for item in _toplevel(stmt.body):
                        if isinstance(item, _FUNC_TYPES):
                            self._register(cls.methods,
                                           (sf.module, f"{stmt.name}.{item.name}"),
                                           item.name, item)
                    info.classes[stmt.name] = cls
                    self.graph.add_node(
                        cls.id, module=sf.module, qualname=stmt.name, kind="class",
                        lineno=stmt.lineno, start_lineno=_first_line(stmt),
                        end_lineno=stmt.end_lineno, is_test=sf.is_test,
                        doc=_short(ast.get_docstring(stmt), 300), bases=[],
                        spans=[[_first_line(stmt), stmt.end_lineno]],
                    )
            for name, fdef in info.functions.items():
                self._add_func_node(sf, name, fdef, None)
            for cls in info.classes.values():
                for name, fdef in cls.methods.items():
                    self._add_func_node(sf, f"{cls.name}.{name}", fdef, cls)
            self._collect_imports(info, sf.imports)

    def _add_func_node(self, sf, qualname, fdef, cls: ClassInfo | None):
        defs = self._all_defs[(sf.module, qualname)]
        self.graph.add_node(
            node_id(sf.module, qualname), module=sf.module, qualname=qualname,
            kind="method" if cls else "function", class_id=cls.id if cls else None,
            lineno=fdef.lineno, start_lineno=_first_line(fdef),
            end_lineno=fdef.end_lineno, is_test=sf.is_test,
            is_async=isinstance(fdef, ast.AsyncFunctionDef),
            signature=_signature(fdef), doc=_short(ast.get_docstring(fdef), 300),
            spans=[[_first_line(d), d.end_lineno] for d in defs],
        )

    def _collect_imports(self, info: ModuleInfo, nodes):
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        info.module_aliases[alias.asname] = alias.name
                    else:
                        # `import a.b.c` binds `a`; attribute chains at the use
                        # site rebuild the full dotted path.
                        top = alias.name.split(".")[0]
                        info.module_aliases.setdefault(top, top)
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_from_base(node.module, node.level,
                                          info.module, info.is_package)
                for alias in node.names:
                    if alias.name != "*":
                        info.symbol_aliases[alias.asname or alias.name] = (base, alias.name)

    # -- class hierarchy -----------------------------------------------------

    def _resolve_bases(self):
        for info in self.infos.values():
            ctx = _Ctx(info)
            for cls in info.classes.values():
                bases = []
                for expr in cls.node.bases:
                    if isinstance(expr, ast.Subscript):   # Generic[T], Base[int]
                        expr = expr.value
                    ref = self._resolve_expr(expr, ctx)
                    if ref and ref[0] == "class" and ref[1] is not cls:
                        bases.append(ref[1])
                self._bases[id(cls)] = bases
                self.graph.nodes[cls.id]["bases"] = [b.id for b in bases]
                for base in bases:
                    self._subclasses[id(base)].append(cls)

    def _mro_of(self, cls: ClassInfo) -> list[ClassInfo]:
        key = id(cls)
        if key in self._mro:
            return self._mro[key]
        self._mro[key] = [cls]            # cycle guard for pathological code
        order = [cls]
        for base in self._bases.get(key, []):
            for c in self._mro_of(base):
                if c not in order:
                    order.append(c)
        self._mro[key] = order
        return order

    def _lookup(self, cls: ClassInfo, name: str, skip_self=False):
        mro = self._mro_of(cls)
        for c in (mro[1:] if skip_self else mro):
            if name in c.methods:
                return c
        return None

    def _all_subclasses(self, cls: ClassInfo) -> list[ClassInfo]:
        seen, out, stack = {id(cls)}, [], list(self._subclasses.get(id(cls), []))
        while stack:
            c = stack.pop()
            if id(c) in seen:
                continue
            seen.add(id(c))
            out.append(c)
            stack.extend(self._subclasses.get(id(c), []))
        return out

    # -- symbol & expression resolution --------------------------------------

    def _resolve_symbol(self, module: str, name: str, depth: int = 0):
        info = self.infos.get(module)
        if info is None or depth > 10:
            return None
        if name in info.functions:
            return ("function", module, name)
        if name in info.classes:
            return ("class", info.classes[name])
        if name in info.symbol_aliases:
            base, sym = info.symbol_aliases[name]
            full = f"{base}.{sym}" if base else sym
            if full in self.infos:
                return ("module", full)
            if base in self.infos:
                return self._resolve_symbol(base, sym, depth + 1)
            return None
        if name in info.module_aliases:
            target = info.module_aliases[name]
            return ("module", target) if target in self.infos else ("modprefix", target)
        return None

    def _module_globals(self, module: str) -> dict:
        if module in self._globals:
            return self._globals[module]
        self._globals[module] = {}        # guard against self-reference
        info = self.infos[module]
        scan = self._module_scan(info)
        self._globals[module] = self._infer_locals(scan, _Ctx(info), params=())
        return self._globals[module]

    def _scan_of(self, key, roots) -> _Scan:
        scan = self._scans.get(key)
        if scan is None:
            scan = self._scans[key] = _scan(roots)
        return scan

    def _module_scan(self, info: ModuleInfo) -> _Scan:
        return self._scan_of(("module", info.module), _import_time_roots(info.tree.body))

    def _class_scan(self, cls: ClassInfo) -> _Scan:
        roots = [*cls.node.decorator_list, *cls.node.keywords,
                 *_import_time_roots(cls.node.body)]
        return self._scan_of(("class", id(cls)), roots)

    def _resolve_name(self, name: str, ctx: _Ctx):
        if ctx.cls is not None:
            if name == ctx.self_name:
                return ("instance", ctx.cls)
            if name == ctx.cls_name:
                return ("clsvar", ctx.cls)
        if name in ctx.locals:
            return ctx.locals[name]
        ref = self._resolve_symbol(ctx.info.module, name)
        if ref is not None:
            return ref
        return self._module_globals(ctx.info.module).get(name)

    def _attr_of(self, ref, attr: str):
        kind = ref[0]
        if kind == "module":
            sub = f"{ref[1]}.{attr}"
            if sub in self.infos:
                return ("module", sub)
            return self._resolve_symbol(ref[1], attr)
        if kind == "modprefix":
            sub = f"{ref[1]}.{attr}"
            return ("module", sub) if sub in self.infos else ("modprefix", sub)
        if kind == "class":
            definer = self._lookup(ref[1], attr)
            return ("method", definer, attr, None) if definer else None
        if kind in ("instance", "clsvar"):
            definer = self._lookup(ref[1], attr)
            if definer:
                return ("method", definer, attr, ref[1])
            attr_cls = self._attr_type(ref[1], attr)
            return ("instance", attr_cls) if attr_cls else None
        if kind == "super":
            definer = self._lookup(ref[1], attr, skip_self=True)
            return ("method", definer, attr, None) if definer else None
        return None

    def _call_result(self, ref):
        """The type of value produced by calling ``ref``."""
        kind = ref[0]
        if kind in ("class", "clsvar"):
            return ("instance", ref[1])
        if kind == "function":
            fdef = self.infos[ref[1]].functions[ref[2]]
            cls = self._annotation_class(fdef.returns, self.infos[ref[1]])
            return ("instance", cls) if cls else None
        if kind == "method":
            definer, name, receiver = ref[1], ref[2], ref[3]
            fdef = definer.methods[name]
            cls = self._annotation_class(fdef.returns, self.infos[definer.module],
                                         self_cls=receiver or definer)
            return ("instance", cls) if cls else None
        return None

    def _resolve_expr(self, expr, ctx: _Ctx):
        if isinstance(expr, ast.Name):
            return self._resolve_name(expr.id, ctx)
        if isinstance(expr, ast.Attribute):
            base = self._resolve_expr(expr.value, ctx)
            return self._attr_of(base, expr.attr) if base else None
        if isinstance(expr, ast.Call):
            func = expr.func
            if isinstance(func, ast.Name) and func.id == "super" and ctx.cls:
                return ("super", ctx.cls)
            if (isinstance(func, ast.Name) and func.id == "getattr"
                    and len(expr.args) >= 2
                    and isinstance(expr.args[1], ast.Constant)
                    and isinstance(expr.args[1].value, str)):
                base = self._resolve_expr(expr.args[0], ctx)
                return self._attr_of(base, expr.args[1].value) if base else None
            callee = self._resolve_expr(func, ctx)
            return self._call_result(callee) if callee else None
        if isinstance(expr, ast.Await):
            return self._resolve_expr(expr.value, ctx)
        if isinstance(expr, ast.NamedExpr):
            return self._resolve_expr(expr.value, ctx)
        return None

    def _annotation_class(self, expr, info: ModuleInfo, self_cls=None):
        """The repo class an annotation names, if exactly one (None otherwise)."""
        if expr is None:
            return None
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            try:
                expr = ast.parse(expr.value, mode="eval").body
            except SyntaxError:
                return None
        if isinstance(expr, ast.Name) and expr.id == "Self":
            return self_cls
        if isinstance(expr, ast.Subscript):
            head = expr.value.attr if isinstance(expr.value, ast.Attribute) else \
                getattr(expr.value, "id", "")
            items = expr.slice.elts if isinstance(expr.slice, ast.Tuple) else [expr.slice]
            if head in ("Optional", "Annotated", "Final", "ClassVar"):
                return self._annotation_class(items[0], info, self_cls)
            if head == "Union":
                real = [i for i in items if not _is_none(i)]
                return self._annotation_class(real[0], info, self_cls) if len(real) == 1 else None
            return None
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            sides = [s for s in (expr.left, expr.right) if not _is_none(s)]
            return self._annotation_class(sides[0], info, self_cls) if len(sides) == 1 else None
        if isinstance(expr, (ast.Name, ast.Attribute)):
            ref = self._resolve_expr(expr, _Ctx(info))
            return ref[1] if ref and ref[0] == "class" else None
        return None

    # -- type inference --------------------------------------------------------

    def _infer_locals(self, scan: _Scan, ctx: _Ctx, params) -> dict:
        """Flow-insensitive local types: a name keeps a type only if every
        assignment to it agrees."""
        types: dict = {}
        for arg in params:
            cls = self._annotation_class(arg.annotation, ctx.info, ctx.cls)
            if cls:
                types[arg.arg] = ("instance", cls)
        ctx.locals = types
        for _ in range(2):  # second pass lets `y = x.make()` use x's type
            seen: dict = {}
            conflicts: set = set()
            for name, value, annotation in scan.assigns:
                if annotation is None and _is_none(value):
                    continue                # `x = None` placeholders are neutral
                ref = None
                if annotation is not None:
                    cls = self._annotation_class(annotation, ctx.info, ctx.cls)
                    ref = ("instance", cls) if cls else None
                if ref is None and value is not None:
                    ref = self._resolve_expr(value, ctx)
                    if ref and ref[0] in ("modprefix", "super"):
                        ref = None
                if name in seen and seen[name] != ref:
                    conflicts.add(name)
                seen[name] = ref
            types = {n: r for n, r in seen.items() if r is not None and n not in conflicts}
            for arg in params:
                cls = self._annotation_class(arg.annotation, ctx.info, ctx.cls)
                if cls and arg.arg not in seen:
                    types[arg.arg] = ("instance", cls)
            ctx.locals = types
        return types

    def _ctx_for(self, fdef, info: ModuleInfo, cls: ClassInfo | None) -> _Ctx:
        key = id(fdef)
        if key in self._ctx_cache:
            return self._ctx_cache[key]
        ctx = _Ctx(info, cls)
        params = [*fdef.args.posonlyargs, *fdef.args.args, *fdef.args.kwonlyargs]
        if cls is not None and params:
            decorators = _decorator_names(fdef)
            if "staticmethod" in decorators:
                pass
            elif "classmethod" in decorators or fdef.name in ("__new__", "__init_subclass__",
                                                               "__class_getitem__"):
                ctx.cls_name = params[0].arg
            else:
                ctx.self_name = params[0].arg
        if key in self._busy:
            return ctx                      # recursion: resolve without locals
        self._busy.add(key)
        try:
            self._infer_locals(self._scan_of(key, [fdef]), ctx, params)
        finally:
            self._busy.discard(key)
        self._ctx_cache[key] = ctx
        return ctx

    def _attr_table(self, cls: ClassInfo) -> dict:
        """attr -> [(value, annotation, fdef_or_None)] for one class, built once:
        class-level ``attr = ...``/``attr: T`` plus ``self.attr = ...`` writes."""
        table = self._attr_tables.get(id(cls))
        if table is not None:
            return table
        table = self._attr_tables[id(cls)] = defaultdict(list)
        for name, value, annotation in self._class_scan(cls).assigns:
            table[name].append((value, annotation, None))
        for method in cls.methods:
            for fdef in self._all_defs[(cls.module, f"{cls.name}.{method}")]:
                scan = self._scan_of(id(fdef), [fdef])
                self_name = _self_param(fdef)
                for base, attr, value, annotation in scan.attr_assigns:
                    if base == self_name:
                        table[attr].append((value, annotation, fdef))
        return table

    def _attr_type(self, cls: ClassInfo, attr: str) -> ClassInfo | None:
        """Infer the class of ``instance.attr`` from assignments in the class
        (nearest class in the MRO wins). Ambiguity means no answer."""
        key = (id(cls), attr)
        if key in self._attr_cache:
            return self._attr_cache[key]
        if key in self._busy:
            return None
        self._busy.add(key)
        found: list = []
        try:
            for c in self._mro_of(cls):
                entries = self._attr_table(c).get(attr)
                if not entries:
                    continue
                info = self.infos[c.module]
                for value, annotation, fdef in entries:
                    if annotation is None and _is_none(value):
                        continue        # `self.x = None` placeholders are neutral
                    hit = self._annotation_class(annotation, info, c)
                    if hit is None and value is not None:
                        ctx = self._ctx_for(fdef, info, c) if fdef else _Ctx(info)
                        ref = self._resolve_expr(value, ctx)
                        hit = ref[1] if ref and ref[0] == "instance" else None
                    found.append(hit)
                if found:
                    break
        finally:
            self._busy.discard(key)
        classes = {id(c): c for c in found if c is not None}
        result = next(iter(classes.values())) if len(classes) == 1 and None not in found else None
        self._attr_cache[key] = result
        return result

    # -- pass 2: edges -----------------------------------------------------------

    def _emit(self, graph, src: str, dst: str, kind: str, strength: str, lineno: int):
        if src == dst or not self.graph.has_node(dst):
            return
        if graph.has_edge(src, dst):
            data = graph.edges[src, dst]
            data["linenos"].append(lineno)
            if KIND_RANK[kind] > KIND_RANK[data["kind"]]:
                data["kind"] = kind
            if STRENGTH_RANK[strength] > STRENGTH_RANK[data["strength"]]:
                data["strength"] = strength
        else:
            graph.add_edge(src, dst, kind=kind, strength=strength, linenos=[lineno])

    def _method_id(self, definer: ClassInfo, name: str) -> str:
        return node_id(definer.module, f"{definer.name}.{name}")

    def _emit_ref(self, graph, src, ref, lineno, *, called: bool):
        """Turn a resolved ref into edges. ``called`` = it's a call site."""
        kind = ref[0]
        strength = "direct" if called else "reference"
        if kind == "function":
            self._emit(graph, src, node_id(ref[1], ref[2]), "call", strength, lineno)
        elif kind == "method":
            definer, name, receiver = ref[1], ref[2], ref[3]
            self._emit(graph, src, self._method_id(definer, name), "call", strength, lineno)
            if receiver is not None:
                overrides = [c for c in self._all_subclasses(receiver)
                             if name in c.methods and c is not definer]
                for c in overrides[:MAX_VIRTUAL_TARGETS]:
                    self._emit(graph, src, self._method_id(c, name), "call", "virtual", lineno)
        elif kind in ("class", "clsvar"):
            cls = ref[1]
            own = src.startswith(cls.id + ".")   # a class's own methods
            if called:
                if not own:
                    self._emit(graph, src, cls.id, "instantiate", "direct", lineno)
                for ctor in ("__init__", "__new__"):
                    definer = self._lookup(cls, ctor)
                    if definer:
                        self._emit(graph, src, self._method_id(definer, ctor),
                                   "call", "direct", lineno)
            elif not own:
                self._emit(graph, src, cls.id, "uses", "reference", lineno)
        elif kind == "instance" and called:
            definer = self._lookup(ref[1], "__call__")
            if definer:
                self._emit(graph, src, self._method_id(definer, "__call__"),
                           "call", "direct", lineno)

    def _walk(self, graph, src: str, scan: _Scan, ctx: _Ctx):
        """Resolve every call and value reference recorded in ``scan``."""
        for call in scan.calls:
            ref = self._resolve_expr(call.func, ctx)
            if ref is not None:
                self._emit_ref(graph, src, ref, call.lineno, called=True)
        for node in scan.refs:
            self._reference(graph, src, node, ctx)

    def _reference(self, graph, src, node, ctx):
        ref = self._resolve_expr(node, ctx)
        if ref is not None and ref[0] in ("function", "method", "class", "clsvar"):
            self._emit_ref(graph, src, ref, node.lineno, called=False)
            return
        # `Config.DEFAULT`: an unresolvable attribute of a class still uses it.
        cur = node.value if isinstance(node, ast.Attribute) else None
        while isinstance(cur, (ast.Attribute, ast.Name)):
            ref = self._resolve_expr(cur, ctx)
            if ref is not None and ref[0] == "class":
                self._emit_ref(graph, src, ref, node.lineno, called=False)
                return
            cur = cur.value if isinstance(cur, ast.Attribute) else None

    def _resolve_all(self):
        for info in self.infos.values():
            for name in info.functions:
                for fdef in self._all_defs[(info.module, name)]:
                    ctx = self._ctx_for(fdef, info, None)
                    self._walk(self.graph, node_id(info.module, name),
                               self._scan_of(id(fdef), [fdef]), ctx)
            for cls in info.classes.values():
                for name in cls.methods:
                    for fdef in self._all_defs[(info.module, f"{cls.name}.{name}")]:
                        ctx = self._ctx_for(fdef, info, cls)
                        self._walk(self.graph, self._method_id(cls, name),
                                   self._scan_of(id(fdef), [fdef]), ctx)
                # Class keywords/decorators/body assignments run at import time.
                self._walk(self.module_graph, info.module, self._class_scan(cls), _Ctx(info))
            # Module-level code (scripts, registrations, `if __name__ == "__main__"`).
            self._walk(self.module_graph, info.module, self._module_scan(info), _Ctx(info))

    def build(self) -> nx.DiGraph:
        self._collect()
        self._resolve_bases()
        # Anything inferred while bases were still unknown may be incomplete.
        for cache in (self._mro, self._ctx_cache, self._globals, self._attr_cache):
            cache.clear()
        self._resolve_all()
        self.graph.graph["module_edges"] = list(self.module_graph.edges(data=True))
        return self.graph


def _import_time_roots(body) -> list:
    """Expressions/statements that execute at import time, excluding def bodies.
    Compound blocks contribute only their headers; ``_toplevel`` yields their
    inner statements separately, so nothing is walked twice."""
    roots = []
    for stmt in _toplevel(body):
        if isinstance(stmt, (*_FUNC_TYPES, ast.ClassDef)):
            continue
        if isinstance(stmt, ast.If):
            roots.append(stmt.test)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            roots.extend(item.context_expr for item in stmt.items)
        elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
            continue
        else:
            roots.append(stmt)
    return roots


def _self_param(fdef) -> str | None:
    """Name of the parameter bound to the instance, if ``fdef`` is an instance
    method (i.e. not a static/class method)."""
    params = [*fdef.args.posonlyargs, *fdef.args.args]
    if not params:
        return None
    decorators = _decorator_names(fdef)
    if decorators & {"staticmethod", "classmethod"} or fdef.name in (
            "__new__", "__init_subclass__", "__class_getitem__"):
        return None
    return params[0].arg


def _is_overload(fdef) -> bool:
    return "overload" in _decorator_names(fdef)


def _is_none(expr) -> bool:
    return isinstance(expr, ast.Constant) and expr.value is None


def build_call_graph(repo, exclude=()) -> nx.DiGraph:
    """Build the function-level call graph for a repo (path or ``RepoIndex``).

    Module-level dependencies (code that runs at import time) are returned in
    ``graph.graph["module_edges"]`` as ``(module, target, data)`` triples, since
    modules are not nodes of the call graph itself.
    """
    return CallGraphBuilder(repo, exclude).build()
