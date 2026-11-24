"""Tests for the call graph's name/type resolution.

Each case is a tiny hand-written repo where the correct edge is obvious by
reading it, plus a "must not resolve" counterpart where guessing would invent an
edge. Edge convention: ``A -> B`` means A depends on B.
"""

from __future__ import annotations

import textwrap

from pyimpact.call_graph import build_call_graph


def make_repo(tmp_path, files: dict[str, str]) -> str:
    for rel, source in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(tmp_path)


def graph(tmp_path, files):
    return build_call_graph(make_repo(tmp_path, files))


def edge(g, a, b):
    return g.edges[a, b] if g.has_edge(a, b) else None


# --- constructors & classes ------------------------------------------------

def test_constructor_links_class_and_init(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Repo:
            def __init__(self, url):
                self.url = url

        def make():
            return Repo("x")
    """})
    assert edge(g, "m:make", "m:Repo")["kind"] == "instantiate"
    assert edge(g, "m:make", "m:Repo.__init__")["kind"] == "call"


def test_inherited_init_is_resolved(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Base:
            def __init__(self):
                pass

        class Child(Base):
            pass

        def make():
            return Child()
    """})
    assert g.has_edge("m:make", "m:Base.__init__")
    assert g.nodes["m:Child"]["bases"] == ["m:Base"]


def test_classmethod_call_on_class(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Config:
            @classmethod
            def load(cls):
                return cls()

        def boot():
            return Config.load()
    """})
    assert g.has_edge("m:boot", "m:Config.load")


def test_class_reference_is_a_uses_edge(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Token:
            pass

        def check(x):
            return isinstance(x, Token)
    """})
    assert edge(g, "m:check", "m:Token")["kind"] == "uses"


# --- self, inheritance, super, virtual dispatch -------------------------------

def test_self_call_to_inherited_method(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Base:
            def helper(self):
                return 1

        class Child(Base):
            def run(self):
                return self.helper()
    """})
    assert g.has_edge("m:Child.run", "m:Base.helper")


def test_super_call(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Base:
            def save(self):
                return 1

        class Child(Base):
            def save(self):
                return super().save()
    """})
    assert g.has_edge("m:Child.save", "m:Base.save")


def test_virtual_dispatch_to_override(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Handler:
            def handle(self):
                return self.process()

            def process(self):
                return 0

        class JsonHandler(Handler):
            def process(self):
                return 1
    """})
    assert edge(g, "m:Handler.handle", "m:Handler.process")["strength"] == "direct"
    assert edge(g, "m:Handler.handle", "m:JsonHandler.process")["strength"] == "virtual"


def test_cls_in_classmethod_resolves_like_self(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Model:
            @classmethod
            def create(cls):
                return cls.validate()

            @classmethod
            def validate(cls):
                return True
    """})
    assert g.has_edge("m:Model.create", "m:Model.validate")


# --- local / attribute / annotation inference ------------------------------------

def test_with_statement_binding(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Session:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def get(self):
                return 1

        def fetch():
            with Session() as s:
                return s.get()
    """})
    assert g.has_edge("m:fetch", "m:Session.get")


def test_annotated_parameter(tmp_path):
    g = graph(tmp_path, {"m.py": """
        from typing import Optional

        class Db:
            def query(self):
                return []

        def a(db: Db):
            return db.query()

        def b(db: Optional[Db] = None):
            return db.query()

        def c(db: "Db | None"):
            return db.query()
    """})
    for fn in ("m:a", "m:b", "m:c"):
        assert g.has_edge(fn, "m:Db.query"), fn


def test_return_annotation_propagates(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Client:
            def send(self):
                return 1

        def get_client() -> Client:
            return Client()

        def go():
            return get_client().send()
    """})
    assert g.has_edge("m:go", "m:Client.send")


def test_instance_attribute_type(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Router:
            def match(self, path):
                return path

        class App:
            def __init__(self):
                self.router = None          # placeholder, ignored
                self.router = Router()

            def handle(self, path):
                return self.router.match(path)
    """})
    assert g.has_edge("m:App.handle", "m:Router.match")


def test_conflicting_assignments_do_not_guess(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class A:
            def run(self):
                pass

        class B:
            def run(self):
                pass

        def f(flag):
            x = A()
            if flag:
                x = B()
            return x.run()
    """})
    assert not g.has_edge("m:f", "m:A.run")
    assert not g.has_edge("m:f", "m:B.run")


def test_module_level_instance(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Registry:
            def register(self, fn):
                return fn

        registry = Registry()

        def plugin():
            return registry.register(plugin)
    """})
    assert g.has_edge("m:plugin", "m:Registry.register")


# --- imports, re-exports, getattr, references ----------------------------------

def test_reexport_through_package_init(tmp_path):
    g = graph(tmp_path, {
        "lib/__init__.py": "from .sessions import Session\n",
        "lib/sessions.py": """
            class Session:
                def get(self):
                    return 1
        """,
        "app.py": """
            from lib import Session

            def run():
                return Session().get()
        """,
    })
    assert g.has_edge("app:run", "lib.sessions:Session.get")
    assert g.has_edge("app:run", "lib.sessions:Session")


def test_src_layout_absolute_imports(tmp_path):
    g = graph(tmp_path, {
        "src/pkg/__init__.py": "",
        "src/pkg/core.py": "def work():\n    return 1\n",
        "tests/test_core.py": """
            from pkg.core import work

            def test_work():
                assert work() == 1
        """,
    })
    assert g.has_edge("tests.test_core:test_work", "pkg.core:work")


def test_getattr_with_literal(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Api:
            def ping(self):
                return 1

            def call(self):
                return getattr(self, "ping")()
    """})
    assert g.has_edge("m:Api.call", "m:Api.ping")


def test_getattr_with_computed_name_is_not_guessed(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Api:
            def ping(self):
                return 1

            def call(self, name):
                return getattr(self, name)()
    """})
    assert not g.has_edge("m:Api.call", "m:Api.ping")


def test_callback_reference(tmp_path):
    g = graph(tmp_path, {"m.py": """
        import threading

        def work():
            return 1

        def start():
            t = threading.Thread(target=work)
            t.start()
    """})
    assert edge(g, "m:start", "m:work")["strength"] == "reference"


def test_decorator_is_a_dependency(tmp_path):
    g = graph(tmp_path, {"m.py": """
        def cached(fn):
            return fn

        @cached
        def compute():
            return 42
    """})
    assert g.has_edge("m:compute", "m:cached")


def test_overload_stubs_do_not_hide_implementation(tmp_path):
    g = graph(tmp_path, {"m.py": """
        import typing

        def helper():
            return 1

        @typing.overload
        def main(x: int) -> int: ...
        @typing.overload
        def main(x: str) -> str: ...
        def main(x):
            return helper()
    """})
    assert g.has_edge("m:main", "m:helper")
    assert g.nodes["m:main"]["lineno"] == 11   # the real implementation
    assert len(g.nodes["m:main"]["spans"]) == 3  # stubs + implementation


def test_module_level_calls_are_recorded(tmp_path):
    g = graph(tmp_path, {"m.py": """
        def main():
            return 0

        if __name__ == "__main__":
            main()
    """})
    module_edges = {(u, v) for u, v, _ in g.graph["module_edges"]}
    assert ("m", "m:main") in module_edges


def test_class_methods_and_nodes_carry_metadata(tmp_path):
    g = graph(tmp_path, {"m.py": """
        class Service:
            '''Talks to the backend.'''

            async def fetch(self, url: str, retries: int = 3):
                '''Fetch a URL.'''
                return url
    """})
    node = g.nodes["m:Service.fetch"]
    assert node["kind"] == "method" and node["is_async"]
    assert node["class_id"] == "m:Service"
    assert "url: str" in node["signature"]
    assert node["doc"] == "Fetch a URL."
    assert g.nodes["m:Service"]["kind"] == "class"
